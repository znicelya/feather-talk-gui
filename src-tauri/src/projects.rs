//! Project locations are app-owned; project.json remains the worker's index.
use feathertalk_core::*;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    sync::Mutex,
};

pub struct ProjectStore {
    index: PathBuf,
    mutation: Mutex<()>,
}

impl ProjectStore {
    pub fn new(index: PathBuf) -> Result<Self> {
        let parent = index
            .parent()
            .filter(|p| index.is_absolute() && !p.as_os_str().is_empty())
            .ok_or_else(|| {
                project_error(
                    ErrorCode::InvalidConfig,
                    &index,
                    "The project index requires an absolute path",
                )
            })?;
        fs::create_dir_all(parent).map_err(|e| path_error(&index, e))?;
        let store = Self {
            index,
            mutation: Mutex::new(()),
        };
        store.locations()?;
        Ok(store)
    }

    fn locations(&self) -> Result<BTreeMap<ProjectId, PathBuf>> {
        match fs::read(&self.index) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| project_error(ErrorCode::ResourceCorrupt, &self.index, e)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(BTreeMap::new()),
            Err(e) => Err(path_error(&self.index, e)),
        }
    }

    fn register(&self, id: ProjectId, root: &Path) -> Result<()> {
        let _guard = self
            .mutation
            .lock()
            .map_err(|e| project_error(ErrorCode::Internal, &self.index, e))?;
        let mut locations = self.locations()?;
        if locations
            .iter()
            .any(|(other, path)| *other != id && path == root)
        {
            return Err(project_error(
                ErrorCode::ModelMismatch,
                root,
                "Directory is already registered to another project",
            ));
        }
        if locations.get(&id).is_some_and(|path| path == root) {
            return Ok(());
        }
        locations.insert(id, root.to_path_buf());
        publish(&self.index, &locations, false)
    }

    pub fn create(&self, input: NewProject) -> Result<ProjectSummary> {
        input.validate()?;
        if !input.directory.is_absolute() || !input.source_video.is_absolute() {
            return Err(project_error(
                ErrorCode::InvalidConfig,
                &input.directory,
                "Select absolute project and source paths",
            ));
        }
        let source = regular_file(&input.source_video)?;
        let (sha256, size_bytes) = fingerprint(&source)?;
        fs::create_dir_all(&input.directory).map_err(|e| path_error(&input.directory, e))?;
        let root = project_root(&input.directory)?;
        let manifest = ProjectManifest {
            schema_version: 1,
            id: ProjectId::new(),
            name: input.name.trim().into(),
            source_video: SourceVideo {
                path: source.strip_prefix(&root).unwrap_or(&source).to_path_buf(),
                sha256,
                size_bytes,
            },
            preprocessing: None,
            model_identity: None,
            checkpoints: vec![],
            metrics_path: "metrics.jsonl".into(),
        };
        publish(&root.join("project.json"), &manifest, true)?;
        self.register(manifest.id, &root)?;
        Ok(summary(&root, &manifest))
    }

    pub fn open(&self, directory: &Path) -> Result<ProjectSummary> {
        let (root, manifest) = self.inspect(directory)?;
        self.register_open(&root, &manifest)
    }

    pub(crate) fn inspect(&self, directory: &Path) -> Result<(PathBuf, ProjectManifest)> {
        let root = project_root(directory)?;
        let manifest = read_manifest(&root, None)?;
        verify_source(&root, &manifest)?;
        Ok((root, manifest))
    }

    pub(crate) fn register_open(
        &self,
        root: &Path,
        manifest: &ProjectManifest,
    ) -> Result<ProjectSummary> {
        self.register(manifest.id, root)?;
        Ok(summary(root, manifest))
    }

    pub fn list(&self) -> Result<Vec<ProjectSummary>> {
        self.locations()?
            .into_iter()
            .map(|(id, dir)| {
                let root = checked_location(&dir)?;
                Ok(summary(&root, &read_manifest(&root, Some(id))?))
            })
            .collect()
    }

    pub fn get(&self, id: ProjectId) -> Result<ProjectManifest> {
        self.load(id).map(|(_, manifest)| manifest)
    }

    pub fn directory(&self, id: ProjectId) -> Result<PathBuf> {
        let locations = self.locations()?;
        let directory = locations.get(&id).ok_or_else(|| {
            project_error(
                ErrorCode::InvalidConfig,
                &self.index,
                format!("Unknown project {id}"),
            )
        })?;
        checked_location(directory)
    }

    fn load(&self, id: ProjectId) -> Result<(PathBuf, ProjectManifest)> {
        let root = self.directory(id)?;
        let manifest = read_manifest(&root, Some(id))?;
        Ok((root, manifest))
    }

    pub fn verify_source(&self, id: ProjectId) -> Result<()> {
        let (root, manifest) = self.load(id)?;
        verify_source(&root, &manifest)
    }

    pub(crate) fn bind(&self, id: ProjectId) -> Result<ProjectBinding> {
        let (directory, manifest) = self.load(id)?;
        verify_source(&directory, &manifest)?;
        Ok(ProjectBinding {
            project_id: id,
            directory,
        })
    }
}

fn summary(root: &Path, manifest: &ProjectManifest) -> ProjectSummary {
    ProjectSummary {
        id: manifest.id,
        name: manifest.name.clone(),
        directory: root.into(),
        source_video: root.join(&manifest.source_video.path),
        latest_model: manifest
            .checkpoints
            .last()
            .map(|entry| entry.model_ref.clone()),
    }
}

fn project_root(path: &Path) -> Result<PathBuf> {
    let root = path.canonicalize().map_err(|e| path_error(path, e))?;
    if !path.is_absolute() || !root.is_dir() {
        return Err(project_error(
            ErrorCode::InvalidConfig,
            path,
            "Expected an absolute project directory",
        ));
    }
    Ok(root)
}

fn checked_location(path: &Path) -> Result<PathBuf> {
    let root = project_root(path)?;
    if root != path {
        return Err(project_error(
            ErrorCode::ModelMismatch,
            path,
            "Registered project now resolves to a different directory",
        ));
    }
    Ok(root)
}

fn read_manifest(root: &Path, id: Option<ProjectId>) -> Result<ProjectManifest> {
    let path = root.join("project.json");
    let canonical = regular_file(&path)?;
    if canonical != path {
        return Err(project_error(
            ErrorCode::ResourceCorrupt,
            &path,
            "Project manifest must be a file in its own directory",
        ));
    }
    let bytes = fs::read(&path).map_err(|e| path_error(&path, e))?;
    let manifest: ProjectManifest = serde_json::from_slice(&bytes)
        .map_err(|e| project_error(ErrorCode::ResourceCorrupt, &path, e))?;
    if manifest.schema_version != 1
        || id.is_some_and(|id| id != manifest.id)
        || manifest.name.trim().is_empty()
    {
        return Err(project_error(
            ErrorCode::ModelMismatch,
            &path,
            "Schema, name or project identity does not match",
        ));
    }
    validate_owned(root, &manifest.metrics_path, false)?;
    if let Some(prepared) = &manifest.preprocessing {
        validate_owned(root, &prepared.cache_directory, true)?;
        if prepared.source_sha256 != manifest.source_video.sha256 {
            return Err(project_error(
                ErrorCode::ModelMismatch,
                &path,
                "Preprocessing source identity does not match",
            ));
        }
    }
    let mut checkpoints = HashSet::new();
    for entry in &manifest.checkpoints {
        let r = &entry.model_ref;
        if r.project_id != manifest.id
            || !checkpoints.insert((&r.run_id, &r.checkpoint_id))
            || r.checkpoint_id.is_empty()
            || entry
                .relative_path
                .file_name()
                .is_none_or(|p| p != "manifest.json")
        {
            return Err(project_error(
                ErrorCode::ModelMismatch,
                &path,
                "Checkpoint index contains a foreign or duplicate reference",
            ));
        }
        validate_owned(root, &entry.relative_path, true)?;
    }
    if !manifest.source_video.path.is_absolute() {
        validate_owned(root, &manifest.source_video.path, true)?;
    }
    Ok(manifest)
}

fn validate_owned(root: &Path, relative: &Path, required: bool) -> Result<()> {
    let path = root.join(relative);
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(project_error(
            ErrorCode::ResourceCorrupt,
            &path,
            "Expected a project-relative path",
        ));
    }
    let parent = path.parent().unwrap();
    let candidate = match fs::symlink_metadata(&path) {
        Ok(_) => path.canonicalize().map_err(|e| path_error(&path, e))?,
        Err(e) if !required && e.kind() == io::ErrorKind::NotFound => {
            parent.canonicalize().map_err(|e| path_error(parent, e))?
        }
        Err(e) => return Err(path_error(&path, e)),
    };
    if !candidate.starts_with(root) {
        return Err(project_error(
            ErrorCode::ResourceCorrupt,
            &path,
            "Path escapes the project",
        ));
    }
    Ok(())
}

fn verify_source(root: &Path, manifest: &ProjectManifest) -> Result<()> {
    let path = root.join(&manifest.source_video.path);
    let source = regular_file(&path)?;
    let (sha256, size) = fingerprint(&source)?;
    if size != manifest.source_video.size_bytes || sha256 != manifest.source_video.sha256 {
        return Err(project_error(
            ErrorCode::ModelMismatch,
            &path,
            "Source video differs from the project's original contents",
        ));
    }
    Ok(())
}

pub(crate) fn regular_file(path: &Path) -> Result<PathBuf> {
    let real = path.canonicalize().map_err(|e| path_error(path, e))?;
    if !real.is_file() {
        return Err(project_error(
            ErrorCode::InvalidConfig,
            path,
            "Expected a regular file",
        ));
    }
    Ok(real)
}

fn fingerprint(path: &Path) -> Result<(String, u64)> {
    let mut file = fs::File::open(path).map_err(|e| path_error(path, e))?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 65536];
    let mut size = 0u64;
    loop {
        let count = file.read(&mut buffer).map_err(|e| path_error(path, e))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
        size += count as u64;
    }
    if size == 0 {
        return Err(project_error(
            ErrorCode::InvalidConfig,
            path,
            "Source video is empty",
        ));
    }
    Ok((format!("{:x}", digest.finalize()), size))
}

fn publish(path: &Path, value: &impl serde::Serialize, create: bool) -> Result<()> {
    let mut temp =
        tempfile::NamedTempFile::new_in(path.parent().unwrap()).map_err(|e| path_error(path, e))?;
    serde_json::to_writer_pretty(&mut temp, value)
        .map_err(|e| project_error(ErrorCode::Internal, path, e))?;
    temp.flush()
        .and_then(|()| temp.as_file().sync_all())
        .map_err(|e| path_error(path, e))?;
    if create {
        temp.persist_noclobber(path)
    } else {
        temp.persist(path)
    }
    .map_err(|e| path_error(path, e.error))?;
    Ok(())
}

fn path_error(path: &Path, error: io::Error) -> AppError {
    let kind = match error.kind() {
        io::ErrorKind::NotFound => ErrorCode::ResourceMissing,
        io::ErrorKind::PermissionDenied => ErrorCode::PermissionDenied,
        io::ErrorKind::StorageFull => ErrorCode::DiskFull,
        io::ErrorKind::AlreadyExists => ErrorCode::InvalidConfig,
        _ => ErrorCode::ResourceCorrupt,
    };
    project_error(kind, path, error)
}

fn project_error(code: ErrorCode, path: &Path, message: impl std::fmt::Display) -> AppError {
    let mut error = AppError::new(code, "project", message.to_string());
    error.evidence.push(path.display().to_string());
    error.actions.push("Open the current project directory if it moved, or restore the original project files and source video.".into());
    error
}
