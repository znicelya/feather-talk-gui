//! Bind an accepted job to verified installed resources without loading Burn in the host.
use super::{Component, ResourceManager, catalog, compatible, fail, inventory};
use crate::worker_host::WorkerHost;
use feathertalk_core::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Component as PathComponent, Path, PathBuf},
};

const PREPROCESS_MODELS: [&str; 4] = [
    "feather_hubert",
    "scrfd_2_5g",
    "pfld_ghost_one",
    "mean_face",
];

#[derive(Clone, Copy)]
pub(super) struct WorkerRole {
    pub id: &'static str,
    kind: &'static str,
    exact_target: bool,
    executables: &'static [&'static str],
}

const WORKER: WorkerRole = WorkerRole {
    id: "worker",
    kind: "worker",
    exact_target: true,
    executables: &["feathertalk-worker"],
};
const PROBE: WorkerRole = WorkerRole {
    id: "probe",
    kind: "probe",
    exact_target: true,
    executables: &["feathertalk-probe"],
};
const MEDIA: WorkerRole = WorkerRole {
    id: "media",
    kind: "media",
    exact_target: false,
    executables: &["ffmpeg", "ffprobe"],
};
const ASSETS: WorkerRole = WorkerRole {
    id: "public-assets",
    kind: "model_bundle",
    exact_target: false,
    executables: &[],
};
const CUDA: WorkerRole = WorkerRole {
    id: "cuda-math",
    kind: "cuda_runtime",
    exact_target: true,
    executables: &[],
};
const LIBDEVICE: &str = "nvvm/libdevice/libdevice.10.bc";

pub(super) fn required_roles(target: &TargetPlatform) -> impl Iterator<Item = WorkerRole> {
    [WORKER, PROBE, MEDIA, ASSETS]
        .into_iter()
        .chain((target.backend == "cuda").then_some(CUDA))
}

impl WorkerRole {
    pub fn validate(self, component: &Component, target: &TargetPlatform) -> Result<()> {
        if component.kind != self.kind
            || !compatible(component, target)
            || (self.exact_target && component.target.as_ref() != Some(target))
        {
            return Err(AppError::new(
                ErrorCode::UnsupportedBackend,
                "resources",
                format!(
                    "Installed {} does not match the selected platform/backend",
                    self.id
                ),
            ));
        }
        for name in self.executables {
            executable_path(component, name, target)?;
        }
        if self.id == ASSETS.id && (component.target.is_some() || !has_bundle_layout(component)) {
            return Err(fail(
                "Installed asset bundle lacks the FeatherTalk models or mean face",
            ));
        }
        if self.id == CUDA.id && !component.files().contains_key(Path::new(LIBDEVICE)) {
            return Err(missing("The installed CUDA component lacks libdevice"));
        }
        Ok(())
    }
}

fn executable_path<'a>(
    component: &'a Component,
    name: &str,
    target: &TargetPlatform,
) -> Result<&'a Path> {
    let name = if target.os == "windows" {
        format!("{name}.exe")
    } else {
        name.to_owned()
    };
    let mut found = component
        .executables
        .iter()
        .filter(|p| p.file_name().is_some_and(|file| file == name.as_str()));
    let path = found
        .next()
        .ok_or_else(|| missing(format!("Missing executable {name}")))?;
    if found.next().is_some() {
        return Err(fail(format!("Ambiguous executable {name}")));
    }
    Ok(path)
}

#[derive(Debug)]
pub struct InstalledWorker {
    binary: PathBuf,
    context: WorkerContext,
}

impl InstalledWorker {
    pub fn spawn(self) -> Result<WorkerHost> {
        WorkerHost::spawn(&self.binary, self.context)
    }
}

struct Installed {
    root: PathBuf,
    component: Component,
}

impl Installed {
    fn executable(&self, name: &str, target: &TargetPlatform) -> Result<PathBuf> {
        let path = executable_path(&self.component, name, target)?;
        inventory::contained(&self.root, &self.root.join(path))
    }

    fn runtime(self) -> RuntimeComponent {
        let files = self
            .component
            .files()
            .into_iter()
            .map(|(path, file)| RuntimeFile {
                path: self.root.join(path),
                sha256: file.sha256,
            })
            .collect();
        RuntimeComponent {
            target: self.component.target,
            pin: self.component.pin,
            root: self.root,
            files,
        }
    }
}

impl ResourceManager {
    /// The job's checkpoint overrides only the selected asset bundle. Runtime pins stay explicit.
    pub fn bind_worker(
        &self,
        selection: &ResourceSelection,
        mut project: ProjectBinding,
        job: &JobSpec,
    ) -> Result<InstalledWorker> {
        let target = &selection.target;
        catalog::validate_target(target)?;
        if target.os != std::env::consts::OS || target.architecture != std::env::consts::ARCH {
            return Err(AppError::new(
                ErrorCode::UnsupportedBackend,
                "resources",
                "Cannot launch a worker for another platform",
            ));
        }
        let (project_id, reference) = match job {
            JobSpec::Train(request) => (request.project_id, request.resume.as_ref()),
            JobSpec::Generate(request) => (request.project_id, Some(&request.model_ref)),
        };
        if !project.directory.is_absolute() || project.project_id != project_id {
            return Err(checkpoint_error(
                "Worker project binding differs from the accepted job",
            ));
        }
        project.directory = project.directory.canonicalize().map_err(fail)?;
        let manifest_path =
            inventory::contained(&project.directory, &project.directory.join("project.json"))?;
        let manifest: ProjectManifest =
            serde_json::from_slice(&fs::read(manifest_path).map_err(fail)?).map_err(fail)?;
        if manifest.schema_version != 1 || manifest.id != project_id {
            return Err(checkpoint_error(
                "Project manifest differs from the accepted project",
            ));
        }
        let dependencies = reference
            .map(|reference| {
                checkpoint_dependencies(
                    &project.directory,
                    &manifest,
                    reference,
                    matches!(job, JobSpec::Generate(_)),
                )
            })
            .transpose()?;

        let mut pins = BTreeMap::new();
        for pin in &selection.resources {
            catalog::token(&pin.id)?;
            catalog::token(&pin.version)?;
            if pins.insert(pin.id.as_str(), pin).is_some() {
                return Err(fail(format!(
                    "More than one resource selected for {}",
                    pin.id
                )));
            }
        }
        let selected = |role: WorkerRole| -> Result<Installed> {
            let id = role.id;
            let pin = pins
                .get(id)
                .ok_or_else(|| missing(format!("Select an installed {id} component")))?;
            let path = self.root.join(&pin.id).join(&pin.version);
            if !path.exists() {
                return Err(missing(format!(
                    "Install {}/{} for backend {}",
                    pin.id, pin.version, target.backend
                )));
            }
            let c = self.find(pin)?;
            role.validate(&c, target)?;
            let root = inventory::contained(&self.root, &path)?;
            self.verify_installed(&root, &c)?;
            Ok(Installed { root, component: c })
        };

        let worker = selected(WORKER)?;
        let probe = selected(PROBE)?;
        let media = selected(MEDIA)?;
        let cuda = if target.backend == "cuda" {
            Some(selected(CUDA)?)
        } else {
            None
        };
        let bundle = if let Some(dependencies) = dependencies {
            self.resolve_model_generation(target, |path, candidate| {
                if candidate.pin.id != "public-assets"
                    || candidate.kind != "model_bundle"
                    || candidate.target.is_some()
                    || !has_bundle_layout(candidate)
                {
                    return Ok(None);
                }
                let files = candidate.files();
                for wanted in &dependencies {
                    let file = nested_manifest(&wanted.kind);
                    inventory::verify_file(path, &file, &files[&file])?;
                    let actual: AssetManifest =
                        serde_json::from_slice(&fs::read(path.join(file)).map_err(fail)?)
                            .map_err(fail)?;
                    if actual.identity != *wanted {
                        return Ok(None);
                    }
                }
                Ok(Some(Installed {
                    root: path.to_path_buf(),
                    component: candidate.clone(),
                }))
            })?
        } else {
            selected(ASSETS)?
        };

        let binary = worker.executable("feathertalk-worker", target)?;
        let mut context = WorkerContext {
            protocol_version: PROTOCOL_VERSION,
            projects: vec![project],
            target: target.clone(),
            assets: bundle.root.clone(),
            ffmpeg: media.executable("ffmpeg", target)?,
            ffprobe: media.executable("ffprobe", target)?,
            probe_runner: probe.executable("feathertalk-probe", target)?,
            cuda_root: None,
            components: vec![
                worker.runtime(),
                probe.runtime(),
                media.runtime(),
                bundle.runtime(),
            ],
        };
        if let Some(cuda) = cuda {
            context.cuda_root = Some(cuda.root.clone());
            context.components.push(cuda.runtime());
        }
        Ok(InstalledWorker { binary, context })
    }
}

fn nested_manifest(kind: &str) -> PathBuf {
    PathBuf::from(kind).join("manifest.json")
}

fn has_bundle_layout(candidate: &Component) -> bool {
    PREPROCESS_MODELS.into_iter().chain(["vgg19"]).all(|kind| {
        candidate
            .manifest
            .relative_files
            .contains(&nested_manifest(kind))
    })
}

// The host only reads the resource projection. The engine still validates and restores the full state.
#[derive(Deserialize)]
struct CheckpointManifest {
    metadata: CheckpointResources,
}

#[derive(Deserialize)]
struct CheckpointResources {
    schema_version: u32,
    project_id: ProjectId,
    run_id: TrainingRunId,
    global_step: u64,
    preprocessing: PreprocessingIdentity,
    vgg_identity: Option<ModelIdentity>,
}

fn checkpoint_dependencies(
    root: &Path,
    project: &ProjectManifest,
    reference: &ModelRef,
    generating: bool,
) -> Result<Vec<ModelIdentity>> {
    let mut entries = project
        .checkpoints
        .iter()
        .filter(|e| &e.model_ref == reference);
    let entry = entries
        .next()
        .ok_or_else(|| checkpoint_error("Checkpoint is not committed in this project"))?;
    if project.id != reference.project_id
        || entries.next().is_some()
        || entry
            .relative_path
            .file_name()
            .is_none_or(|name| name != "manifest.json")
        || entry
            .relative_path
            .components()
            .any(|c| !matches!(c, PathComponent::Normal(_)))
    {
        return Err(checkpoint_error(
            "Checkpoint reference is ambiguous or has an invalid path",
        ));
    }
    let path = inventory::contained(root, &root.join(&entry.relative_path))?;
    let data = fs::read(path).map_err(fail)?;
    if data.len() as u64 != entry.size_bytes
        || format!("{:x}", Sha256::digest(&data)) != entry.sha256
    {
        return Err(checkpoint_error(
            "Committed checkpoint manifest failed integrity verification",
        ));
    }
    let CheckpointManifest { metadata } = serde_json::from_slice(&data).map_err(fail)?;
    let mut identities = metadata.preprocessing.component_identities;
    if metadata.schema_version != 2
        || metadata.project_id != project.id
        || metadata.run_id != reference.run_id
        || metadata.global_step != entry.step
        || metadata.preprocessing.source_sha256 != project.source_video.sha256
        || identities.len() != PREPROCESS_MODELS.len()
        || !PREPROCESS_MODELS
            .iter()
            .all(|kind| identities.iter().any(|identity| &identity.kind == kind))
    {
        return Err(checkpoint_error(
            "Checkpoint metadata does not match its project or preprocessing dependencies",
        ));
    }
    if generating && entry.step == 0 {
        let mut error = AppError::new(
            ErrorCode::InvalidConfig,
            "generate",
            "Train the character before generating video",
        );
        error.actions.push(
            "Select a committed checkpoint with at least one completed training step.".into(),
        );
        return Err(error);
    }
    match metadata.vgg_identity {
        Some(identity) if identity.kind == "vgg19" => identities.push(identity),
        None if entry.step == 0 => {}
        _ => {
            return Err(checkpoint_error(
                "Checkpoint is missing its verified VGG dependency",
            ));
        }
    }
    Ok(identities)
}

fn missing(message: impl Into<String>) -> AppError {
    let mut error = AppError::new(ErrorCode::ResourceMissing, "resources", message);
    error
        .actions
        .push("Install the verified resources for the selected backend or checkpoint.".into());
    error
}

fn checkpoint_error(message: &str) -> AppError {
    AppError::new(ErrorCode::ModelMismatch, "checkpoint", message)
}
