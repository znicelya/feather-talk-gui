//! Explicit resource preparation. Inventory and identity resolution never use HTTP.
pub mod catalog;
mod download;
mod installed_worker;
mod inventory;
pub use catalog::*;
use feathertalk_core::*;
use fs2::FileExt;
pub use installed_worker::InstalledWorker;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};
pub(crate) fn fail(e: impl std::fmt::Display) -> AppError {
    AppError::new(ErrorCode::ResourceCorrupt, "resources", e.to_string())
}
pub struct ResourceManager {
    root: PathBuf,
    catalog: Catalog,
    mutation: Mutex<()>,
}
impl ResourceManager {
    pub fn new(root: PathBuf, catalog: Catalog) -> Result<Self> {
        catalog.validate()?;
        fs::create_dir_all(&root).map_err(fail)?;
        let root = root.canonicalize().map_err(fail)?;
        Ok(Self {
            root,
            catalog,
            mutation: Mutex::new(()),
        })
    }
    fn cache_lock(&self) -> Result<fs::File> {
        let path = self.root.join(".lock");
        let f = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(fail)?;
        inventory::contained(&self.root, &path)?;
        f.try_lock_exclusive().map_err(|e| {
            AppError::new(
                ErrorCode::Busy,
                "resources",
                format!("Another resource operation owns this cache: {e}"),
            )
        })?;
        Ok(f)
    }
    fn destination(&self, c: &Component) -> PathBuf {
        self.root.join(&c.pin.id).join(&c.pin.version)
    }
    fn find(&self, pin: &ResourcePin) -> Result<Component> {
        catalog::token(&pin.id)?;
        catalog::token(&pin.version)?;
        if let Some(c) = self.catalog.components.iter().find(|c| &c.pin == pin) {
            return Ok(c.clone());
        }
        let path = self.root.join(&pin.id).join(&pin.version);
        inventory::contained(&self.root, &path)?;
        let c: Component =
            serde_json::from_slice(&fs::read(path.join(".component.json")).map_err(fail)?)
                .map_err(fail)?;
        c.validate()?;
        if &c.pin != pin {
            return Err(fail("installed pin mismatch"));
        }
        Ok(c)
    }
    fn status(&self, pin: &ResourcePin, target: &TargetPlatform) -> ResourceStatus {
        let mut status = ResourceStatus {
            resource: pin.clone(),
            identity: None,
            path: None,
            ready: false,
            reason: None,
        };
        let check = (|| {
            let c = self.find(pin)?;
            status.identity = Some(c.manifest.identity.clone());
            if !compatible(&c, target) {
                return Err(fail("resource target/backend incompatible with selection"));
            }
            let path = self.destination(&c);
            inventory::contained(&self.root, &path)?;
            self.verify_installed(&path, &c)?;
            status.path = Some(path);
            Ok(())
        })();
        match check {
            Ok(()) => status.ready = true,
            Err(e) => status.reason = Some(e.to_string()),
        }
        status
    }
    fn verify_installed(&self, path: &Path, c: &Component) -> Result<()> {
        inventory::contained(&self.root, &path.join(".component.json"))?;
        let stored: Component =
            serde_json::from_slice(&fs::read(path.join(".component.json")).map_err(fail)?)
                .map_err(fail)?;
        if &stored != c {
            return Err(fail("immutable pin has conflicting package metadata"));
        }
        inventory::verify(path, c)
    }
    pub fn inventory(&self, target: &TargetPlatform) -> Result<ResourceInventory> {
        let selection = self
            .catalog
            .defaults
            .iter()
            .find(|s| &s.target == target)
            .cloned()
            .unwrap_or(ResourceSelection {
                target: target.clone(),
                resources: vec![ResourcePin {
                    id: "application".into(),
                    version: "unconfigured".into(),
                }],
            });
        self.inventory_selection(&selection)
    }
    pub fn inventory_selection(&self, selection: &ResourceSelection) -> Result<ResourceInventory> {
        catalog::validate_target(&selection.target)?;
        let items: Vec<_> = selection
            .resources
            .iter()
            .map(|p| self.status(p, &selection.target))
            .collect();
        Ok(ResourceInventory {
            ready: !items.is_empty() && items.iter().all(|i| i.ready),
            items,
        })
    }
    /// Resolve persisted component identities, including versions absent from a newer catalog.
    /// Bundle identities return the bundle root; nested identities return their component root.
    pub fn resolve_identity(
        &self,
        identity: &ModelIdentity,
        target: &TargetPlatform,
    ) -> Result<PathBuf> {
        self.resolve_model_generation(target, |path, c| {
            if &c.manifest.identity == identity {
                return Ok(Some(path.to_path_buf()));
            }
            let files = c.files();
            for p in &c.manifest.relative_files {
                if p.file_name().is_some_and(|n| n == "manifest.json") {
                    inventory::verify_file(path, p, &files[p])?;
                    let nested: AssetManifest =
                        serde_json::from_slice(&fs::read(path.join(p)).map_err(fail)?)
                            .map_err(fail)?;
                    if &nested.identity == identity {
                        return Ok(Some(path.join(p).parent().unwrap().to_path_buf()));
                    }
                }
            }
            Ok(None)
        })
    }
    fn resolve_model_generation<T>(
        &self,
        target: &TargetPlatform,
        matches: impl Fn(&Path, &Component) -> Result<Option<T>>,
    ) -> Result<T> {
        catalog::validate_target(target)?;
        let mut evidence = Vec::new();
        for id in fs::read_dir(&self.root).map_err(fail)? {
            let id = id.map_err(fail)?;
            if !id.file_type().map_err(fail)?.is_dir()
                || id.file_name().to_string_lossy().starts_with('.')
            {
                continue;
            }
            // Enumeration errors remain fatal; candidate corruption is handled below.
            inventory::contained(&self.root, &id.path())?;
            let mut versions = fs::read_dir(id.path())
                .map_err(fail)?
                .collect::<std::io::Result<Vec<_>>>()
                .map_err(fail)?;
            // Stable generation order makes selection and failure diagnostics reproducible.
            versions.sort_by_key(|entry| entry.file_name());
            for version in versions {
                if !version.file_type().map_err(fail)?.is_dir() {
                    continue;
                }
                let path = version.path();
                match self.model_candidate(&path, target, &matches) {
                    Ok(Some(found)) => return Ok(found),
                    Ok(None) => {}
                    Err(error) => {
                        // Bound evidence independently of the number/size of damaged packages.
                        if evidence.len() < 8 {
                            evidence.push(
                                format!("{}: {}", path.display(), error)
                                    .chars()
                                    .take(512)
                                    .collect(),
                            );
                        }
                    }
                }
            }
        }
        let mut error = AppError::new(
            ErrorCode::ResourceMissing,
            "resources",
            "Pinned model dependencies have no fully verified installed generation",
        );
        error.evidence = evidence;
        Err(error)
    }
    /// Failure rejects this entire generation, never the remaining candidates.
    fn model_candidate<T>(
        &self,
        path: &Path,
        target: &TargetPlatform,
        matches: &impl Fn(&Path, &Component) -> Result<Option<T>>,
    ) -> Result<Option<T>> {
        inventory::contained(&self.root, path)?;
        inventory::contained(&self.root, &path.join(".component.json"))?;
        let c: Component =
            serde_json::from_slice(&fs::read(path.join(".component.json")).map_err(fail)?)
                .map_err(fail)?;
        c.validate()?;
        if self.destination(&c) != path {
            return Err(fail("installed metadata does not match generation path"));
        }
        if !matches!(c.kind.as_str(), "model" | "model_bundle") || !compatible(&c, target) {
            return Ok(None);
        }
        let found = matches(path, &c)?;
        if found.is_some() {
            self.verify_installed(path, &c)?;
        }
        Ok(found)
    }
    pub fn prepare(
        &self,
        selection: &ResourceSelection,
        observer: &dyn TaskObserver,
        cancel: &Cancellation,
    ) -> Result<ResourceInventory> {
        catalog::validate_target(&selection.target)?;
        let _guard = self.mutation.lock().map_err(fail)?;
        let _cache_lock = self.cache_lock()?;
        download::cancelled(cancel)?;
        for pin in &selection.resources {
            if self.status(pin, &selection.target).ready {
                continue;
            }
            let c = self.find(pin)?;
            if !compatible(&c, &selection.target) {
                return Err(fail("resource target/backend incompatible with selection"));
            }
            self.install(&c, None, observer, cancel)?;
        }
        self.inventory_selection(selection)
    }
    /// Import only a manifest whose exact bytes and inventory already occur in the configured catalog.
    pub fn import_local(&self, manifest: &Path) -> Result<()> {
        let _guard = self.mutation.lock().map_err(fail)?;
        let _cache_lock = self.cache_lock()?;
        let raw = fs::read(manifest).map_err(fail)?;
        let asset: AssetManifest = serde_json::from_slice(&raw).map_err(fail)?;
        let candidates: Vec<_> = self
            .catalog
            .components
            .iter()
            .filter(|c| c.manifest == asset)
            .collect();
        if candidates.len() != 1 {
            return Err(fail(
                "local manifest must match exactly one configured component",
            ));
        }
        let c = candidates[0];
        let source = manifest
            .parent()
            .ok_or_else(|| fail("manifest has no parent"))?
            .canonicalize()
            .map_err(fail)?;
        inventory::verify_file(
            &source,
            Path::new("manifest.json"),
            &c.files()[Path::new("manifest.json")],
        )?;
        self.install(c, Some(&source), &Silent, &Cancellation::new())
    }
    fn install(
        &self,
        c: &Component,
        source: Option<&Path>,
        observer: &dyn TaskObserver,
        cancel: &Cancellation,
    ) -> Result<()> {
        let dest = self.destination(c);
        let parent = dest.parent().unwrap();
        fs::create_dir_all(parent).map_err(fail)?;
        inventory::contained(&self.root, parent)?;
        if dest.exists() {
            inventory::contained(&self.root, &dest)?;
            inventory::contained(&self.root, &dest.join(".component.json"))?;
            let stored: Component =
                serde_json::from_slice(&fs::read(dest.join(".component.json")).map_err(fail)?)
                    .map_err(fail)?;
            if stored != *c {
                return Err(fail("refusing to change immutable pin"));
            }
            if inventory::verify(&dest, c).is_ok() {
                return Ok(());
            }
        }
        let staging = tempfile::Builder::new()
            .prefix(".staging-")
            .tempdir_in(&self.root)
            .map_err(fail)?;
        for (relative, spec) in c.files() {
            download::cancelled(cancel)?;
            let path = staging.path().join(&relative);
            fs::create_dir_all(path.parent().unwrap()).map_err(fail)?;
            if let Some(source) = source {
                inventory::verify_file(source, &relative, &spec)?;
                fs::copy(
                    inventory::contained(source, &source.join(&relative))?,
                    &path,
                )
                .map_err(fail)?;
            } else {
                download::fetch(&path, &spec, observer, cancel)?;
            }
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for p in &c.executables {
                fs::set_permissions(staging.path().join(p), fs::Permissions::from_mode(0o755))
                    .map_err(fail)?;
            }
        }
        inventory::verify(staging.path(), c)?;
        fs::write(
            staging.path().join(".component.json"),
            serde_json::to_vec(c).map_err(fail)?,
        )
        .map_err(fail)?;
        // Flush imported/downloaded bytes and metadata before publishing the directory.
        for relative in c
            .files()
            .keys()
            .chain(std::iter::once(&PathBuf::from(".component.json")))
        {
            fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(staging.path().join(relative))
                .map_err(fail)?
                .sync_all()
                .map_err(fail)?;
        }
        download::cancelled(cancel)?;
        // Invalid bytes are quarantined only after the replacement is fully verified.
        let quarantine = if dest.exists() {
            let q = tempfile::Builder::new()
                .prefix(".corrupt-")
                .tempdir_in(&self.root)
                .map_err(fail)?;
            fs::rename(&dest, q.path().join("package")).map_err(fail)?;
            Some(q)
        } else {
            None
        };
        if let Err(e) = fs::rename(staging.path(), &dest) {
            if let Some(q) = &quarantine {
                let _ = fs::rename(q.path().join("package"), &dest);
            }
            return Err(fail(e));
        }
        observer.emit(EventPayload::Progress {
            stage: format!("resources/{}", c.pin.id),
            completed: 1,
            total: Some(1),
        })?;
        Ok(())
    }
}
struct Silent;
impl TaskObserver for Silent {
    fn emit(&self, _: EventPayload) -> Result<()> {
        Ok(())
    }
}

fn compatible(c: &Component, t: &TargetPlatform) -> bool {
    c.target.as_ref().is_none_or(|p| {
        p.os == t.os
            && p.architecture == t.architecture
            && (p.backend == "any" || p.backend == t.backend)
    })
}
