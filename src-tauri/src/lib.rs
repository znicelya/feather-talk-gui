mod commands;
pub mod projects;
pub mod resources;
pub mod state;
pub mod worker_host;
mod worker_process;

use commands::*;
use feathertalk_core::{AppError, ErrorCode, Result};
use projects::ProjectStore;
use resources::{Catalog, ResourceManager};
use state::{AppState, CloseState, TASK_EVENT};
use std::{fs, io, path::Path};
use tauri::{Emitter, Manager, RunEvent, WindowEvent};

fn desktop_state(handle: &tauri::AppHandle) -> Result<AppState> {
    let data = handle
        .path()
        .app_local_data_dir()
        .map_err(|e| desktop_error("app-data", e))?;
    // A separate debug data directory lets the local resource catalog be exercised in isolation.
    #[cfg(debug_assertions)]
    let data = match std::env::var_os("FT_DESKTOP_DATA_DIR").map(std::path::PathBuf::from) {
        Some(path) if path.is_absolute() => path,
        Some(_) => {
            return Err(AppError::new(
                ErrorCode::InvalidConfig,
                "app-data",
                "FT_DESKTOP_DATA_DIR must be absolute",
            ));
        }
        None => data,
    };
    let catalog_path = data.join("resource-catalog.json");
    let bundled = handle
        .path()
        .resolve("bootstrap", tauri::path::BaseDirectory::Resource)
        .ok();
    let catalog_error = |path: &Path, e: &dyn std::fmt::Display| {
        let mut error = AppError::new(
            ErrorCode::ResourceCorrupt,
            "resource-catalog",
            e.to_string(),
        );
        error.evidence.push(path.display().to_string());
        error
            .actions
            .push("Restore a valid resource catalog at this path.".into());
        error
    };
    let bundled_catalog = bundled.as_ref().map(|root| root.join("catalog.json"));
    let mut loaded = None;
    for path in std::iter::once(&catalog_path).chain(bundled_catalog.as_ref()) {
        match fs::read(path) {
            Ok(bytes) => {
                loaded = Some((path, bytes));
                break;
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => return Err(catalog_error(path, &e)),
        }
    }
    let (catalog_source, bytes) = loaded
        .as_ref()
        .map(|(path, bytes)| (path.as_path(), bytes.as_slice()))
        .unwrap_or((
            Path::new("embedded development-catalog.json"),
            include_bytes!("../../resources/development-catalog.json"),
        ));
    let catalog: Catalog =
        serde_json::from_slice(bytes).map_err(|e| catalog_error(catalog_source, &e))?;
    let projects = ProjectStore::new(data.join("projects.json"))?;
    let resources = ResourceManager::new(data.join("resources"), catalog)
        .map_err(|e| catalog_error(catalog_source, &e))?;
    if let Some(root) = bundled {
        resources.install_bundled_cpu(&root);
    }
    let handle = handle.clone();
    Ok(AppState::new(projects, resources, move |event| {
        if let Err(error) = handle.emit_to("main", TASK_EVENT, event) {
            eprintln!("Could not forward task event: {error}");
        }
    }))
}

fn finish_close(app: AppState, close: impl FnOnce() + Send + 'static) {
    tauri::async_runtime::spawn_blocking(move || match app.wait_until_idle() {
        Ok(()) => close(),
        Err(error) => eprintln!("Could not wait for task shutdown: {error}"),
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let desktop = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            app.manage(desktop_state(app.handle())?);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_projects,
            create_project,
            open_project,
            get_project,
            list_devices,
            list_resource_selections,
            get_resources,
            prepare_resources,
            import_resource,
            get_train_defaults,
            start_training,
            start_generation,
            cancel_task,
            get_task,
            read_metrics,
            select_path,
            allow_video_preview,
            open_output_directory,
        ])
        .on_window_event(|window, event| {
            if window.label() != "main" {
                return;
            }
            if let WindowEvent::CloseRequested { api, .. } = event {
                let app = window.state::<AppState>().inner().clone();
                match app.request_close() {
                    Ok(CloseState::Ready) => {}
                    Ok(CloseState::Waiting) => {
                        api.prevent_close();
                        let window = window.clone();
                        finish_close(app, move || {
                            if let Err(error) = window.close() {
                                eprintln!("Could not close the main window: {error}");
                            }
                        });
                    }
                    Ok(CloseState::AlreadyWaiting) => api.prevent_close(),
                    Err(error) => {
                        api.prevent_close();
                        eprintln!("Could not request task shutdown: {error}");
                    }
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building the FeatherTalk desktop application");

    desktop.run(|handle, event| {
        if let RunEvent::ExitRequested { api, code, .. } = event {
            let app = handle.state::<AppState>().inner().clone();
            match app.request_close() {
                Ok(CloseState::Ready) => {}
                Ok(CloseState::Waiting) => {
                    api.prevent_exit();
                    let handle = handle.clone();
                    finish_close(app, move || handle.exit(code.unwrap_or_default()));
                }
                Ok(CloseState::AlreadyWaiting) => api.prevent_exit(),
                Err(error) => {
                    api.prevent_exit();
                    eprintln!("Could not request task shutdown: {error}");
                }
            }
        }
    });
}
