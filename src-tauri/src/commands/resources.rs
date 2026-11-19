use super::blocking;
use crate::state::AppState;
use feathertalk_core::{ResourceInventory, ResourceSelection, Result, TargetPlatform, TaskId};
use std::path::PathBuf;
use tauri::State;

#[tauri::command]
pub fn list_resource_selections(app: State<'_, AppState>) -> Vec<ResourceSelection> {
    app.resource_selections()
}

#[tauri::command]
pub async fn get_resources(
    app: State<'_, AppState>,
    target: TargetPlatform,
) -> Result<ResourceInventory> {
    let app = app.inner().clone();
    blocking(move || app.get_resources(&target)).await
}

#[tauri::command]
pub fn prepare_resources(app: State<'_, AppState>, selection: ResourceSelection) -> Result<TaskId> {
    app.prepare_resources(selection)
}

#[tauri::command]
pub fn import_resource(
    app: State<'_, AppState>,
    target: TargetPlatform,
    manifest: PathBuf,
) -> Result<TaskId> {
    app.import_resource(target, manifest)
}
