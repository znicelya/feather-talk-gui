use super::blocking;
use crate::state::AppState;
use feathertalk_core::{NewProject, ProjectId, ProjectManifest, ProjectSummary, Result};
use std::path::PathBuf;
use tauri::State;

#[tauri::command]
pub async fn list_projects(app: State<'_, AppState>) -> Result<Vec<ProjectSummary>> {
    let app = app.inner().clone();
    blocking(move || app.list_projects()).await
}

#[tauri::command]
pub async fn create_project(
    app: State<'_, AppState>,
    project: NewProject,
) -> Result<ProjectSummary> {
    let app = app.inner().clone();
    blocking(move || app.create_project(project)).await
}

#[tauri::command]
pub async fn open_project(app: State<'_, AppState>, directory: PathBuf) -> Result<ProjectSummary> {
    let app = app.inner().clone();
    blocking(move || app.open_project(&directory)).await
}

#[tauri::command]
pub async fn get_project(
    app: State<'_, AppState>,
    project_id: ProjectId,
) -> Result<ProjectManifest> {
    let app = app.inner().clone();
    blocking(move || app.get_project(project_id)).await
}
