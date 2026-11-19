use super::blocking;
use crate::state::AppState;
use feathertalk_core::{DeviceDescriptor, Result};
use tauri::State;

#[tauri::command]
pub async fn list_devices(app: State<'_, AppState>) -> Result<Vec<DeviceDescriptor>> {
    let app = app.inner().clone();
    blocking(move || app.list_devices()).await
}
