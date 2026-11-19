use super::blocking;
use crate::state::AppState;
use feathertalk_core::*;
use tauri::State;

#[tauri::command]
pub fn get_train_defaults() -> TrainConfig {
    TrainConfig::default()
}

#[tauri::command]
pub fn start_training(app: State<'_, AppState>, request: TrainRequest) -> Result<TaskId> {
    if request.config.seed > 9_007_199_254_740_991 {
        let mut error = AppError::new(
            ErrorCode::InvalidConfig,
            "configuration",
            "The training seed exceeds the desktop's exact integer range",
        );
        error
            .actions
            .push("Choose an integer from 0 to 9007199254740991.".into());
        return Err(error);
    }
    app.start_training(request)
}

#[tauri::command]
pub fn start_generation(app: State<'_, AppState>, request: GenerateRequest) -> Result<TaskId> {
    app.start_generation(request)
}

#[tauri::command]
pub fn cancel_task(app: State<'_, AppState>, task_id: TaskId) -> Result<()> {
    app.cancel_task(task_id)
}

#[tauri::command]
pub fn get_task(app: State<'_, AppState>) -> Result<Option<TaskSnapshot>> {
    app.snapshot()
}

#[tauri::command]
pub async fn read_metrics(
    app: State<'_, AppState>,
    request: MetricsPageRequest,
) -> Result<MetricsPage> {
    let app = app.inner().clone();
    blocking(move || app.read_metrics(&request)).await
}
