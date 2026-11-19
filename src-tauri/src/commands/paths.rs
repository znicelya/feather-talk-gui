use super::{blocking, desktop_error};
use crate::state::AppState;
use feathertalk_core::{ProjectId, Result, TaskId};
use serde::Deserialize;
use std::path::PathBuf;
use tauri::{Manager, State, WebviewWindow};
use tauri_plugin_dialog::DialogExt;

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathChoice {
    ProjectDirectory,
    SourceVideo,
    DrivingAudio,
    OutputVideo,
    ResourceManifest,
}

#[tauri::command]
pub async fn select_path(window: WebviewWindow, kind: PathChoice) -> Result<Option<PathBuf>> {
    let dialog = window.dialog().file().set_parent(&window);
    blocking(move || {
        let path = match kind {
            PathChoice::ProjectDirectory => dialog.blocking_pick_folder(),
            PathChoice::SourceVideo => dialog
                .add_filter("Video", &["mp4", "mov", "mkv", "webm", "avi", "m4v"])
                .blocking_pick_file(),
            PathChoice::DrivingAudio => dialog
                .add_filter(
                    "Audio",
                    &["wav", "mp3", "m4a", "aac", "flac", "ogg", "opus"],
                )
                .blocking_pick_file(),
            PathChoice::OutputVideo => dialog
                .add_filter("MP4", &["mp4"])
                .set_file_name("FeatherTalk.mp4")
                .blocking_save_file(),
            PathChoice::ResourceManifest => dialog
                .add_filter("Manifest", &["json"])
                .blocking_pick_file(),
        };
        path.map(|path| {
            path.into_path()
                .map_err(|e| desktop_error("file-dialog", e))
        })
        .transpose()
    })
    .await
}

#[tauri::command]
pub async fn allow_video_preview(
    window: WebviewWindow,
    app: State<'_, AppState>,
    project_id: ProjectId,
    task_id: TaskId,
) -> Result<PathBuf> {
    let app = app.inner().clone();
    blocking(move || {
        let path = app.completed_video(project_id, task_id)?;
        window
            .asset_protocol_scope()
            .allow_file(&path)
            .map_err(|e| desktop_error("video-preview", e))?;
        Ok(path)
    })
    .await
}

#[tauri::command]
pub async fn open_output_directory(
    app: State<'_, AppState>,
    project_id: ProjectId,
    task_id: TaskId,
) -> Result<()> {
    let app = app.inner().clone();
    blocking(move || {
        let path = app.completed_video(project_id, task_id)?;
        tauri_plugin_opener::reveal_item_in_dir(path)
            .map_err(|e| desktop_error("output-directory", e))
    })
    .await
}
