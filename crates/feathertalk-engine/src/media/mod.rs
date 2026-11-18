pub mod ffmpeg;
pub mod probe;
use std::path::PathBuf;
#[derive(Clone, Debug)]
pub struct MediaTools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}
impl MediaTools {
    pub fn from_env() -> feathertalk_core::Result<Self> {
        let path = |key| {
            std::env::var_os(key)
                .map(PathBuf::from)
                .filter(|p| p.is_file())
                .ok_or_else(|| {
                    feathertalk_core::AppError::new(
                        feathertalk_core::ErrorCode::ResourceMissing,
                        "media-tools",
                        format!("Set {key} to the explicit executable path"),
                    )
                })
        };
        Ok(Self {
            ffmpeg: path("FT_FFMPEG")?,
            ffprobe: path("FT_FFPROBE")?,
        })
    }
}
