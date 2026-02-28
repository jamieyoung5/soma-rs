use std::fmt;

/// All the ways things can go wrong in soma.
#[derive(Debug, thiserror::Error)]
pub enum SomaError {
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Data error: {0}")]
    Data(String),

    #[error("Unknown algorithm: {0}")]
    UnknownAlgorithm(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Invalid model file: {0}")]
    InvalidModelFile(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

impl SomaError {
    pub fn model(msg: impl fmt::Display) -> Self {
        Self::Model(msg.to_string())
    }

    pub fn data(msg: impl fmt::Display) -> Self {
        Self::Data(msg.to_string())
    }

    pub fn config(msg: impl fmt::Display) -> Self {
        Self::Config(msg.to_string())
    }
}

pub type Result<T> = std::result::Result<T, SomaError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        assert_eq!(
            SomaError::model("training failed").to_string(),
            "Model error: training failed"
        );
        assert_eq!(
            SomaError::data("empty dataset").to_string(),
            "Data error: empty dataset"
        );
        assert_eq!(
            SomaError::UnknownAlgorithm("magic_tree".into()).to_string(),
            "Unknown algorithm: magic_tree"
        );
        assert_eq!(
            SomaError::ColumnNotFound("target".into()).to_string(),
            "Column not found: target"
        );
        assert_eq!(
            SomaError::config("missing required flag").to_string(),
            "Configuration error: missing required flag"
        );
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err: SomaError = io_err.into();
        assert!(matches!(err, SomaError::Io(_)));
        assert!(err.to_string().contains("file missing"));
    }

    #[test]
    fn is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SomaError>();
    }
}
