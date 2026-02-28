use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SomaError};
use crate::models::algorithm::Algorithm;
use crate::models::wrapper::TrainedModel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub algorithm: Algorithm,
    pub feature_names: Vec<String>,
    pub target_name: String,
    pub n_train_samples: usize,
    pub n_features: usize,
}

impl std::fmt::Display for ModelMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ModelMetadata(algorithm={}, features=[{}], target='{}', n_train={}, n_features={})",
            self.algorithm,
            self.feature_names.join(", "),
            self.target_name,
            self.n_train_samples,
            self.n_features,
        )
    }
}

/// A trained model bundled with metadata, ready to be saved/loaded from disk.
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelStore {
    version: u32,
    pub metadata: ModelMetadata,
    pub model: TrainedModel,
}

const FORMAT_VERSION: u32 = 1;

impl ModelStore {
    pub fn new(model: TrainedModel, metadata: ModelMetadata) -> Self {
        Self {
            version: FORMAT_VERSION,
            metadata,
            model,
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let encoded = bincode::serialize(self)
            .map_err(|e| SomaError::model(format!("failed to serialize model: {e}")))?;

        fs::write(path, encoded)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(SomaError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("model file not found: {}", path.display()),
            )));
        }

        let bytes = fs::read(path)?;

        let store: Self = bincode::deserialize(&bytes).map_err(|e| {
            SomaError::InvalidModelFile(format!("failed to deserialize '{}': {e}", path.display()))
        })?;

        if store.version != FORMAT_VERSION {
            return Err(SomaError::InvalidModelFile(format!(
                "unsupported model format version {} (expected {FORMAT_VERSION})",
                store.version,
            )));
        }

        Ok(store)
    }

    pub fn validate_features(&self, n_input_features: usize) -> Result<()> {
        if n_input_features != self.metadata.n_features {
            return Err(SomaError::data(format!(
                "model expects {} features but input has {} columns",
                self.metadata.n_features, n_input_features,
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Algorithm;
    use smartcore::linalg::basic::matrix::DenseMatrix;

    fn test_store() -> ModelStore {
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]])
            .unwrap();
        let y = vec![1.0, 2.0, 3.0, 4.0];

        let model = Algorithm::LinearRegression.train(&x, &y).unwrap();

        let metadata = ModelMetadata {
            algorithm: Algorithm::LinearRegression,
            feature_names: vec!["feat_a".into(), "feat_b".into()],
            target_name: "target".into(),
            n_train_samples: 4,
            n_features: 2,
        };

        ModelStore::new(model, metadata)
    }

    #[test]
    fn save_load_roundtrip() {
        let store = test_store();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_model.soma");

        store.save(&path).unwrap();
        assert!(path.exists());

        let loaded = ModelStore::load(&path).unwrap();
        assert_eq!(loaded.metadata.algorithm, Algorithm::LinearRegression);
        assert_eq!(loaded.metadata.feature_names, vec!["feat_a", "feat_b"]);
        assert_eq!(loaded.metadata.target_name, "target");
        assert_eq!(loaded.metadata.n_train_samples, 4);
        assert_eq!(loaded.version, FORMAT_VERSION);
    }

    #[test]
    fn predictions_stable_after_roundtrip() {
        let store = test_store();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.soma");
        store.save(&path).unwrap();

        let loaded = ModelStore::load(&path).unwrap();

        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[5.0, 6.0]]).unwrap();
        let before = store.model.predict(&x).unwrap();
        let after = loaded.model.predict(&x).unwrap();

        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).abs() < 1e-10, "diverged: {a} vs {b}");
        }
    }

    #[test]
    fn load_missing_file() {
        let result = ModelStore::load(Path::new("/tmp/does_not_exist_at_all.soma"));
        assert!(matches!(result.unwrap_err(), SomaError::Io(_)));
    }

    #[test]
    fn load_garbage_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.soma");
        fs::write(&path, b"not a valid model").unwrap();

        assert!(matches!(
            ModelStore::load(&path).unwrap_err(),
            SomaError::InvalidModelFile(_)
        ));
    }

    #[test]
    fn validate_features_ok_and_mismatch() {
        let store = test_store();
        assert!(store.validate_features(2).is_ok());

        let err = store.validate_features(5).unwrap_err();
        assert!(matches!(err, SomaError::Data(_)));
        assert!(err.to_string().contains("expects 2 features"));
    }

    #[test]
    fn metadata_display() {
        let s = test_store().metadata.to_string();
        assert!(s.contains("feat_a"));
        assert!(s.contains("target"));
    }

    #[test]
    fn creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a").join("b").join("c").join("model.soma");

        test_store().save(&nested).unwrap();
        assert!(nested.exists());
    }
}
