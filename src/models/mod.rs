mod algorithm;
pub mod store;
mod wrapper;

pub use algorithm::{Algorithm, TaskType};
pub use store::ModelStore;
pub use wrapper::TrainedModel;
