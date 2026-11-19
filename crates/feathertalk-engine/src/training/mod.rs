pub mod batch_probe;
pub mod checkpoint;
#[cfg(test)]
mod checkpoint_tests;
mod checkpoint_validation;
pub mod lifecycle;
pub mod losses;
pub mod metrics;
pub mod probe_transport;
pub mod state;
pub mod trainer;
pub use batch_probe::{ProbeContext, ProbeSelection, TrainingRuntime, probe_batch_size};
pub use lifecycle::{StoppingReason, TrainingResult, train};
#[cfg(all(debug_assertions, feature = "dev-worker-acceptance"))]
mod worker_acceptance;
