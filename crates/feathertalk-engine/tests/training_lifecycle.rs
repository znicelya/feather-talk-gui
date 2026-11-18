mod training_fixture;
use feathertalk_core::*;
use feathertalk_engine::training::{
    batch_probe::TrainingRuntime, checkpoint::load_checkpoint, metrics::metrics_page, *,
};
use std::{path::Path, sync::Mutex};
struct Observe {
    events: Mutex<Vec<EventPayload>>,
    cancel: Cancellation,
}
impl TaskObserver for Observe {
    fn emit(&self, e: EventPayload) -> Result<()> {
        if matches!(e, EventPayload::Metrics(_)) {
            self.cancel.cancel();
        }
        self.events.lock().unwrap().push(e);
        Ok(())
    }
}
#[test]
fn actual_cancel_save_resume_total_epochs_history() {
    let (_root, prepared) = training_fixture::fixture();
    let device = training_fixture::device();
    eprintln!("ACTUAL LIFECYCLE DEVICE {:?}", device.descriptor);
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/assets/vgg19");
    let runtime = TrainingRuntime {
        probe_runner: prepared.project_directory.join("must-never-run.exe"),
    };
    let cancel = Cancellation::new();
    let observer = Observe {
        events: Mutex::new(vec![]),
        cancel: cancel.clone(),
    };
    let mut request = TrainRequest {
        project_id: prepared.project_id,
        batch_mode: BatchMode::Fixed,
        config: TrainConfig {
            epochs: 2,
            batch_size: 1,
            save_every: 1,
            ..Default::default()
        },
        device: DevicePolicy::Auto,
        resume: None,
    };
    let first = train(
        &request, &prepared, &assets, &device, &runtime, &observer, &cancel,
    )
    .unwrap();
    assert_eq!(first.stopping_reason, StoppingReason::Cancelled);
    assert_eq!(first.global_step, 1);
    assert_eq!(first.epoch, 1);
    let old =
        load_checkpoint(&prepared.project_directory, &first.committed_model, &device).unwrap();
    assert_eq!(old.metadata.global_step, 1);
    request.resume = Some(first.committed_model.clone());
    request.config.lr = 0.0005;
    request.config.prefetch_capacity = 0;
    let second = train(
        &request,
        &prepared,
        &assets,
        &device,
        &runtime,
        &observer,
        &Cancellation::new(),
    )
    .unwrap();
    assert_eq!(second.stopping_reason, StoppingReason::Completed);
    assert_eq!(second.global_step, 2);
    assert_eq!(second.run_id, first.run_id);
    let loaded = load_checkpoint(
        &prepared.project_directory,
        &second.committed_model,
        &device,
    )
    .unwrap();
    assert_eq!(loaded.metadata.config.lr, 0.0005);
    assert_eq!(loaded.metadata.config.prefetch_capacity, 0);
    request.resume = Some(second.committed_model.clone());
    let done = train(
        &request,
        &prepared,
        &assets,
        &device,
        &runtime,
        &observer,
        &Cancellation::new(),
    )
    .unwrap();
    assert_eq!(done.committed_model, second.committed_model);
    request.config.epochs = 1;
    assert!(
        train(
            &request,
            &prepared,
            &assets,
            &device,
            &runtime,
            &observer,
            &Cancellation::new()
        )
        .unwrap_err()
        .message
        .contains("below")
    );
    let page = metrics_page(
        &prepared.project_directory,
        &MetricsPageRequest {
            project_id: prepared.project_id,
            run_id: first.run_id,
            offset: 0,
            limit: 10,
        },
    )
    .unwrap();
    assert_eq!(page.items.len(), 2);
    let events = observer.events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, EventPayload::CheckpointCommitted { .. }))
            .count(),
        2
    );
    assert!(!events.iter().any(|e| matches!(
        e,
        EventPayload::Completed { .. } | EventPayload::Cancelled { .. }
    )));
}
