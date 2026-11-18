mod training_convergence;
mod training_fixture;
use burn::{module::Module, tensor::Tensor};
use feathertalk_core::*;
use feathertalk_engine::{
    models::vgg::VggFeatures,
    training::{
        checkpoint::{MaterializedSnapshot, load_checkpoint, save_checkpoint},
        metrics::{SampledMetrics, metrics_page},
        state::TrainingState,
        trainer::complete_step,
    },
};
use std::{fs, path::Path};
use training_fixture::*;
fn equal(a: &[u8], b: &[u8]) {
    let a = burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(a.to_vec())).unwrap();
    let b = burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(b.to_vec())).unwrap();
    assert_eq!(a.scalars(), b.scalars());
    assert_eq!(a.metadata(), b.metadata());
    let a = a.into_tensors().unwrap();
    let b = b.into_tensors().unwrap();
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(&b) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.param_id, b.param_id);
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.dtype, b.dtype);
        assert_eq!(a.to_bytes().unwrap(), b.to_bytes().unwrap(), "{}", a.name);
    }
}
fn model(state: &TrainingState, root: &Path, label: &str) -> Vec<u8> {
    let p = root.join(label);
    MaterializedSnapshot::capture(state)
        .unwrap()
        .write_private(&p)
        .unwrap();
    fs::read(p.join("model.bpk")).unwrap()
}
#[test]
fn full_objective_commits_decreases_and_exactly_resumes() {
    let device = device();
    eprintln!("ACTUAL DEVICE {:?}", device.descriptor);
    let (root, prepared) = fixture();
    let request = TrainRequest {
        project_id: prepared.project_id,
        batch_mode: BatchMode::Fixed,
        config: TrainConfig {
            batch_size: 1,
            ..Default::default()
        },
        device: DevicePolicy::Auto,
        resume: None,
    };
    let mut state = TrainingState::fresh(&request, &prepared, &device).unwrap();
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/assets/vgg19");
    let vgg = VggFeatures::load(&assets, &device.device.clone().autodiff()).unwrap();
    struct Frozen(Vec<Vec<f32>>);
    impl burn::module::ModuleVisitor for Frozen {
        fn visit_float<const D: usize>(&mut self, p: &burn::module::Param<Tensor<D>>) {
            assert!(!p.is_require_grad());
            self.0.push(p.val().to_data().try_to_vec::<f32>().unwrap());
        }
    }
    assert!(state.metadata.vgg_identity.is_none());
    let mut before = Frozen(vec![]);
    vgg.visit(&mut before);
    assert_eq!(before.0.len(), 14);
    let pristine = model(&state, &root, "pristine");
    let initial = state.metadata.clone();
    let cancelled = Cancellation::new();
    cancelled.cancel();
    assert!(complete_step(&mut state, &prepared, &vgg, &device, &cancelled).is_err());
    assert_eq!(state.metadata, initial);
    equal(&pristine, &model(&state, &root, "cancelled"));
    let mut broken = prepared.clone();
    broken.features = std::sync::Arc::new(vec![1.; broken.features.len()]);
    assert!(complete_step(&mut state, &broken, &vgg, &device, &Cancellation::new()).is_err());
    assert_eq!(state.metadata, initial);
    eprintln!("Starting first complete objective step");
    let first = complete_step(&mut state, &prepared, &vgg, &device, &Cancellation::new()).unwrap();
    let first_loss = first.losses.to_scalars();
    first_loss.validate().unwrap();
    assert_eq!(state.metadata.vgg_identity.as_ref(), Some(vgg.identity()));
    assert_eq!(state.metadata.global_step, 1);
    assert_eq!(state.metadata.epoch, 1);
    assert_eq!(first.pairs, vec![[0, 1]]);
    let mut metrics =
        SampledMetrics::open(&root, state.metadata.project_id, state.metadata.run_id).unwrap();
    assert!(
        metrics
            .sample(&state, &first.losses, &device, 0, 1)
            .unwrap()
            .is_some()
    );
    let saved = save_checkpoint(&state, &root.join("checkpoints/first")).unwrap();
    let mut resumed = load_checkpoint(&root, &saved, &device).unwrap();
    assert_eq!(resumed.metadata.vgg_identity.as_ref(), Some(vgg.identity()));
    eprintln!("Starting second complete objective step");
    let second = complete_step(&mut state, &prepared, &vgg, &device, &Cancellation::new()).unwrap();
    let loss = second.losses.to_scalars();
    loss.validate().unwrap();
    eprintln!(
        "FULL OBJECTIVE first={} second={}",
        first_loss.total, loss.total
    );
    training_convergence::assert_full_objective_decreased(&first_loss, &loss);
    assert!(
        metrics
            .sample(&state, &second.losses, &device, 499, 1)
            .unwrap()
            .is_none()
    );
    assert!(
        metrics
            .sample(&state, &second.losses, &device, 500, 1)
            .unwrap()
            .is_some()
    );
    drop(metrics);
    let mut metrics =
        SampledMetrics::open(&root, state.metadata.project_id, state.metadata.run_id).unwrap();
    assert!(
        metrics
            .sample(&state, &second.losses, &device, 501, 1)
            .unwrap()
            .is_none()
    );
    drop(metrics);
    let req = MetricsPageRequest {
        project_id: state.metadata.project_id,
        run_id: state.metadata.run_id,
        offset: 0,
        limit: 1,
    };
    let page = metrics_page(&root, &req).unwrap();
    assert_eq!(page.items[0].step, 1);
    assert_eq!(page.next_offset, Some(1));
    let page = metrics_page(&root, &MetricsPageRequest { offset: 1, ..req }).unwrap();
    assert_eq!(page.items[0].step, 2);
    assert_eq!(page.next_offset, None);
    let replay =
        complete_step(&mut resumed, &prepared, &vgg, &device, &Cancellation::new()).unwrap();
    assert_eq!(second.pairs, replay.pairs);
    assert_eq!(second.references, replay.references);
    assert_eq!(second.seed, replay.seed);
    assert_eq!(state.metadata, resumed.metadata);
    equal(
        &model(&state, &root, "continuous"),
        &model(&resumed, &root, "resumed"),
    );
    equal(
        &state.optimizer_bytes().unwrap(),
        &resumed.optimizer_bytes().unwrap(),
    );
    let mut after = Frozen(vec![]);
    vgg.visit(&mut after);
    assert_eq!(before.0, after.0);
    if std::env::var("FT_CROSS_DEVICE").is_ok() {
        let cpu = feathertalk_engine::open_device(&DeviceRequest {
            backend: BackendKind::Flex,
            device_id: "flex:0".into(),
        })
        .unwrap();
        let snap = MaterializedSnapshot::capture(&state).unwrap();
        let mut cpu_state = snap.restore(&root, &cpu).unwrap();
        let cpu_vgg = VggFeatures::load(&assets, &cpu.device.clone().autodiff()).unwrap();
        // Same persisted parameters on both devices: finite forward comparison,
        // not a claim of bitwise cross-backend optimizer trajectories.
        let batch = feathertalk_engine::data::sampler::PairSampler::new(&prepared, 42)
            .unwrap()
            .next_batch(1)
            .unwrap()
            .unwrap();
        let predict = |s: &TrainingState, d: &feathertalk_engine::ExecutionDevice| {
            let image = Tensor::from_data(
                burn::tensor::TensorData::new(batch.image.values.clone(), [2, 6, 144, 144]),
                &d.device,
            );
            let audio = Tensor::from_data(
                burn::tensor::TensorData::new(batch.audio.values.clone(), [2, 40, 1024]),
                &d.device,
            );
            s.model
                .valid()
                .forward(image, audio)
                .to_data()
                .try_to_vec::<f32>()
                .unwrap()
        };
        let gpu_values = predict(&state, &device);
        let cpu_values = predict(&cpu_state, &cpu);
        let mut max_error = 0f32;
        for (a, b) in gpu_values.iter().zip(cpu_values) {
            assert!(a.is_finite() && b.is_finite());
            max_error = max_error.max((a - b).abs());
            assert!((a - b).abs() <= 0.001 + 0.001 * b.abs());
        }
        eprintln!("Cross-device eval maximum absolute error={max_error}; atol=0.001 rtol=0.001");
        let cross = complete_step(
            &mut cpu_state,
            &prepared,
            &cpu_vgg,
            &cpu,
            &Cancellation::new(),
        )
        .unwrap();
        cross.losses.to_scalars().validate().unwrap();
        let mut back = MaterializedSnapshot::capture(&cpu_state)
            .unwrap()
            .restore(&root, &device)
            .unwrap();
        complete_step(&mut back, &prepared, &vgg, &device, &Cancellation::new())
            .unwrap()
            .losses
            .to_scalars()
            .validate()
            .unwrap();
        MaterializedSnapshot::capture(&back).unwrap();
        eprintln!(
            "GPU -> CPU -> GPU finite full objective committed through step {}",
            back.metadata.global_step
        );
    }
}
