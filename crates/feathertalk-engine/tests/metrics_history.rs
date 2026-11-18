mod training_fixture;
use feathertalk_core::*;
use feathertalk_engine::training::metrics::{SampledMetrics, metrics_page};
use std::{fs, io::Write};
use training_fixture::*;
fn metric(project_id: ProjectId, run_id: TrainingRunId, step: u64) -> TrainingMetrics {
    TrainingMetrics {
        project_id,
        run_id,
        step,
        epoch: 0,
        elapsed_ms: step * 500,
        lr: 0.001,
        losses: LossScalars {
            full: 1.,
            mouth: 1.,
            temporal: 1.,
            temporal_mouth: 1.,
            perceptual: 1.,
            mouth_gradient: 1.,
            total: 6.,
        },
        device: device().descriptor,
        precision: Precision::F32,
        batch_size: 1,
    }
}
#[test]
fn canonical_history_filters_runs_bounds_pages_and_rejects_torn_tail() {
    let (root, p) = fixture();
    let run = TrainingRunId::new();
    let other = TrainingRunId::new();
    let mut file = fs::File::create(root.join("metrics.jsonl")).unwrap();
    for (r, s) in [(run, 1), (other, 1), (run, 2), (other, 2), (run, 3)] {
        let m = metric(p.project_id, r, s);
        serde_json::to_writer(&mut file, &m).unwrap();
        file.write_all(b"\n").unwrap();
    }
    drop(file);
    let q = MetricsPageRequest {
        project_id: p.project_id,
        run_id: run,
        offset: 0,
        limit: 2,
    };
    let first = metrics_page(&root, &q).unwrap();
    assert_eq!(
        first.items.iter().map(|m| m.step).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(first.next_offset, Some(2));
    let last = metrics_page(
        &root,
        &MetricsPageRequest {
            offset: 2,
            ..q.clone()
        },
    )
    .unwrap();
    assert_eq!(last.items[0].step, 3);
    assert_eq!(last.next_offset, None);
    assert!(
        metrics_page(
            &root,
            &MetricsPageRequest {
                limit: 1001,
                ..q.clone()
            }
        )
        .is_err()
    );
    assert!(
        metrics_page(
            &root,
            &MetricsPageRequest {
                project_id: ProjectId::new(),
                ..q.clone()
            }
        )
        .is_err()
    );
    let writer = SampledMetrics::open(&root, p.project_id, run).unwrap();
    assert_eq!(metrics_page(&root, &q).unwrap().items.len(), 2);
    assert!(SampledMetrics::open(&root, p.project_id, other).is_err());
    drop(writer);
    let before = fs::read(root.join("metrics.jsonl")).unwrap();
    fs::OpenOptions::new()
        .append(true)
        .open(root.join("metrics.jsonl"))
        .unwrap()
        .write_all(b"{\"interrupted\":")
        .unwrap();
    let err = SampledMetrics::open(&root, p.project_id, run)
        .err()
        .unwrap();
    assert!(err.message.contains("Interrupted trailing"));
    assert_eq!(
        &fs::read(root.join("metrics.jsonl")).unwrap()[..before.len()],
        &before
    );
    assert!(
        metrics_page(
            &root,
            &MetricsPageRequest {
                limit: 1000,
                ..q.clone()
            }
        )
        .is_err()
    );
    let mut foreign = metric(ProjectId::new(), run, 1);
    let mut bad = serde_json::to_vec(&foreign).unwrap();
    bad.push(b'\n');
    fs::write(root.join("metrics.jsonl"), bad).unwrap();
    assert!(metrics_page(&root, &q).is_err());
    foreign.project_id = p.project_id;
    let mut bad = serde_json::to_value(&foreign).unwrap();
    bad["losses"]["total"] = serde_json::Value::Null;
    let mut bytes = serde_json::to_vec(&bad).unwrap();
    bytes.push(b'\n');
    fs::write(root.join("metrics.jsonl"), bytes).unwrap();
    assert!(metrics_page(&root, &q).is_err());
    fs::write(root.join("metrics.jsonl"), vec![b'x'; 65537]).unwrap();
    assert!(metrics_page(&root, &q).is_err());
}
