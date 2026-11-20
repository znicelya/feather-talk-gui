//! Versioned SplitMix64 streams; speculative work commits only on consumer receive.
use super::{
    geometry::{audio_window, invalid},
    prepared::PreparedProject,
};
use feathertalk_core::*;
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, mpsc},
    thread::{self, JoinHandle},
    time::Duration,
};
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor<const N: usize> {
    pub dims: [usize; N],
    pub values: Vec<f32>,
}
impl<const N: usize> HostTensor<N> {
    pub fn new(dims: [usize; N], values: Vec<f32>) -> Result<Self> {
        if dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(values.len())
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("Invalid host tensor dimensions or values"));
        }
        Ok(Self { dims, values })
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingBatch {
    pub image: HostTensor<5>,
    pub audio: HostTensor<4>,
    pub target: HostTensor<5>,
    pub mask: HostTensor<5>,
    pub pairs: Vec<[usize; 2]>,
    pub references: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplerState {
    pub version: u32,
    pub algorithm: String,
    pub prepared_identity: String,
    pub seed: u64,
    pub epoch: u64,
    pub cursor: usize,
    pub permutation: Vec<usize>,
    pub shuffle_state: u64,
    pub reference_state: u64,
    pub reference_draws: u64,
}
fn word(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn choice(state: &mut u64, n: usize) -> usize {
    let n = n as u64;
    let threshold = n.wrapping_neg() % n;
    loop {
        let v = word(state);
        if v >= threshold {
            return (v % n) as usize;
        }
    }
}
#[derive(Clone)]
pub struct PairSampler {
    prepared: Arc<PreparedProject>,
    state: SamplerState,
    check_frame_hashes: bool,
}
impl PairSampler {
    pub fn new(prepared: &PreparedProject, seed: u64) -> Result<Self> {
        prepared.validate()?;
        Ok(Self {
            prepared: Arc::new(prepared.clone()),
            state: Self::initial_state(prepared, seed),
            check_frame_hashes: true,
        })
    }
    pub(crate) fn initial_state(prepared: &PreparedProject, seed: u64) -> SamplerState {
        let mut state = SamplerState {
            version: 1,
            algorithm: "splitmix64-rejection-fisher-yates-v1".into(),
            prepared_identity: prepared.identity.clone(),
            seed,
            epoch: 0,
            cursor: 0,
            permutation: (0..prepared.usable_frames.saturating_sub(1)).collect(),
            shuffle_state: seed ^ 0x73687566666c6500,
            reference_state: seed ^ 0x7265666572656e63,
            reference_draws: 0,
        };
        shuffle(&mut state);
        state
    }
    pub fn restore(prepared: &PreparedProject, state: SamplerState) -> Result<Self> {
        prepared.validate()?;
        Self::validate_state(prepared, &state)?;
        Ok(Self {
            prepared: Arc::new(prepared.clone()),
            state,
            check_frame_hashes: true,
        })
    }
    pub(crate) fn validate_state(prepared: &PreparedProject, state: &SamplerState) -> Result<()> {
        let mut sorted = state.permutation.clone();
        sorted.sort_unstable();
        if state.version != 1
            || state.algorithm != "splitmix64-rejection-fisher-yates-v1"
            || state.prepared_identity != prepared.identity
            || state.cursor > sorted.len()
            || sorted != (0..prepared.usable_frames.saturating_sub(1)).collect::<Vec<_>>()
        {
            return Err(invalid(
                "Sampler state does not match prepared dataset or algorithm",
            ));
        }
        Ok(())
    }
    /// The generation and initial sampler state were verified at the task boundary.
    pub(crate) fn restore_verified(prepared: Arc<PreparedProject>, state: SamplerState) -> Self {
        Self {
            prepared,
            state,
            check_frame_hashes: false,
        }
    }
    pub fn snapshot(&self) -> SamplerState {
        self.state.clone()
    }
    pub(crate) fn at_epoch_end(&self) -> bool {
        self.state.cursor == self.state.permutation.len()
    }
    fn read_frame(&self, index: usize) -> Result<super::geometry::PreparedFrame> {
        if self.check_frame_hashes {
            self.prepared.read_frame(index)
        } else {
            self.prepared.read_verified_frame(index)
        }
    }
    /// Returns None at the epoch boundary. Call advance_epoch explicitly, after checkpointing.
    pub fn next_batch(&mut self, batch_size: usize) -> Result<Option<TrainingBatch>> {
        if batch_size == 0 || batch_size > 1024 {
            return Err(invalid("Batch size must be 1..=1024"));
        }
        if self.at_epoch_end() {
            return Ok(None);
        }
        let mut next = self.state.clone();
        let end = (next.cursor + batch_size).min(next.permutation.len());
        let mut image = vec![];
        let mut target = vec![];
        let mut mask = vec![];
        let mut audio = vec![];
        let mut pairs = vec![];
        let mut references = vec![];
        for cursor in next.cursor..end {
            let i = next.permutation[cursor];
            let reference = choice(&mut next.reference_state, self.prepared.usable_frames);
            next.reference_draws = next
                .reference_draws
                .checked_add(1)
                .ok_or_else(|| invalid("Reference counter overflow"))?;
            let r = self.read_frame(reference)?;
            references.push(reference);
            pairs.push([i, i + 1]);
            for index in [i, i + 1] {
                let f = self.read_frame(index)?;
                image.extend_from_slice(&r.target);
                image.extend_from_slice(&f.masked);
                target.extend(f.target);
                mask.extend(f.mask);
                audio.extend(audio_window(
                    &self.prepared.features,
                    index,
                    0,
                    self.prepared.usable_frames,
                )?);
            }
        }
        let b = pairs.len();
        let batch = TrainingBatch {
            image: HostTensor::new([b, 2, 6, 144, 144], image)?,
            audio: HostTensor::new([b, 2, 40, 1024], audio)?,
            target: HostTensor::new([b, 2, 3, 144, 144], target)?,
            mask: HostTensor::new([b, 2, 1, 144, 144], mask)?,
            pairs,
            references,
        };
        next.cursor = end;
        self.state = next;
        Ok(Some(batch))
    }
    pub fn advance_epoch(&mut self) -> Result<()> {
        if !self.at_epoch_end() {
            return Err(invalid("Cannot advance an incomplete epoch"));
        }
        self.state.epoch = self
            .state
            .epoch
            .checked_add(1)
            .ok_or_else(|| invalid("Epoch overflow"))?;
        self.state.cursor = 0;
        shuffle(&mut self.state);
        Ok(())
    }
    pub fn prefetch(
        self,
        batch_size: usize,
        capacity: usize,
        cancel: Cancellation,
    ) -> Result<PrefetchSampler> {
        if capacity == 0 || capacity > 32 || batch_size == 0 || batch_size > 1024 {
            return Err(invalid(
                "Prefetch capacity must be 1..=32 and batch size 1..=1024",
            ));
        }
        let committed = self.clone();
        let consumer_cancel = cancel.clone();
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let stop = Cancellation::new();
        let worker_stop = stop.clone();
        let worker = thread::spawn(move || {
            let mut sampler = self;
            loop {
                if cancel.is_cancelled() || worker_stop.is_cancelled() {
                    break;
                }
                let result = sampler.next_batch(batch_size);
                let done = !matches!(&result, Ok(Some(_)));
                let mut item = (result, sampler.snapshot());
                loop {
                    if cancel.is_cancelled() || worker_stop.is_cancelled() {
                        return;
                    }
                    match sender.try_send(item) {
                        Ok(()) => break,
                        Err(mpsc::TrySendError::Full(value)) => {
                            item = value;
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(_) => return,
                    }
                }
                if done {
                    break;
                }
            }
        });
        Ok(PrefetchSampler {
            committed,
            receiver,
            stop,
            cancel: consumer_cancel,
            ended: false,
            worker: Some(worker),
        })
    }
}
fn shuffle(state: &mut SamplerState) {
    state.permutation.sort_unstable();
    for i in (1..state.permutation.len()).rev() {
        let j = choice(&mut state.shuffle_state, i + 1);
        state.permutation.swap(i, j);
    }
}
type PrefetchItem = (Result<Option<TrainingBatch>>, SamplerState);
pub struct PrefetchSampler {
    committed: PairSampler,
    receiver: mpsc::Receiver<PrefetchItem>,
    stop: Cancellation,
    cancel: Cancellation,
    ended: bool,
    worker: Option<JoinHandle<()>>,
}
impl PrefetchSampler {
    pub fn snapshot(&self) -> SamplerState {
        self.committed.snapshot()
    }
    pub(crate) fn into_sampler(self) -> PairSampler {
        self.committed.clone()
    }
    pub fn next_batch(&mut self) -> Result<Option<TrainingBatch>> {
        crate::media::ffmpeg::check(&self.cancel)?;
        if self.ended {
            return Ok(None);
        }
        let (result, state) = self.receiver.recv().map_err(|_| {
            AppError::new(
                ErrorCode::Cancelled,
                "dataset-prefetch",
                "Prefetch cancelled or stopped",
            )
        })?;
        crate::media::ffmpeg::check(&self.cancel)?;
        if matches!(result, Ok(None)) {
            self.ended = true;
        }
        if result.is_ok() {
            self.committed.state = state;
        }
        result
    }
}
impl Drop for PrefetchSampler {
    fn drop(&mut self) {
        self.stop.cancel();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::prepared::FrameRecord;
    use std::{fs, path::PathBuf};
    fn prepared() -> PreparedProject {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../.artifacts/task8-sampler/{}",
            ProjectId::new()
        ));
        fs::create_dir_all(root.join("frames")).unwrap();
        let root = fs::canonicalize(root).unwrap();
        let mut frames = vec![];
        for i in 0..6 {
            let file = PathBuf::from(format!("frames/{i}.f32"));
            let mut bytes = vec![];
            for n in 0..(4 * 144 * 144) {
                bytes.extend_from_slice(&(i as f32 + n as f32 / 1000.).to_le_bytes());
            }
            fs::write(root.join(&file), &bytes).unwrap();
            frames.push(FrameRecord {
                file,
                sha256: crate::data::prepared::digest(&bytes),
                landmarks: vec![],
            });
        }
        PreparedProject {
            project_directory: root.clone(),
            project_id: ProjectId::new(),
            identity: "test".into(),
            preprocessing: feathertalk_core::PreprocessingIdentity {
                version: "v".into(),
                source_sha256: "s".into(),
                config_sha256: "c".into(),
                cache_directory: PathBuf::from("cache"),
                component_identities: vec![],
            },
            canonical_video: root.join("v"),
            canonical_audio: root.join("a"),
            canonical_frame_count: 6,
            feature_frame_count: 6,
            usable_frames: 6,
            frames,
            features: Arc::new((0..6 * 2048).map(|n| n as f32).collect()),
            generation: root,
            manifest_sha256: "test".into(),
        }
    }
    #[test]
    fn sampler_replays_after_restore_and_keeps_tail() {
        let p = prepared();
        let mut a = PairSampler::new(&p, 42).unwrap();
        assert_eq!(word(&mut 0), 16294208416658607535);
        assert_eq!(a.snapshot().permutation, vec![4, 0, 1, 2, 3]);
        let first = a.next_batch(2).unwrap().unwrap();
        assert_eq!(first.image.dims, [2, 2, 6, 144, 144]);
        assert_eq!(first.audio.dims, [2, 2, 40, 1024]);
        assert_eq!(first.references, vec![1, 4]);
        assert_eq!(first.pairs, vec![[4, 5], [0, 1]]);
        for pair in 0..2 {
            let base = pair * 12 * 144 * 144;
            assert_eq!(
                &first.image.values[base..base + 3 * 144 * 144],
                &first.image.values[base + 6 * 144 * 144..base + 9 * 144 * 144]
            );
        }
        let state = a.snapshot();
        let encoded = serde_json::to_vec(&state).unwrap();
        let mut b = PairSampler::restore(&p, serde_json::from_slice(&encoded).unwrap()).unwrap();
        let second = a.next_batch(2).unwrap().unwrap();
        assert_eq!(Some(second.clone()), b.next_batch(2).unwrap());
        assert_eq!(second.references, vec![3, 1]);
        let tail = a.next_batch(2).unwrap().unwrap();
        assert_eq!(Some(tail.clone()), b.next_batch(2).unwrap());
        assert_eq!(tail.pairs, vec![[3, 4]]);
        assert_eq!(tail.references, vec![3]);
        assert_eq!(tail.image.dims, [1, 2, 6, 144, 144]);
        assert_eq!(a.next_batch(2).unwrap(), None);
        a.advance_epoch().unwrap();
        b.advance_epoch().unwrap();
        assert_eq!(a.next_batch(3).unwrap(), b.next_batch(3).unwrap());
        let mut bad = p.clone();
        bad.identity = "changed".into();
        assert!(PairSampler::restore(&bad, state).is_err());
    }
    #[test]
    fn prefetch_does_not_change_committed_state_before_receive() {
        let p = prepared();
        let mut direct = PairSampler::new(&p, 7).unwrap();
        let expected = direct.next_batch(2).unwrap().unwrap();
        let cancel = Cancellation::new();
        let mut pf = PairSampler::new(&p, 7)
            .unwrap()
            .prefetch(2, 2, cancel)
            .unwrap();
        assert_eq!(pf.snapshot().cursor, 0);
        assert_eq!(pf.next_batch().unwrap().unwrap(), expected);
        assert_eq!(pf.snapshot().cursor, 2);
    }
    #[test]
    fn cancel_prefetch_preserves_snapshot_and_unrestricted_references() {
        let p = prepared();
        let mut allowed = std::collections::BTreeSet::new();
        let mut equals_target = false;
        for seed in 0..16 {
            let mut sampler = PairSampler::new(&p, seed).unwrap();
            let state = sampler.snapshot();
            assert!(sampler.next_batch(0).is_err());
            assert_eq!(state, sampler.snapshot());
            let batch = sampler.next_batch(5).unwrap().unwrap();
            for (reference, pair) in batch.references.into_iter().zip(batch.pairs) {
                allowed.insert(reference);
                equals_target |= pair.contains(&reference);
            }
        }
        assert_eq!(allowed, (0..6).collect());
        assert!(equals_target);
        let cancel = Cancellation::new();
        let sampler = PairSampler::new(&p, 9).unwrap();
        let state = sampler.snapshot();
        let mut pf = sampler.prefetch(1, 1, cancel.clone()).unwrap();
        cancel.cancel();
        assert_eq!(pf.next_batch().unwrap_err().code, ErrorCode::Cancelled);
        assert_eq!(pf.snapshot(), state);
        drop(pf);
    }
    #[test]
    fn empty_and_singleton_have_no_pairs() {
        for count in [0, 1] {
            let mut p = prepared();
            p.frames.truncate(count);
            p.canonical_frame_count = count;
            p.feature_frame_count = count;
            p.usable_frames = count;
            p.features = Arc::new(vec![0.; count * 2048]);
            assert_eq!(
                PairSampler::new(&p, 1).unwrap().next_batch(8).unwrap(),
                None
            );
        }
    }
}
