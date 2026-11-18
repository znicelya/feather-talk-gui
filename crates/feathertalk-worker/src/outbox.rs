//! Bounded ordered output and task admission share one lock. Waiting producers release it.
use feathertalk_core::*;
use std::{
    collections::{HashSet, VecDeque},
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};
const COMPUTE_CAPACITY: usize = 64;
const TOTAL_CAPACITY: usize = 128;
struct State {
    seq: u64,
    active: Option<TaskId>,
    used: HashSet<TaskId>,
    events: VecDeque<WorkerEvent>,
    metrics: Option<Instant>,
    closed: bool,
}
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}
#[derive(Clone)]
pub struct EventOwner(Arc<Shared>);
pub struct EventReceiver(Arc<Shared>);
pub fn event_channel() -> (EventOwner, EventReceiver) {
    let shared = Arc::new(Shared {
        state: Mutex::new(State {
            seq: 0,
            active: None,
            used: HashSet::new(),
            events: VecDeque::new(),
            metrics: None,
            closed: false,
        }),
        changed: Condvar::new(),
    });
    (EventOwner(shared.clone()), EventReceiver(shared))
}
pub(crate) fn terminal(p: &EventPayload) -> bool {
    matches!(
        p,
        EventPayload::Completed { .. }
            | EventPayload::Cancelled { .. }
            | EventPayload::Failed { .. }
    )
}
impl State {
    fn push(&mut self, task: Option<TaskId>, payload: EventPayload) -> Result<()> {
        if self.closed {
            return Err(protocol_error("Output closed"));
        }
        if self.events.len() == TOTAL_CAPACITY {
            self.closed = true;
            return Err(protocol_error(
                "Fatal transport overload: control output capacity exhausted",
            ));
        }
        let event = WorkerEvent::new(task, self.seq, payload)?;
        self.seq = self
            .seq
            .checked_add(1)
            .ok_or_else(|| protocol_error("Sequence exhausted"))?;
        self.events.push_back(event);
        Ok(())
    }
}
impl EventOwner {
    pub fn start(&self, task: TaskId, kind: JobKind) -> Result<()> {
        let mut s = self.0.state.lock().unwrap();
        if s.active.is_some() {
            return Err(AppError::new(
                ErrorCode::Busy,
                "worker",
                "A job is already running",
            ));
        }
        if s.used.contains(&task) {
            return Err(protocol_error("Task ID reused"));
        }
        let result = s.push(Some(task), EventPayload::Started { job: kind });
        if result.is_ok() {
            s.used.insert(task);
            s.active = Some(task);
            s.metrics = None;
        }
        self.0.changed.notify_all();
        result
    }
    pub fn emit(&self, task: TaskId, payload: EventPayload) -> Result<()> {
        if matches!(
            payload,
            EventPayload::Ready { .. }
                | EventPayload::RequestRejected { .. }
                | EventPayload::Started { .. }
        ) {
            return Err(protocol_error("Invalid task callback"));
        }
        let mut s = self.0.state.lock().unwrap();
        if s.active != Some(task) {
            return Err(protocol_error("Late or foreign task event"));
        }
        while !s.closed && s.events.len() >= COMPUTE_CAPACITY {
            s = self.0.changed.wait(s).unwrap();
        }
        if s.active != Some(task) {
            return Err(protocol_error("Late or foreign task event"));
        }
        if s.closed {
            return Err(protocol_error("Output closed"));
        }
        if matches!(payload, EventPayload::Metrics(_)) {
            if s.metrics
                .is_some_and(|t| t.elapsed() < Duration::from_millis(500))
            {
                return Ok(());
            }
            s.metrics = Some(Instant::now());
        }
        let final_event = terminal(&payload);
        s.push(Some(task), payload)?;
        // Admission is released in the same critical section as terminal publication.
        if final_event {
            s.active = None;
        }
        self.0.changed.notify_all();
        Ok(())
    }
    pub(crate) fn active(&self) -> bool {
        self.0.state.lock().unwrap().active.is_some()
    }
    pub(crate) fn control(&self, payload: EventPayload) -> Result<()> {
        let result = self.0.state.lock().unwrap().push(None, payload);
        self.0.changed.notify_all();
        result
    }
    pub fn close(&self) {
        self.0.state.lock().unwrap().closed = true;
        self.0.changed.notify_all();
    }
    pub fn is_closed(&self) -> bool {
        self.0.state.lock().unwrap().closed
    }
}
impl EventReceiver {
    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> std::result::Result<WorkerEvent, std::sync::mpsc::RecvTimeoutError> {
        let s = self.0.state.lock().unwrap();
        let (mut s, _) = self
            .0
            .changed
            .wait_timeout_while(s, timeout, |s| s.events.is_empty() && !s.closed)
            .unwrap();
        if let Some(event) = s.events.pop_front() {
            self.0.changed.notify_all();
            return Ok(event);
        }
        Err(if s.closed {
            std::sync::mpsc::RecvTimeoutError::Disconnected
        } else {
            std::sync::mpsc::RecvTimeoutError::Timeout
        })
    }
    pub fn try_iter(&self) -> impl Iterator<Item = WorkerEvent> + '_ {
        std::iter::from_fn(|| self.recv_timeout(Duration::ZERO).ok())
    }
}
impl Drop for EventReceiver {
    fn drop(&mut self) {
        EventOwner(self.0.clone()).close();
    }
}
