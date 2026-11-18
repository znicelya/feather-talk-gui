use feathertalk_core::*;
use std::io::{BufReader, Cursor};
#[test]
fn fragmented_and_multiple_records() {
    let mut r = BufReader::with_capacity(2, Cursor::new(b"abc\ndef\n"));
    assert_eq!(read_frame(&mut r).unwrap(), Some(b"abc".to_vec()));
    assert_eq!(read_frame(&mut r).unwrap(), Some(b"def".to_vec()));
    assert_eq!(read_frame(&mut r).unwrap(), None);
}
#[test]
fn limit_applies_without_delimiter() {
    let mut r = BufReader::with_capacity(7, Cursor::new(vec![b'x'; MAX_MESSAGE_BYTES + 1]));
    assert_eq!(read_frame(&mut r).unwrap_err().code, ErrorCode::Protocol);
    assert!(r.get_ref().position() <= (MAX_MESSAGE_BYTES + 7) as u64);
}
#[test]
fn exact_limit_and_truncated_eof() {
    let mut bytes = vec![b'x'; MAX_MESSAGE_BYTES];
    bytes.push(b'\n');
    assert_eq!(
        read_frame(&mut Cursor::new(bytes)).unwrap().unwrap().len(),
        MAX_MESSAGE_BYTES
    );
    assert_eq!(
        read_frame(&mut Cursor::new(b"{}")).unwrap_err().code,
        ErrorCode::Protocol
    );
}

#[test]
fn rejection_is_control_and_validates_attempted_owner() {
    let error = AppError::new(ErrorCode::Busy, "control", "busy");
    let payload = EventPayload::RequestRejected {
        request: RequestKind::Start,
        requested_task_id: Some(TaskId::new()),
        error: error.clone(),
    };
    assert!(WorkerEvent::new(None, 1, payload.clone()).is_ok());
    assert!(WorkerEvent::new(Some(TaskId::new()), 1, payload).is_err());
    for request in [RequestKind::Hello, RequestKind::Shutdown] {
        assert!(
            WorkerEvent::new(
                None,
                1,
                EventPayload::RequestRejected {
                    request,
                    requested_task_id: Some(TaskId::new()),
                    error: error.clone()
                }
            )
            .is_err()
        );
    }
    for request in [RequestKind::Start, RequestKind::Cancel] {
        assert!(
            WorkerEvent::new(
                None,
                1,
                EventPayload::RequestRejected {
                    request,
                    requested_task_id: None,
                    error: error.clone()
                }
            )
            .is_err()
        );
    }
}
