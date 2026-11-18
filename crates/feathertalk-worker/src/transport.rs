use feathertalk_core::*;
use std::io::Write;
pub fn write_event(writer: &mut impl Write, event: &WorkerEvent) -> Result<()> {
    let bytes = serde_json::to_vec(event).map_err(protocol_error)?;
    if bytes.len() > MAX_MESSAGE_BYTES {
        return Err(protocol_error("Outbound event exceeds limit"));
    }
    writer
        .write_all(&bytes)
        .and_then(|_| writer.write_all(b"\n"))
        .and_then(|_| writer.flush())
        .map_err(protocol_error)
}
