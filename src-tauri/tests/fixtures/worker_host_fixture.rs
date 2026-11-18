// Deliberately adversarial transport fixture. Performs no neural computation.
use std::io::{self,BufRead,Write};
fn event(seq:u64,task:&str,payload:&str){println!(r#"{{"protocol_version":1,"task_id":{task},"seq":{seq},"payload":{payload}}}"#);io::stdout().flush().unwrap();}
fn main(){
 let input=io::stdin();let mut lines=input.lock().lines();let context=lines.next().unwrap().unwrap();let _hello=lines.next().unwrap().unwrap();
 event(0,"null",r#"{"type":"ready","capabilities":{"engine_version":"test","components":[],"training":false,"generation":false},"devices":[]}"#);
 let start=lines.next().unwrap().unwrap();let id=start.split("\"task_id\":\"").nth(1).unwrap().split('"').next().unwrap();let task=format!("\"{id}\"");
 if context.contains("pre-start-control") {
  event(1,"null",&format!(r#"{{"type":"request_rejected","request":"start","requested_task_id":{task},"error":{{"code":"busy","stage":"control","message":"busy","evidence":[],"actions":[],"log_path":null}}}}"#));
 }
 if context.contains("gen-") { event(2,&task,r#"{"type":"started","job":"generate"}"#); }
 else { event(2,&task,r#"{"type":"started","job":"train"}"#); }
 event(3,&task,r#"{"type":"progress","stage":"fixture-compute","completed":1,"total":null}"#);
 if context.contains("gen-") {
  for mode in ["gen-metrics", "gen-checkpoint", "gen-cancel-checkpoint", "gen-wrong-output", "gen-valid-output"] {
   if context.contains(mode) { let payload = std::fs::read_to_string(std::env::current_exe().unwrap().parent().unwrap().join(format!("{mode}.json"))).unwrap(); event(4,&task,&payload); }
  }
 }else if context.contains("quiet-cancel") {
  let cancel = lines.next().unwrap().unwrap(); assert!(cancel.contains("cancel"));
  event(4,&task,r#"{"type":"cancelled","checkpoint":null}"#);
 }else if context.contains("flood-drop") {
  for seq in 4..10000 { event(seq,&task,r#"{"type":"progress","stage":"flood","completed":0,"total":null}"#); }
 }else if context.contains("control-error"){
  event(4,"null",r#"{"type":"request_rejected","request":"cancel","requested_task_id":"00000000-0000-4000-8000-000000000001","error":{"code":"protocol","stage":"control","message":"stale cancel","evidence":[],"actions":[],"log_path":null}}"#);
  event(5,&task,r#"{"type":"cancelled","checkpoint":null}"#);
  // A late event must not produce a second terminal.
  event(6,&task,r#"{"type":"progress","stage":"late","completed":0,"total":null}"#);
 }else{
  eprintln!("{}", "stderr-only-".repeat(9000));eprintln!("{{\"type\":\"completed\"}} actual diagnostic tail");
  if context.contains("oversized"){print!("{}","x".repeat(1024*1024+1));io::stdout().flush().unwrap();}
  if context.contains("wrong-seq"){event(1,&task,r#"{"type":"cancelled","checkpoint":null}"#);}
  if context.contains("foreign"){event(4,r#""00000000-0000-4000-8000-000000000001""#,r#"{"type":"cancelled","checkpoint":null}"#);}
  if context.contains("malformed"){println!("{{bad json");io::stdout().flush().unwrap();}
  std::process::exit(23);
 }
}
