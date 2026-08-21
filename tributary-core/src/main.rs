mod mlx_client;
mod protocol;

use clap::Parser;
use std::io::Write;
use std::time::Instant;
use mlx_client::{MlxClient, Tensor};
use protocol::{Frame, MsgType, read_frame, write_frame};
use tokio::net::{TcpListener, TcpStream};

#[derive(clap::ValueEnum, Clone, PartialEq)]
enum Mode {
    Single,
    Coordinator,
    Worker,
}

#[derive(Parser)]
#[command(name = "tributary")]
#[command(about = "Distributed LLM inference across Apple Silicon devices")]
struct Args {
    #[arg(long, value_enum, default_value = "single")]
    mode: Mode,

    #[arg(long)]
    prompt: Option<String>,
    
    #[arg(long, default_value = "200")]
    max_tokens: u32,

    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    
    #[arg(long, default_value = "http://localhost:8765")]
    mlx_server: String,

    #[arg(long)]
    mlx_server_b: Option<String>,

    #[arg(long)]
    worker: Option<String>,

    #[arg(long)]
    listen: Option<u16>,

    #[arg(long)]
    timing_csv: Option<String>,

    #[arg(long, default_value_t = 0)]
    spec_k: u32,

    #[arg(long)]
    draft_model: Option<String>,
}

#[derive(Clone, Copy, Default)]
struct StepTiming {
    local_us: u128,
    serialize_us: u128,
    roundtrip_us: u128,
    worker_us: u128,
    network_us: u128, 
    deserialize_us: u128,
    sample_us: u128,
    activation_bytes: usize,
    logits_bytes: usize,
}

fn percentile(sorted: &[u128], p: usize) -> u128 {
    let n = sorted.len();
    let rank = (p * n).div_ceil(100).max(1);
    sorted[rank - 1]
}

fn summarize(label: &str, values: &[u128]) {
    if values.is_empty() { return; }
    let mut v = values.to_vec();
    v.sort_unstable();
    let n = v.len();
    let mean = v.iter().sum::<u128>() / n as u128;
    let p50 = percentile(&v, 50);
    let p90 = percentile(&v, 90);
    let max = v[n - 1];
    eprintln!("{label:<12} {mean:>8} {p50:>8} {p90:>8} {max:>8}");
}

fn print_timing_summary(prefill: &StepTiming, steps: &[StepTiming]) {
    eprintln!("\nper-token timing over {} decode tokens (µs)", steps.len());
    eprintln!("{:<12} {:>8} {:>8} {:>8} {:>8}", "stage", "mean", "p50", "p90", "max");
    let col = |f: fn(&StepTiming) -> u128| steps.iter().map(f).collect::<Vec<_>>();
    summarize("local",       &col(|s| s.local_us));
    summarize("serialize",   &col(|s| s.serialize_us));
    summarize("network",     &col(|s| s.network_us));
    summarize("worker",      &col(|s| s.worker_us));
    summarize("deserialize", &col(|s| s.deserialize_us));
    summarize("sample",      &col(|s| s.sample_us));
    summarize("roundtrip",   &col(|s| s.roundtrip_us));
    eprintln!(
        "prefill: local={} net={} worker={} sample={} | activation={}B logits={}B",
        prefill.local_us, prefill.network_us, prefill.worker_us, prefill.sample_us,
        prefill.activation_bytes, prefill.logits_bytes
    );
}

fn write_timing_csv(path: &str, prefill: &StepTiming, steps: &[StepTiming]) -> std::io::Result<()> {
    let mut s = String::from(
        "token,phase,local_us,serialize_us,network_us,worker_us,roundtrip_us,deserialize_us,sample_us,activation_bytes,logits_bytes\n",
    );
    let mut row = |i: usize, phase: &str, t: &StepTiming| {
        s.push_str(&format!(
            "{i},{phase},{},{},{},{},{},{},{},{},{}\n",
            t.local_us, t.serialize_us, t.network_us, t.worker_us, t.roundtrip_us,
            t.deserialize_us, t.sample_us, t.activation_bytes, t.logits_bytes
        ));
    };
    row(0, "prefill", prefill);
    for (i, t) in steps.iter().enumerate() {
        row(i + 1, "decode", t);
    }
    std::fs::write(path, s)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.mode {
        Mode::Single if args.spec_k > 0 => run_spec_loop(&args).await,
        Mode::Single => run_loop(&args).await,
        Mode::Coordinator => run_coordinator(&args).await,
        Mode::Worker => run_worker(&args).await,
    }
}

async fn run_spec_loop(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = args.prompt.as_ref().ok_or("spec mode requires --prompt")?;
    let draft_url = args.draft_model.as_ref().ok_or("--spec-k requires --draft-model")?;
    let k = args.spec_k;
    let target = MlxClient::new(args.mlx_server.clone());
    let draft = MlxClient::new(draft_url.clone());

    target.reset().await?;
    draft.reset().await?;

    let (ids, eos) = target.tokenize(prompt).await?;
    let t_start = Instant::now();

    let ht = target.forward(&target.embed(&ids).await?, "prefill").await?;
    let _ = draft.forward(&draft.embed(&ids).await?, "prefill").await?;
    let mut cur = *target
        .argmax(&ht)
        .await?
        .last()
        .ok_or("empty prefill argmax")?;

    let mut rounds: u64 = 0;
    let mut accepted_sum: u64 = 0;
    let mut draft_us: u128 = 0;
    let mut verify_us: u128 = 0;
    let mut token_count: u32 = 1;

    print!("{}", target.detokenize(&[cur]).await?);
    std::io::stdout().flush()?;
    let t_first = Instant::now();

    if cur != eos {
        'outer: while token_count < args.max_tokens {
            let t_d = Instant::now();
            let x = draft.draft(cur, k).await?;
            draft_us += t_d.elapsed().as_micros();

            let t_v = Instant::now();
            let mut verify_ids = Vec::with_capacity(x.len() + 1);
            verify_ids.push(cur);
            verify_ids.extend_from_slice(&x);
            let h = target.forward(&target.embed(&verify_ids).await?, "decode").await?;
            let t = target.argmax(&h).await?;
            verify_us += t_v.elapsed().as_micros();

            let mut a = 0usize;
            while a < k as usize && t[a] == x[a] {
                a += 1;
            }
            let mut emitted: Vec<u32> = x[..a].to_vec();
            emitted.push(t[a]);

            let trim = k - a as u32;
            target.trim(trim).await?;
            draft.trim(trim).await?;

            rounds += 1;
            accepted_sum += a as u64;

            for &tok in &emitted {
                print!("{}", target.detokenize(&[tok]).await?);
                std::io::stdout().flush()?;
                token_count += 1;
                cur = tok;
                if tok == eos || token_count >= args.max_tokens {
                    break 'outer;
                }
            }
        }
    }

    let elapsed = t_first.elapsed().as_secs_f32();
    let gen_tokens = token_count.saturating_sub(1); // tokens after #0, comparable to baseline
    let alpha = if rounds > 0 { accepted_sum as f64 / (rounds as f64 * k as f64) } else { 0.0 };
    let mean_acc = if rounds > 0 { gen_tokens as f64 / rounds as f64 } else { 0.0 };
    eprintln!("\n");
    eprintln!(
        "tributary (spec K={k}) | {token_count} tokens | {:.1} tok/s | ttft: {:.2}s",
        if elapsed > 0.0 { gen_tokens as f32 / elapsed } else { 0.0 },
        (t_first - t_start).as_secs_f32()
    );
    eprintln!(
        "spec | rounds={rounds} accept_rate α={alpha:.3} mean_accepted/verify={mean_acc:.2} | \
         draft={}µs verify={}µs (mean/round)",
        if rounds > 0 { draft_us / rounds as u128 } else { 0 },
        if rounds > 0 { verify_us / rounds as u128 } else { 0 },
    );
    Ok(())
}

async fn run_loop (args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = args.prompt.as_ref().ok_or("single mode requires --prompt")?;
    let mut servers = vec![MlxClient::new(args.mlx_server.clone())];
    if let Some(url) = &args.mlx_server_b {
        servers.push(MlxClient::new(url.clone()));
    }
    let first = &servers[0];
    let last = servers.last().unwrap();
    for server in &servers {
        server.reset().await?;
    }
    let (ids, eos_token_id) = first.tokenize(prompt).await?;
    let t_start = Instant::now();

    let mut hidden = first.embed(&ids).await?;
    for server in &servers {
        hidden = server.forward(&hidden, "prefill").await?;
    }
    let logits = last.logits(&hidden).await?;
    let mut next_id = first.sample(&logits, args.temperature).await?;
    let mut token_count: u32 = 0;
    let t_first = Instant::now();

    for _ in 0..args.max_tokens {
        let token = first.detokenize(&[next_id]).await?;
        print!("{}", token);
        std::io::stdout().flush()?;
        if next_id == eos_token_id {
            break;
        }

        let mut h = first.embed(&[next_id]).await?;
        for server in &servers {
            h = server.forward(&h, "decode").await?;
        }
        let logits = last.logits(&h).await?;
        next_id = first.sample(&logits, args.temperature).await?;
        token_count += 1;
    }

    let elapsed = t_first.elapsed().as_secs_f32();
    eprintln!("\n");
    eprintln!(
        "tributary | {} tokens | {:.1} tok/s | ttft: {:.2}s",
        token_count,
        if elapsed > 0.0 { token_count as f32 / elapsed } else { 0.0 },
        (t_first - t_start).as_secs_f32()
    );

    Ok(())
}

async fn run_coordinator(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = args.prompt.as_ref().ok_or("coordinator requires --prompt")?;
    let worker_addr = args.worker.as_ref().ok_or("coordinator requires --worker")?;
    let local = MlxClient::new(args.mlx_server.clone());
    let mut stream: TcpStream = TcpStream::connect(worker_addr).await?;
    let mut seq: u32 = 0;

    let coord = local.info().await?;
    write_frame(&mut stream, &Frame::control(MsgType::Info, seq)).await?;
    let winfo = read_frame(&mut stream).await?;
    if winfo.msg_type != MsgType::Info || winfo.shape.len() != 3 {
        return Err("worker sent a malformed Info reply".into());
    }
    let (worker_start, worker_end, worker_num) = (winfo.shape[0], winfo.shape[1], winfo.shape[2]);
    if !coord.is_first {
        return Err(format!(
            "coordinator's local server must start at layer 0, got {}..{}",
            coord.start_layer, coord.end_layer
        ).into());
    }
    if coord.num_layers != worker_num {
        return Err(format!(
            "model mismatch: coordinator has {} layers, worker has {}",
            coord.num_layers, worker_num
        ).into());
    }
    if coord.end_layer != worker_start {
        return Err(format!(
            "layer split is not contiguous: coordinator ends at {}, worker starts at {} (gap or overlap)",
            coord.end_layer, worker_start
        ).into());
    }
    if worker_end != worker_num {
        return Err(format!(
            "worker must end at the final layer {}, got {}",
            worker_num, worker_end
        ).into());
    }
    eprintln!(
        "split OK: coordinator 0..{} + worker {}..{} = {} layers",
        coord.end_layer, worker_start, worker_end, worker_num
    );
    seq += 1;

    local.reset().await?;
    write_frame(&mut stream, &Frame::control(MsgType::ResetCache, seq)).await?;
    seq += 1;

    let (ids, eos_token_id) = local.tokenize(prompt).await?;
    let t_start = Instant::now();
    let t_local = Instant::now();
    let hidden = local.embed(&ids).await?;
    let hidden = local.forward(&hidden, "prefill").await?;
    let local_us = t_local.elapsed().as_micros();

    let (mut next_id, mut prefill_timing) = exchange(&mut stream, &local, &hidden, MsgType::Prefill, seq, args.temperature).await?;
    prefill_timing.local_us = local_us;
    seq += 1;

    let mut timings: Vec<StepTiming> = Vec::new();

    let mut token_count: u32 = 0;
    let t_first = Instant::now();

    for _ in 0..args.max_tokens {
        let token = local.detokenize(&[next_id]).await?;
        print!("{}", token);
        std::io::stdout().flush()?;
        if next_id == eos_token_id {
            break;
        }

        let t_local = Instant::now();
        let h = local.embed(&[next_id]).await?;
        let h = local.forward(&h, "decode").await?;
        let local_us = t_local.elapsed().as_micros();

        let (nid, mut st) = exchange(&mut stream, &local, &h, MsgType::DecodeStep, seq, args.temperature).await?;
        st.local_us = local_us;
        timings.push(st);
        next_id = nid;
        seq += 1;
        token_count += 1;
    }

    let elapsed = t_first.elapsed().as_secs_f32();
    eprintln!("\n");
    eprintln!(
        "tributary (coordinator) | {} tokens | {:.1} tok/s | ttft: {:.2}s",
        token_count,
        if elapsed > 0.0 { token_count as f32 / elapsed } else { 0.0 },
        (t_first - t_start).as_secs_f32()
    );
    print_timing_summary(&prefill_timing, &timings);
    if let Some(path) = &args.timing_csv {
        write_timing_csv(path, &prefill_timing, &timings)?;
        eprintln!("wrote timing CSV to {path}");
    }
    Ok(())
}

async fn exchange(
    stream: &mut TcpStream,
    local: &MlxClient,
    hidden: &Tensor,
    msg_type: MsgType,
    seq: u32,
    temperature: f32,
) -> Result<(u32, StepTiming), Box<dyn std::error::Error>> {
    let mut t = StepTiming::default();

    let t_ser = Instant::now();
    let frame_out = Frame::from_tensor(msg_type, seq, hidden);
    t.serialize_us = t_ser.elapsed().as_micros();
    t.activation_bytes = frame_out.payload.len();

    let t_rt = Instant::now();
    write_frame(stream, &frame_out).await?;
    let reply = recv_logits(stream, seq).await?;
    t.roundtrip_us = t_rt.elapsed().as_micros();
    t.worker_us = reply.worker_compute_us as u128;
    t.network_us = t.roundtrip_us.saturating_sub(t.worker_us);
    t.logits_bytes = reply.payload.len();

    let t_de = Instant::now();
    let logits = reply.into_tensor();
    t.deserialize_us = t_de.elapsed().as_micros();

    let t_s = Instant::now();
    let next_id = local.sample(&logits, temperature).await?;
    t.sample_us = t_s.elapsed().as_micros();

    Ok((next_id, t))

}

async fn recv_logits(stream: &mut TcpStream, expected_seq: u32) -> Result<Frame, Box<dyn std::error::Error>> {
    let frame = read_frame(stream).await?;
    if frame.msg_type != MsgType::Logits {
        return Err(format!("expected Logits frame, got {:?}", frame.msg_type).into());
    }
    if frame.seq != expected_seq {
        return Err(format!("seq mismatch: sent {expected_seq}, got {}", frame.seq).into());
    }
    Ok(frame)
}

async fn run_worker(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let port = args.listen.ok_or("worker requires --listen")?;
    let local = MlxClient::new(args.mlx_server.clone());
    let listener = TcpListener::bind(("0.0.0.0", port)).await?;
    eprintln!("worker listening on 0.0.0.0:{port}");

    loop {
        let (mut stream, peer) = listener.accept().await?;
        eprintln!("coordinator connected from {peer}");

        loop {
            let frame = match read_frame(&mut stream).await {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("worker: connection closed ({e})");
                    break;
                }
            };

            match frame.msg_type {
                MsgType::Info => {
                    let info = local.info().await?;
                    let mut reply = Frame::control(MsgType::Info, frame.seq);
                    reply.shape = vec![info.start_layer, info.end_layer, info.num_layers];
                    write_frame(&mut stream, &reply).await?;
                }
                MsgType::ResetCache => {
                    local.reset().await?;
                }
                MsgType::Prefill | MsgType::DecodeStep => {
                    let seq = frame.seq;
                    let mode = if frame.msg_type == MsgType::Prefill { "prefill" } else { "decode" };

                    let t_compute = Instant::now();
                    let tensor = frame.into_tensor();
                    let hidden = local.forward(&tensor, mode).await?;
                    let logits = local.logits(&hidden).await?;
                    let compute_us = t_compute.elapsed().as_micros() as u64;

                    let mut reply = Frame::from_tensor(MsgType::Logits, seq, &logits);
                    reply.worker_compute_us = compute_us;
                    write_frame(&mut stream, &reply).await?;
                }
                MsgType::Logits => {
                    return Err("worker received an unexpected Logits frame".into());
                }
            }
        }
        local.reset().await?;
        eprintln!("worker: ready for next connection");
    }
}