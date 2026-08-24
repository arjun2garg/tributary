mod mlx_client;
mod protocol;

use clap::Parser;
use std::io::Write;
use std::time::Instant;
use mlx_client::{MlxClient, Tensor};
use protocol::{Frame, MsgType, read_frame, write_frame, F_LAZY_SAMPLE, F_SPEC_NAIVE, F_SAMPLED, F_GREEDY_DRAFT};
use tokio::net::{TcpListener, TcpStream};

#[derive(clap::ValueEnum, Clone, PartialEq)]
enum Mode {
    Single,
    Coordinator,
    Worker,
}

#[derive(clap::ValueEnum, Clone, Copy, PartialEq)]
enum ReturnMode {
    Naive,
    Lazy,
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

    #[arg(long, default_value_t = 0)]
    spec_seed: u64,

    #[arg(long, default_value_t = 0.0)]
    draft_temp: f32,

    #[arg(long, value_enum, default_value = "lazy")]
    return_mode: ReturnMode,
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
        Mode::Coordinator if args.spec_k > 0 => run_spec_coordinator(&args).await,
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
    let temp = args.temperature;
    let mut cur = if temp > 0.0 {
        target.sample_seeded(&target.logits(&ht).await?, temp, args.spec_seed).await?
    } else {
        *target.argmax(&ht).await?.last().ok_or("empty prefill argmax")?
    };

    let draft_temp = args.draft_temp;
    let sampled_draft = temp > 0.0 && draft_temp > 0.0;

    let mut rounds: u64 = 0;
    let mut accepted_sum: u64 = 0;
    let mut draft_us: u128 = 0;
    let mut verify_us: u128 = 0;
    let mut verify_ret_bytes: u128 = 0;
    let mut token_count: u32 = 1;

    print!("{}", target.detokenize(&[cur]).await?);
    std::io::stdout().flush()?;
    let t_first = Instant::now();

    if cur != eos {
        'outer: while token_count < args.max_tokens {
            let seed = args.spec_seed.wrapping_add(rounds);

            let t_d = Instant::now();
            let x = if sampled_draft {
                draft.draft_sample(cur, k, draft_temp, seed).await?
            } else {
                draft.draft(cur, k).await?
            };
            draft_us += t_d.elapsed().as_micros();

            let t_v = Instant::now();
            let mut verify_ids = Vec::with_capacity(x.len() + 1);
            verify_ids.push(cur);
            verify_ids.extend_from_slice(&x);
            let h = target.forward(&target.embed(&verify_ids).await?, "decode").await?;

            let (a, final_tok): (usize, u32) = if temp == 0.0 {
                let t = target.argmax(&h).await?;
                let mut na = 0usize;
                while na < k as usize && t[na] == x[na] {
                    na += 1;
                }
                verify_ret_bytes += ((k as usize + 1) * 4) as u128;
                (na, t[na])
            } else if sampled_draft {
                let p = target.verify_probs(&h, temp).await?;
                verify_ret_bytes += p.data.len() as u128;
                let (na, ft) = draft.accept(&p, temp, seed).await?;
                (na as usize, ft)
            } else {
                let (na, ft) = target.verify_accept(&h, &x, temp, seed).await?;
                verify_ret_bytes += 8;
                (na as usize, ft)
            };
            verify_us += t_v.elapsed().as_micros();

            let mut emitted: Vec<u32> = x[..a].to_vec();
            emitted.push(final_tok);

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
    let gen_tokens = token_count.saturating_sub(1);
    let alpha = if rounds > 0 { accepted_sum as f64 / (rounds as f64 * k as f64) } else { 0.0 };
    let mean_acc = if rounds > 0 { gen_tokens as f64 / rounds as f64 } else { 0.0 };
    eprintln!("\n");
    eprintln!(
        "tributary (spec K={k} T={temp} draft_T={draft_temp}) | {token_count} tokens | {:.1} tok/s | ttft: {:.2}s",
        if elapsed > 0.0 { gen_tokens as f32 / elapsed } else { 0.0 },
        (t_first - t_start).as_secs_f32()
    );
    eprintln!(
        "spec | rounds={rounds} accept_rate α={alpha:.3} mean_accepted/verify={mean_acc:.2} | \
         draft={}µs verify={}µs verify_ret={}B (mean/round)",
        if rounds > 0 { draft_us / rounds as u128 } else { 0 },
        if rounds > 0 { verify_us / rounds as u128 } else { 0 },
        if rounds > 0 { verify_ret_bytes / rounds as u128 } else { 0 },
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

async fn validate_split(
    stream: &mut TcpStream,
    local: &MlxClient,
    seq: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let coord = local.info().await?;
    write_frame(stream, &Frame::control(MsgType::Info, seq)).await?;
    let winfo = read_frame(stream).await?;
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
    Ok(())
}

async fn run_spec_coordinator(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = args.prompt.as_ref().ok_or("coordinator requires --prompt")?;
    let worker_addr = args.worker.as_ref().ok_or("coordinator requires --worker")?;
    let draft_url = args.draft_model.as_ref().ok_or("--spec-k requires --draft-model")?;
    let k = args.spec_k;
    let temp = args.temperature;
    let draft_temp = args.draft_temp;
    let sampled = temp > 0.0;
    let sampled_draft = sampled && draft_temp > 0.0;
    let return_mode = args.return_mode;
    if return_mode == ReturnMode::Naive && sampled && !sampled_draft {
        return Err("naive return with temperature>0 requires --draft-temp>0 (coordinator-side accept needs the draft distribution); use --return-mode lazy for a greedy draft".into());
    }
    let local = MlxClient::new(args.mlx_server.clone());
    let draft = MlxClient::new(draft_url.clone());
    let mut stream: TcpStream = TcpStream::connect(worker_addr).await?;
    let mut seq: u32 = 0;

    validate_split(&mut stream, &local, seq).await?;
    seq += 1;

    local.reset().await?;
    draft.reset().await?;
    write_frame(&mut stream, &Frame::control(MsgType::ResetCache, seq)).await?;
    seq += 1;

    let (ids, eos_token_id) = local.tokenize(prompt).await?;
    let t_start = Instant::now();

    let hidden = local.forward(&local.embed(&ids).await?, "prefill").await?;
    let mut cur = {
        write_frame(&mut stream, &Frame::from_tensor(MsgType::Prefill, seq, &hidden)).await?;
        let logits = recv_logits(&mut stream, seq).await?.into_tensor();
        seq += 1;
        if sampled {
            local.sample_seeded(&logits, temp, args.spec_seed).await?
        } else {
            local.sample(&logits, 0.0).await?
        }
    };
    let _ = draft.forward(&draft.embed(&ids).await?, "prefill").await?;

    let mut rounds: u64 = 0;
    let mut accepted_sum: u64 = 0;
    let mut token_count: u32 = 1;
    let mut timings: Vec<StepTiming> = Vec::new();
    let mut return_bytes: u128 = 0;

    print!("{}", local.detokenize(&[cur]).await?);
    std::io::stdout().flush()?;
    let t_first = Instant::now();

    if cur != eos_token_id {
        'outer: while token_count < args.max_tokens {
            let seed = args.spec_seed.wrapping_add(rounds);

            let t_d = Instant::now();
            let x = if sampled_draft {
                draft.draft_sample(cur, k, draft_temp, seed).await?
            } else {
                draft.draft(cur, k).await?
            };
            let draft_us = t_d.elapsed().as_micros();

            let t_local = Instant::now();
            let mut verify_ids = Vec::with_capacity(x.len() + 1);
            verify_ids.push(cur);
            verify_ids.extend_from_slice(&x);
            let h = local.forward(&local.embed(&verify_ids).await?, "decode").await?;
            let local_us = t_local.elapsed().as_micros() + draft_us;

            let (a, final_tok, mut vt) = if return_mode == ReturnMode::Naive {
                let (probs, vt) = verify_exchange_full(&mut stream, &h, temp, seq).await?;
                seq += 1;
                return_bytes += probs.data.len() as u128;
                if sampled_draft {
                    let (a, final_tok) = draft.accept(&probs, temp, seed).await?;
                    (a as usize, final_tok, vt)
                } else {
                    let ids = argmax_rows(&probs, k as usize + 1)?;
                    let mut a = 0usize;
                    while a < k as usize && ids[a] == x[a] {
                        a += 1;
                    }
                    (a, ids[a], vt)
                }
            } else if sampled_draft {
                let mut aux_out = Vec::with_capacity(x.len() + 1);
                aux_out.push(temp.to_bits());
                aux_out.extend_from_slice(&x);
                let (bits, vt) = verify_exchange(&mut stream, &h, aux_out, 0, seq).await?;
                seq += 1;
                let px: Vec<f32> = bits.iter().map(|b| f32::from_bits(*b)).collect();
                return_bytes += (px.len() * 4) as u128;
                let (a, pos) = draft.accept_scalars(&px, seed).await?;
                let (p_row, worker2_us, bytes2) = logits_at_exchange(&mut stream, pos, seq).await?;
                seq += 1;
                return_bytes += bytes2 as u128;
                let final_tok = draft.resample_at(&p_row, pos).await?;
                let mut vt = vt;
                vt.worker_us += worker2_us;
                (a as usize, final_tok, vt)
            } else if sampled {
                let mut aux_out = Vec::with_capacity(x.len() + 3);
                aux_out.push(temp.to_bits());
                aux_out.push((seed >> 32) as u32);
                aux_out.push(seed as u32);
                aux_out.extend_from_slice(&x);
                let (res, vt) = verify_exchange(&mut stream, &h, aux_out, F_SAMPLED | F_GREEDY_DRAFT, seq).await?;
                seq += 1;
                return_bytes += (res.len() * 4) as u128;
                (res[0] as usize, res[1], vt)
            } else {
                let (t, vt) = verify_exchange(&mut stream, &h, Vec::new(), 0, seq).await?;
                seq += 1;
                return_bytes += (t.len() * 4) as u128;
                let mut a = 0usize;
                while a < k as usize && t[a] == x[a] {
                    a += 1;
                }
                (a, t[a], vt)
            };

            let mut emitted: Vec<u32> = x[..a].to_vec();
            emitted.push(final_tok);

            let trim = k - a as u32;
            local.trim(trim).await?;
            draft.trim(trim).await?;
            write_frame(&mut stream, &Frame::control_aux(MsgType::Trim, seq, vec![trim])).await?;
            seq += 1;

            vt.local_us = local_us;
            timings.push(vt);
            rounds += 1;
            accepted_sum += a as u64;

            for &tok in &emitted {
                print!("{}", local.detokenize(&[tok]).await?);
                std::io::stdout().flush()?;
                token_count += 1;
                cur = tok;
                if tok == eos_token_id || token_count >= args.max_tokens {
                    break 'outer;
                }
            }
        }
    }

    let elapsed = t_first.elapsed().as_secs_f32();
    let gen_tokens = token_count.saturating_sub(1);
    let alpha = if rounds > 0 { accepted_sum as f64 / (rounds as f64 * k as f64) } else { 0.0 };
    let mean_acc = if rounds > 0 { gen_tokens as f64 / rounds as f64 } else { 0.0 };
    eprintln!("\n");
    eprintln!(
        "tributary (spec-coordinator K={k} T={temp} draft_T={draft_temp}) | {token_count} tokens | {:.1} tok/s | ttft: {:.2}s",
        if elapsed > 0.0 { gen_tokens as f32 / elapsed } else { 0.0 },
        (t_first - t_start).as_secs_f32()
    );
    eprintln!(
        "spec | rounds={rounds} accept_rate α={alpha:.3} mean_accepted/verify={mean_acc:.2} | \
         verify_ret={}B/round round_trips/tok={:.3}",
        if rounds > 0 { return_bytes / rounds as u128 } else { 0 },
        if gen_tokens > 0 { rounds as f64 / gen_tokens as f64 } else { 0.0 },
    );
    if !timings.is_empty() {
        let prefill = StepTiming::default();
        print_timing_summary(&prefill, &timings);
    }
    if let Some(path) = &args.timing_csv {
        write_timing_csv(path, &StepTiming::default(), &timings)?;
        eprintln!("wrote timing CSV to {path}");
    }
    Ok(())
}

async fn run_coordinator(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = args.prompt.as_ref().ok_or("coordinator requires --prompt")?;
    let worker_addr = args.worker.as_ref().ok_or("coordinator requires --worker")?;
    let local = MlxClient::new(args.mlx_server.clone());
    let mut stream: TcpStream = TcpStream::connect(worker_addr).await?;
    let mut seq: u32 = 0;

    validate_split(&mut stream, &local, seq).await?;
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

    let (mut next_id, mut prefill_timing) = exchange(&mut stream, &local, &hidden, MsgType::Prefill, seq, args.temperature, args.return_mode, args.spec_seed).await?;
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

        let (nid, mut st) = exchange(&mut stream, &local, &h, MsgType::DecodeStep, seq, args.temperature, args.return_mode, args.spec_seed.wrapping_add(1 + token_count as u64)).await?;
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
    return_mode: ReturnMode,
    seed: u64,
) -> Result<(u32, StepTiming), Box<dyn std::error::Error>> {
    let mut t = StepTiming::default();

    let t_ser = Instant::now();
    let mut frame_out = Frame::from_tensor(msg_type, seq, hidden);
    if return_mode == ReturnMode::Lazy {
        frame_out.flags = F_LAZY_SAMPLE;
        if temperature > 0.0 {
            frame_out.flags |= F_SAMPLED;
            frame_out.aux = vec![temperature.to_bits(), (seed >> 32) as u32, seed as u32];
        }
    }
    t.serialize_us = t_ser.elapsed().as_micros();
    t.activation_bytes = frame_out.payload.len();

    let t_rt = Instant::now();
    write_frame(stream, &frame_out).await?;

    let next_id = if return_mode == ReturnMode::Lazy {
        let reply = read_frame(stream).await?;
        t.roundtrip_us = t_rt.elapsed().as_micros();
        if reply.msg_type != MsgType::VerifyResult {
            return Err(format!("expected VerifyResult (lazy return), got {:?}", reply.msg_type).into());
        }
        if reply.seq != seq {
            return Err(format!("seq mismatch: sent {seq}, got {}", reply.seq).into());
        }
        t.worker_us = reply.worker_compute_us as u128;
        t.network_us = t.roundtrip_us.saturating_sub(t.worker_us);
        t.logits_bytes = reply.aux.len() * 4;
        *reply.aux.first().ok_or("lazy return: empty aux")?
    } else {
        let reply = recv_logits(stream, seq).await?;
        t.roundtrip_us = t_rt.elapsed().as_micros();
        t.worker_us = reply.worker_compute_us as u128;
        t.network_us = t.roundtrip_us.saturating_sub(t.worker_us);
        t.logits_bytes = reply.payload.len();

        let t_de = Instant::now();
        let logits = reply.into_tensor();
        t.deserialize_us = t_de.elapsed().as_micros();

        let t_s = Instant::now();
        let id = local.sample(&logits, temperature).await?;
        t.sample_us = t_s.elapsed().as_micros();
        id
    };

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

async fn verify_exchange(
    stream: &mut TcpStream,
    hidden: &Tensor,
    aux_out: Vec<u32>,
    flags: u8,
    seq: u32,
) -> Result<(Vec<u32>, StepTiming), Box<dyn std::error::Error>> {
    let mut t = StepTiming::default();

    let t_ser = Instant::now();
    let mut frame_out = Frame::from_tensor(MsgType::Verify, seq, hidden);
    frame_out.aux = aux_out;
    frame_out.flags = flags;
    t.serialize_us = t_ser.elapsed().as_micros();
    t.activation_bytes = frame_out.payload.len();

    let t_rt = Instant::now();
    write_frame(stream, &frame_out).await?;
    let reply = read_frame(stream).await?;
    t.roundtrip_us = t_rt.elapsed().as_micros();
    if reply.msg_type != MsgType::VerifyResult {
        return Err(format!("expected VerifyResult frame, got {:?}", reply.msg_type).into());
    }
    if reply.seq != seq {
        return Err(format!("seq mismatch: sent {seq}, got {}", reply.seq).into());
    }
    t.worker_us = reply.worker_compute_us as u128;
    t.network_us = t.roundtrip_us.saturating_sub(t.worker_us);
    t.logits_bytes = reply.aux.len() * 4;

    Ok((reply.aux, t))
}

async fn verify_exchange_full(
    stream: &mut TcpStream,
    hidden: &Tensor,
    temp: f32,
    seq: u32,
) -> Result<(Tensor, StepTiming), Box<dyn std::error::Error>> {
    let mut t = StepTiming::default();

    let t_ser = Instant::now();
    let mut frame_out = Frame::from_tensor(MsgType::Verify, seq, hidden);
    frame_out.flags = F_SPEC_NAIVE;
    frame_out.aux = vec![temp.to_bits()];
    t.serialize_us = t_ser.elapsed().as_micros();
    t.activation_bytes = frame_out.payload.len();

    let t_rt = Instant::now();
    write_frame(stream, &frame_out).await?;
    let reply = read_frame(stream).await?;
    t.roundtrip_us = t_rt.elapsed().as_micros();
    if reply.msg_type != MsgType::Logits {
        return Err(format!("expected Logits frame (naive verify), got {:?}", reply.msg_type).into());
    }
    if reply.seq != seq {
        return Err(format!("seq mismatch: sent {seq}, got {}", reply.seq).into());
    }
    t.worker_us = reply.worker_compute_us as u128;
    t.network_us = t.roundtrip_us.saturating_sub(t.worker_us);
    t.logits_bytes = reply.payload.len();
    Ok((reply.into_tensor(), t))
}

fn argmax_rows(t: &Tensor, rows: usize) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let dims: Vec<usize> = t.shape.split(',').map(|s| s.parse().unwrap_or(0)).collect();
    let cols = *dims.last().ok_or("argmax_rows: empty shape")?;
    if cols == 0 {
        return Err("argmax_rows: zero-width rows".into());
    }
    let data = &t.data;
    if data.len() < rows * cols * 4 {
        return Err(format!("argmax_rows: payload {}B too small for {rows}x{cols} f32", data.len()).into());
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let base = r * cols * 4;
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for c in 0..cols {
            let off = base + c * 4;
            let v = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            if v > best_v {
                best_v = v;
                best_i = c;
            }
        }
        out.push(best_i as u32);
    }
    Ok(out)
}

async fn logits_at_exchange(
    stream: &mut TcpStream,
    pos: u32,
    seq: u32,
) -> Result<(Tensor, u128, usize), Box<dyn std::error::Error>> {
    write_frame(stream, &Frame::control_aux(MsgType::LogitsAt, seq, vec![pos])).await?;
    let reply = read_frame(stream).await?;
    if reply.msg_type != MsgType::Logits {
        return Err(format!("expected Logits frame, got {:?}", reply.msg_type).into());
    }
    if reply.seq != seq {
        return Err(format!("seq mismatch: sent {seq}, got {}", reply.seq).into());
    }
    let worker_us = reply.worker_compute_us as u128;
    let bytes = reply.payload.len();
    Ok((reply.into_tensor(), worker_us, bytes))
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
                MsgType::Trim => {
                    let n = frame.aux.first().copied().unwrap_or(0);
                    local.trim(n).await?;
                }
                MsgType::Prefill | MsgType::DecodeStep => {
                    let seq = frame.seq;
                    let flags = frame.flags;
                    let aux = frame.aux.clone();
                    let mode = if frame.msg_type == MsgType::Prefill { "prefill" } else { "decode" };

                    let t_compute = Instant::now();
                    let hidden = local.forward(&frame.into_tensor(), mode).await?;

                    if flags & F_LAZY_SAMPLE != 0 {
                        let id = if flags & F_SAMPLED != 0 {
                            let temp = f32::from_bits(aux[0]);
                            let seed = ((aux[1] as u64) << 32) | aux[2] as u64;
                            local.sample_seeded(&local.logits(&hidden).await?, temp, seed).await?
                        } else {
                            *local.argmax(&hidden).await?.last().ok_or("empty argmax")?
                        };
                        let compute_us = t_compute.elapsed().as_micros() as u64;
                        let mut reply = Frame::control_aux(MsgType::VerifyResult, seq, vec![id]);
                        reply.worker_compute_us = compute_us;
                        write_frame(&mut stream, &reply).await?;
                    } else {
                        let logits = local.logits(&hidden).await?;
                        let compute_us = t_compute.elapsed().as_micros() as u64;
                        let mut reply = Frame::from_tensor(MsgType::Logits, seq, &logits);
                        reply.worker_compute_us = compute_us;
                        write_frame(&mut stream, &reply).await?;
                    }
                }
                MsgType::Verify => {
                    let seq = frame.seq;
                    let flags = frame.flags;
                    let aux = frame.aux.clone();
                    let t_compute = Instant::now();
                    let hidden = local.forward(&frame.into_tensor(), "decode").await?;

                    if flags & F_SPEC_NAIVE != 0 {
                        let temp = f32::from_bits(aux[0]);
                        let vp_temp = if temp > 0.0 { temp } else { 1.0 };
                        let probs = local.verify_probs(&hidden, vp_temp).await?;
                        let compute_us = t_compute.elapsed().as_micros() as u64;
                        let mut reply = Frame::from_tensor(MsgType::Logits, seq, &probs);
                        reply.worker_compute_us = compute_us;
                        write_frame(&mut stream, &reply).await?;
                    } else if flags & F_GREEDY_DRAFT != 0 {
                        let temp = f32::from_bits(aux[0]);
                        let seed = ((aux[1] as u64) << 32) | aux[2] as u64;
                        let x: Vec<u32> = aux[3..].to_vec();
                        let (a, final_tok) = local.verify_accept(&hidden, &x, temp, seed).await?;
                        let compute_us = t_compute.elapsed().as_micros() as u64;
                        let mut reply = Frame::control_aux(MsgType::VerifyResult, seq, vec![a, final_tok]);
                        reply.worker_compute_us = compute_us;
                        write_frame(&mut stream, &reply).await?;
                    } else {
                        let result_aux = if aux.is_empty() {
                            local.argmax(&hidden).await?
                        } else {
                            let temp = f32::from_bits(aux[0]);
                            let x: Vec<u32> = aux[1..].to_vec();
                            local.verify_scalars(&hidden, &x, temp).await?
                                .iter().map(|s| s.to_bits()).collect()
                        };
                        let compute_us = t_compute.elapsed().as_micros() as u64;
                        let mut reply = Frame::control_aux(MsgType::VerifyResult, seq, result_aux);
                        reply.worker_compute_us = compute_us;
                        write_frame(&mut stream, &reply).await?;
                    }
                }
                MsgType::LogitsAt => {
                    let seq = frame.seq;
                    let pos = frame.aux.first().copied().unwrap_or(0);
                    let t_compute = Instant::now();
                    let row = local.logits_at(pos).await?;
                    let compute_us = t_compute.elapsed().as_micros() as u64;

                    let mut reply = Frame::from_tensor(MsgType::Logits, seq, &row);
                    reply.worker_compute_us = compute_us;
                    write_frame(&mut stream, &reply).await?;
                }
                MsgType::Logits | MsgType::VerifyResult => {
                    return Err(format!("worker received an unexpected {:?} frame", frame.msg_type).into());
                }
            }
        }
        local.reset().await?;
        eprintln!("worker: ready for next connection");
    }
}