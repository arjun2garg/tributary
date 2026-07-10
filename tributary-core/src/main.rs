use clap::Parser;
use serde::{Deserialize, Serialize};
use futures_util::StreamExt;
use std::io::Write;

#[derive(Parser)]
#[command(name = "tributary")]
#[command(about = "Distributed LLM inference across Apple Silicon devices")]
struct Args {
    #[arg(long)]
    prompt: String,
    
    #[arg(long, default_value = "200")]
    max_tokens: u32,
    
    #[arg(long, default_value = "http://localhost:8765")]
    mlx_server: String,
}

#[derive(Serialize)]
struct GenerateRequest {
    prompt: String,
    max_tokens: u32,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SseEvent {
    Token {
        token: String,
        token_id: u32,
        token_count: u32
    },
    Summary {
        tokens: u32,
        tokens_per_sec: f32,
        time_to_first_token: f32
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("tributary | connecting to MLX server at {}", args.mlx_server);

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/generate", args.mlx_server))
        .json(&GenerateRequest {
            prompt: args.prompt.clone(),
            max_tokens: args.max_tokens,
        })
        .send().await?;

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(pos) = buffer.find("\n\n") {
            let event_str = buffer[..pos].to_string();
            buffer = buffer[pos + 2..].to_string();
            if let Some(json_str) = event_str.strip_prefix("data: ") {
                if let Ok(event) = serde_json::from_str::<SseEvent>(json_str) {
                    match event {
                        SseEvent::Token { token, .. } => {
                            print!("{}", token);
                            std::io::stdout().flush()?;
                        }
                        SseEvent::Summary { tokens, tokens_per_sec, time_to_first_token, .. } => {
                            eprintln!("\n");
                            eprintln!(
                                "specter | {} tokens | {:.1} tok/s | ttft: {:.2}s",
                                tokens,
                                tokens_per_sec,
                                time_to_first_token
                            );
                        }
                    }
                }
            }
        }
    }
    
    Ok(())
}