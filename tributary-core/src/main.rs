use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(name = "tributary")]
#[command(about = "Distributed LLM inference across Apple Silicon devices")]
struct Args {
    #[arg(long)]
    prompt: String,
    
    #[arg(long, default_value = "100")]
    max_tokens: u32,
    
    #[arg(long, default_value = "http://localhost:8765")]
    mlx_server: String,
}

#[derive(Serialize)]
struct GenerateRequest {
    prompt: String,
    max_tokens: u32,
}

#[derive(Deserialize, Debug)]
struct GenerateResponse {
    text: String,
    tokens: u32,
    tokens_per_sec: f32,
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
        .send()
        .await?
        .json::<GenerateResponse>()
        .await?;

    println!("{}", response.text);
    println!("\n---");
    println!("tributary | {} tokens at {:.1} tok/s", response.tokens, response.tokens_per_sec);
    
    Ok(())
}