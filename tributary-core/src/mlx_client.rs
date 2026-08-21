use bytes::Bytes;
use serde::Deserialize;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct Tensor {
    pub data: Bytes,
    pub shape: String,
}

pub struct MlxClient {
    base: String,
    http: reqwest::Client,
}

#[derive(Deserialize)]
struct TokenizeResponse {
    token_ids: Vec<u32>,
    eos_token_id: u32,
}

#[derive(Deserialize)]
struct DetokenizeResponse {
    text: String,
}

#[derive(Deserialize)]
struct SampleResponse {
    token_id: u32,
}

#[derive(Deserialize)]
struct TokenIdsResponse {
    token_ids: Vec<u32>,
}

#[derive(Deserialize)]
pub struct Info {
    pub start_layer: u32,
    pub end_layer: u32,
    pub num_layers: u32,
    pub is_first: bool,
}

impl MlxClient {
    pub fn new(base: String) -> Self {
        Self { base, http: reqwest::Client::new() }
    }
    
    pub async fn info(&self) -> Result<Info> {
        let resp: Info = self.http
            .get(format!("{}/info", self.base))
            .send().await?
            .error_for_status()?
            .json().await?;
        Ok(resp)
    }

    pub async fn reset(&self) -> Result<()> {
        self.http
            .post(format!("{}/reset", self.base))
            .send().await?
            .error_for_status()?;
        Ok(())
    }

    pub async fn tokenize(&self, text: &str) -> Result<(Vec<u32>, u32)> {
        let resp: TokenizeResponse = self.http
            .post(format!("{}/tokenize", self.base))
            .json(&serde_json::json!({ "text": text }))
            .send().await?
            .error_for_status()?
            .json().await?;
        Ok((resp.token_ids, resp.eos_token_id))
    }

    pub async fn detokenize(&self, token_ids: &[u32]) -> Result<String> {
        let resp: DetokenizeResponse = self.http
            .post(format!("{}/detokenize", self.base))
            .json(&serde_json::json!({ "token_ids": token_ids }))
            .send().await?
            .error_for_status()?
            .json().await?;
        Ok(resp.text)
    }

    pub async fn embed(&self, token_ids: &[u32]) -> Result<Tensor> {
        let resp = self.http
            .post(format!("{}/embed", self.base))
            .json(&serde_json::json!({ "token_ids": token_ids }))
            .send().await?
            .error_for_status()?;
        Self::tensor_from_response(resp).await
    }

    pub async fn forward(&self, x: &Tensor, mode: &str) -> Result<Tensor> {
        let resp = self.http
            .post(format!("{}/forward", self.base))
            .query(&[("mode", mode)])
            .header("X-Shape", &x.shape)
            .header("X-Dtype", "float16")
            .body(x.data.clone())
            .send().await?
            .error_for_status()?;
        Self::tensor_from_response(resp).await
    }

    pub async fn logits(&self, x: &Tensor) -> Result<Tensor> {
        let resp = self.http
            .post(format!("{}/logits", self.base))
            .query(&[("last_only", "true")])
            .header("X-Shape", &x.shape)
            .header("X-Dtype", "float16")
            .body(x.data.clone())
            .send().await?
            .error_for_status()?;
        Self::tensor_from_response(resp).await
    }

    pub async fn sample(&self, logits: &Tensor, temperature: f32) -> Result<u32> {
        let resp: SampleResponse = self.http
            .post(format!("{}/sample", self.base))
            .query(&[("temperature", temperature)])
            .header("X-Shape", &logits.shape)
            .header("X-Dtype", "float16")
            .body(logits.data.clone())
            .send().await?
            .error_for_status()?
            .json().await?;
        Ok(resp.token_id)
    }

    pub async fn draft(&self, cur: u32, k: u32) -> Result<Vec<u32>> {
        let resp: TokenIdsResponse = self.http
            .post(format!("{}/draft", self.base))
            .query(&[("cur", cur), ("k", k)])
            .send().await?
            .error_for_status()?
            .json().await?;
        Ok(resp.token_ids)
    }

    pub async fn argmax(&self, x: &Tensor) -> Result<Vec<u32>> {
        let resp: TokenIdsResponse = self.http
            .post(format!("{}/argmax", self.base))
            .header("X-Shape", &x.shape)
            .header("X-Dtype", "float16")
            .body(x.data.clone())
            .send().await?
            .error_for_status()?
            .json().await?;
        Ok(resp.token_ids)
    }

    pub async fn trim(&self, n: u32) -> Result<()> {
        self.http
            .post(format!("{}/trim", self.base))
            .query(&[("n", n)])
            .send().await?
            .error_for_status()?;
        Ok(())
    }

    async fn tensor_from_response(resp: reqwest::Response) -> Result<Tensor> {
        let shape = resp.headers()
            .get("x-shape")
            .ok_or("response missing X-Shape header")?
            .to_str()?
            .to_string();
        Ok(Tensor { data: resp.bytes().await?, shape })
    }
}