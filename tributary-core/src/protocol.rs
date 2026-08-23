use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use crate::mlx_client::Tensor;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const MAX_PAYLOAD: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
pub enum MsgType {
    Prefill = 0,
    DecodeStep = 1,
    Logits = 2,
    ResetCache = 3,
    Info = 4,
    Trim = 5,
    // Speculative-decoding verify: coordinator ships K+1 activations, worker runs its
    // shard over all positions and replies with a VerifyResult.
    Verify = 6,
    // Reply to Verify. For greedy: `aux` = worker's argmax token id at each of the K+1
    // positions. For sampled (temp>0): `payload` = K+1 fp32 target probs p_i(x_i).
    VerifyResult = 7,
    // Phase-2 lazy-logits request: `aux[0]` = position j; worker replies with a Logits
    // frame carrying only that one full distribution.
    LogitsAt = 8,
}

impl TryFrom<u8> for MsgType {
    type Error = Box<dyn std::error::Error>;
    fn try_from(v: u8) -> Result<Self> {
        match v {
            0 => Ok(MsgType::Prefill),
            1 => Ok(MsgType::DecodeStep),
            2 => Ok(MsgType::Logits),
            3 => Ok(MsgType::ResetCache),
            4 => Ok(MsgType::Info),
            5 => Ok(MsgType::Trim),
            6 => Ok(MsgType::Verify),
            7 => Ok(MsgType::VerifyResult),
            8 => Ok(MsgType::LogitsAt),
            _ => Err(format!("unknown msg_type byte: {v}").into()),
        }
    }
}

pub struct Frame {
    pub msg_type: MsgType,
    pub seq: u32,
    pub worker_compute_us: u64,
    pub shape: Vec<u32>,
    // Small control side-channel of u32s: trim count (Trim), position (LogitsAt),
    // drafted token ids (Verify), or result token ids (VerifyResult).
    pub aux: Vec<u32>,
    pub dtype: u8,
    pub payload: Bytes,
}

impl Frame {
    pub fn control(msg_type: MsgType, seq: u32) -> Self {
        Frame { msg_type, seq, worker_compute_us: 0, shape: Vec::new(), aux: Vec::new(), dtype: 0, payload: Bytes::new() }
    }

    pub fn control_aux(msg_type: MsgType, seq: u32, aux: Vec<u32>) -> Self {
        Frame { msg_type, seq, worker_compute_us: 0, shape: Vec::new(), aux, dtype: 0, payload: Bytes::new() }
    }

    pub fn from_tensor(msg_type: MsgType, seq: u32, t: &Tensor) -> Self {
        let shape = t.shape.split(',').map(|s| s.parse().unwrap()).collect();
        Frame { msg_type, seq, worker_compute_us: 0, shape, aux: Vec::new(), dtype: 0, payload: t.data.clone() }
    }

    pub fn into_tensor(self) -> Tensor {
        let shape = self.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
        Tensor { data: self.payload, shape }
    }
}

pub async fn write_frame<W>(w: &mut W, f: &Frame) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let mut header: Vec<u8> = Vec::new();
    header.push(f.msg_type as u8);
    header.extend_from_slice(&f.seq.to_be_bytes());
    header.extend_from_slice(&f.worker_compute_us.to_be_bytes());
    header.push(f.shape.len() as u8); // ndim
    for &d in &f.shape {
        header.extend_from_slice(&d.to_be_bytes());
    }
    header.push(f.aux.len() as u8); // naux
    for &a in &f.aux {
        header.extend_from_slice(&a.to_be_bytes());
    }
    header.push(f.dtype);
    header.extend_from_slice(&(f.payload.len() as u64).to_be_bytes());

    w.write_all(&header).await?;
    w.write_all(&f.payload).await?;
    w.flush().await?;
    Ok(())
}

pub async fn read_frame<R>(r: &mut R) -> Result<Frame>
where
    R: AsyncReadExt + Unpin,
{
    let mut one = [0u8; 1];
    r.read_exact(&mut one).await?;
    let msg_type = MsgType::try_from(one[0])?;

    let mut seq_buf = [0u8; 4];
    r.read_exact(&mut seq_buf).await?;
    let seq = u32::from_be_bytes(seq_buf);

    let mut wc_buf = [0u8; 8];
    r.read_exact(&mut wc_buf).await?;
    let worker_compute_us = u64::from_be_bytes(wc_buf);

    r.read_exact(&mut one).await?;
    let ndim = one[0] as usize;

    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let mut d = [0u8; 4];
        r.read_exact(&mut d).await?;
        shape.push(u32::from_be_bytes(d));
    }

    r.read_exact(&mut one).await?;
    let naux = one[0] as usize;
    let mut aux = Vec::with_capacity(naux);
    for _ in 0..naux {
        let mut a = [0u8; 4];
        r.read_exact(&mut a).await?;
        aux.push(u32::from_be_bytes(a));
    }

    r.read_exact(&mut one).await?;
    let dtype = one[0];

    let mut len_buf = [0u8; 8];
    r.read_exact(&mut len_buf).await?;
    let payload_len = u64::from_be_bytes(len_buf) as usize;
    if payload_len > MAX_PAYLOAD {
        return Err(format!(
            "frame payload_len {payload_len} exceeds max {MAX_PAYLOAD}; stream likely desynced"
        ).into());
    }

    let mut payload = vec![0u8; payload_len];
    r.read_exact(&mut payload).await?;

    Ok(Frame { msg_type, seq, worker_compute_us, shape, aux, dtype, payload: Bytes::from(payload) })
}

