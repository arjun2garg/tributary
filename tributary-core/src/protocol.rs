use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use crate::mlx_client::Tensor;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const MAX_PAYLOAD: usize = 64 * 1024 * 1024;

pub const F_LAZY_SAMPLE: u8 = 1 << 0;
pub const F_SPEC_NAIVE: u8 = 1 << 1;
pub const F_SAMPLED: u8 = 1 << 2;
pub const F_GREEDY_DRAFT: u8 = 1 << 3;

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
pub enum MsgType {
    Prefill = 0,
    DecodeStep = 1,
    Logits = 2,
    ResetCache = 3,
    Info = 4,
    Trim = 5,
    Verify = 6,
    VerifyResult = 7,
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
    pub aux: Vec<u32>,
    pub dtype: u8,
    pub flags: u8,
    pub payload: Bytes,
}

impl Frame {
    pub fn control(msg_type: MsgType, seq: u32) -> Self {
        Frame { msg_type, seq, worker_compute_us: 0, shape: Vec::new(), aux: Vec::new(), dtype: 0, flags: 0, payload: Bytes::new() }
    }

    pub fn control_aux(msg_type: MsgType, seq: u32, aux: Vec<u32>) -> Self {
        Frame { msg_type, seq, worker_compute_us: 0, shape: Vec::new(), aux, dtype: 0, flags: 0, payload: Bytes::new() }
    }

    pub fn from_tensor(msg_type: MsgType, seq: u32, t: &Tensor) -> Self {
        let shape = t.shape.split(',').map(|s| s.parse().unwrap()).collect();
        Frame { msg_type, seq, worker_compute_us: 0, shape, aux: Vec::new(), dtype: 0, flags: 0, payload: t.data.clone() }
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
    header.push(f.flags);
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

    r.read_exact(&mut one).await?;
    let flags = one[0];

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

    Ok(Frame { msg_type, seq, worker_compute_us, shape, aux, dtype, flags, payload: Bytes::from(payload) })
}

