import argparse
import uvicorn
import numpy as np
import mlx.core as mx
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from model import PartialModel

app = FastAPI()
model: PartialModel | None = None

def tensor_from_request(body: bytes, shape_header: str, dtype_header: str) -> mx.array:
    if dtype_header != "float16":
        raise HTTPException(400, f"unsupported dtype: {dtype_header}")
    shape = tuple(int(d) for d in shape_header.split(","))
    return mx.array(np.frombuffer(body, dtype=np.float16).reshape(shape))

def tensor_response(x: mx.array) -> Response:
    arr = np.array(x.astype(mx.float16))
    return Response(
        content=arr.tobytes(),
        media_type="application/octet-stream",
        headers={"X-Shape": ",".join(str(d) for d in arr.shape), "X-Dtype": "float16"},
    )

class DetokenizeRequest(BaseModel):
    token_ids: list[int]

@app.post("/detokenize")
async def detokenize(req: DetokenizeRequest):
    return {"text": model.tokenizer.decode(req.token_ids)}

class EmbedRequest(BaseModel):
    token_ids: list[int]

@app.post("/embed")
async def embed(req: EmbedRequest):
    if not model.is_first:
        raise HTTPException(400, f"/embed requires layer 0; this instance starts at {model.start_layer}")
    return tensor_response(model.embed(req.token_ids))

@app.post("/forward")
async def forward(request: Request, mode: str):
    x = tensor_from_request(await request.body(), request.headers.get("x-shape"), request.headers.get("x-dtype"))
    if mode == "prefill":
        out = model.prefill(x)
    elif mode == "decode":
        out = model.decode_step(x)
    else:
        raise HTTPException(400, f"unknown mode: {mode}")
    return tensor_response(out)

@app.post("/logits")
async def logits(request: Request, last_only: bool = False):
    if not model.is_last:
        raise HTTPException(400, f"/logits requires the final layer ({model.num_layers}); this instance ends at {model.end_layer}")
    x = tensor_from_request(await request.body(), request.headers.get("x-shape"), request.headers.get("x-dtype"))
    if last_only:
        x = x[:, -1:, :]
    return tensor_response(model.decode_logits(x))

@app.post("/reset")
async def reset():
    model.reset_cache()
    return {"ok": True}

@app.post("/sample")
async def sample(request: Request, temperature: float = 0.0):
    logits = tensor_from_request(await request.body(), request.headers.get("x-shape"), request.headers.get("x-dtype"))
    return {"token_id": model.sample_token(logits, temperature)}

class TokenizeRequest(BaseModel):
    text: str

@app.post("/tokenize")
async def tokenize(req: TokenizeRequest):
    return {
        "token_ids": model.tokenizer.encode(req.text),
        "eos_token_id": model.tokenizer.eos_token_id,
    }

@app.get("/info")
async def info():
    return {
        "start_layer": model.start_layer,
        "end_layer": model.end_layer,
        "num_layers": model.num_layers,
        "is_first": model.is_first,
        "is_last": model.is_last,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=None)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    model = PartialModel(args.model, start_layer=args.start_layer, end_layer=args.end_layer)
    print(
        f"tributary-mlx | {args.model} | layers {model.start_layer}..{model.end_layer} "
        f"of {model.num_layers} | embeds: {model.is_first} | logits: {model.is_last} | "
        f"port {args.port}"
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port)