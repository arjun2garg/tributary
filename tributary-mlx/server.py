import time
import json
import asyncio
import mlx.core as mx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from model import PartialModel

app = FastAPI()

model = PartialModel(
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    start_layer=0,
    end_layer=28
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.0

async def token_stream(prompt: str, max_tokens: int, temperature: float):
    model.reset_cache()
    token_ids = model.tokenizer.encode(prompt)
    t_start = time.perf_counter()

    hidden = model.embed(token_ids)
    hidden = model.prefill(hidden)
    logits = model.decode_logits(hidden)
    next_id = model.sample_token(logits[:, -1:, :], temperature)
    token_count = 0
    t_first = time.perf_counter()

    for _ in range(max_tokens):
        token_str = model.tokenizer.decode([next_id])
        event = json.dumps({
            "token": token_str,
            "token_id": next_id,
            "token_count": token_count,
        })
        yield f"data: {event}\n\n"

        if next_id == model.tokenizer.eos_token_id:
            break

        hidden = model.embed([next_id])
        hidden = model.decode_step(hidden)
        logits = model.decode_logits(hidden)
        next_id = model.sample_token(logits, temperature)
        mx.eval(next_id)
        token_count += 1

        await asyncio.sleep(0)
    
    elapsed = time.perf_counter() - t_first
    summary = json.dumps({
        "tokens": token_count,
        "tokens_per_sec": token_count / elapsed if elapsed > 0 else 0,
        "time_to_first_token": t_first - t_start,
    })
    yield f"data: {summary}\n\n"

@app.post("/generate")
async def generate(req: GenerateRequest):
    return StreamingResponse(
        token_stream(req.prompt, req.max_tokens, req.temperature),
        media_type="text/event-stream"
    )
