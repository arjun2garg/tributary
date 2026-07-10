import time

from fastapi import FastAPI
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
    max_tokens: int = 100

class TokenResponse(BaseModel):
    token: str
    token_id: int
    tokens_per_sec: float

@app.post("/generate")
async def generate(req: GenerateRequest):
    tokens = []
    start = time.time()
    hidden = model.embed(req.prompt)
    for i in range(req.max_tokens):
        hidden = model.forward_partial(hidden)
        next_id = model.decode(hidden)
        if next_id == model.tokenizer.eos_token_id:
            break

        token_str = model.tokenizer.decode([next_id])
        tokens.append(token_str)

        all_text = req.prompt + "".join(tokens)
        hidden = model.embed(all_text) # very inefficient since no KV cache yet
    
    elapsed = time.time() - start
    return {
        "text": "".join(tokens),
        "tokens": len(tokens),
        "tokens_per_sec": len(tokens) / elapsed
    }

