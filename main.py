import os
from typing import Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

app = FastAPI(title="Multi-AI Prompt Output Viewer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    openai_api_key: Optional[str] = None
    claude_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None


class GenerateResponse(BaseModel):
    openai: Optional[str] = None
    claude: Optional[str] = None
    llama: Optional[str] = None
    meta: Dict[str, Any] = {}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    prompt = req.prompt.strip()

    results: Dict[str, Optional[str]] = {"openai": None, "claude": None, "llama": None}
    meta: Dict[str, Any] = {"used": {}}

    # OpenAI (if key provided)
    if req.openai_api_key:
        try:
            results["openai"] = call_openai(prompt, req.openai_api_key)
            meta["used"]["openai"] = True
        except Exception as e:
            results["openai"] = f"OpenAI error: {str(e)}"
            meta["used"]["openai"] = False
    else:
        results["openai"] = "No OpenAI API key provided. Showing placeholder response.\n\nThis is where OpenAI's answer would appear."
        meta["used"]["openai"] = False

    # Claude (Anthropic) (if key provided)
    if req.claude_api_key:
        try:
            results["claude"] = call_claude(prompt, req.claude_api_key)
            meta["used"]["claude"] = True
        except Exception as e:
            results["claude"] = f"Claude error: {str(e)}"
            meta["used"]["claude"] = False
    else:
        results["claude"] = "No Claude API key provided. Showing placeholder response.\n\nThis is where Claude's answer would appear."
        meta["used"]["claude"] = False

    # Llama via Groq (always attempt if key provided, else placeholder)
    groq_key = req.groq_api_key or os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            results["llama"] = call_groq_llama(prompt, groq_key)
            meta["used"]["llama_groq"] = True
        except Exception as e:
            results["llama"] = f"Llama (Groq) error: {str(e)}"
            meta["used"]["llama_groq"] = False
    else:
        results["llama"] = (
            "No Groq API key found. Showing a dummy Llama response.\n\n"
            "Example: As an open-source Llama model, I'd outline key points, propose a simple plan, "
            "and suggest next steps."
        )
        meta["used"]["llama_groq"] = False

    return GenerateResponse(**results, meta=meta)


# --- Service helpers ---

def call_openai(prompt: str, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def call_claude(prompt: str, api_key: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Claude HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    # New Messages API returns content list with text
    parts = data.get("content", [])
    if parts and isinstance(parts, list):
        # find first text part
        for p in parts:
            if p.get("type") == "text" and "text" in p:
                return p["text"].strip()
    # fallback
    return str(data)[:1000]


def call_groq_llama(prompt: str, api_key: str) -> str:
    # Groq provides an OpenAI-compatible endpoint
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",  # common free model; if unavailable, try llama3-8b-8192
        "messages": [
            {"role": "system", "content": "You are an open-source Llama model. Be clear and structured."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        # try alternate model name
        alt_payload = payload.copy()
        alt_payload["model"] = "llama3-8b-8192"
        r2 = requests.post(url, headers=headers, json=alt_payload, timeout=60)
        if r2.status_code >= 400:
            raise RuntimeError(f"Groq HTTP {r2.status_code}: {r2.text[:200]}")
        data = r2.json()
        return data["choices"][0]["message"]["content"].strip()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
