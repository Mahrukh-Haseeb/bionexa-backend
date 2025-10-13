from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import google.generativeai as genai


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY environment variable.")


client = Groq(api_key=GROQ_API_KEY)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


with open("organizations.json", "r", encoding="utf-8") as f:
    ORGANIZATIONS = json.load(f)

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
]

app = FastAPI(title="BioNexa GEO Analyzer", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    prompt: str


def check_ai_citation(prompt: str, model: str) -> Dict[str, Any]:
    """
    Sends a prompt to the specified Groq LLM and checks which biotech
    organizations are mentioned in the AI's response.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        ai_response = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{model} API request failed: {str(e)}")

    citation_results = {}
    total_citations = 0

    for org in ORGANIZATIONS:
        name = org["name"]
        category = org.get("category", "Unknown")
        found = re.search(rf"\b{name.replace('&', 'and')}\b", ai_response, re.IGNORECASE)
        confidence = 100 if found else 0
        citation_results[name] = {"cited": bool(found), "category": category, "confidence": confidence}
        if found:
            total_citations += 1

    total_targets = len(ORGANIZATIONS)
    visibility_score = round((total_citations / max(total_targets, 1)) * 100, 2)

    return {
        "llm_model": model,
        "ai_response": ai_response,
        "total_mentions": total_citations,
        "visibility_score": visibility_score,
        "citation_breakdown": citation_results,
    }

def check_gemini_citation(prompt: str) -> Dict[str, Any]:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        ai_response = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API failed: {str(e)}")

    citation_results = {}
    total_citations = 0
    for org in ORGANIZATIONS:
        name = org["name"]
        category = org.get("category", "Unknown")
        found = re.search(rf"\b{name.replace('&', 'and')}\b", ai_response, re.IGNORECASE)
        confidence = 100 if found else 0
        citation_results[name] = {"cited": bool(found), "category": category, "confidence": confidence}
        if found:
            total_citations += 1

    visibility_score = round((total_citations / max(len(ORGANIZATIONS), 1)) * 100, 2)
    return {
        "llm_model": "gemini-1.5-flash",
        "ai_response": ai_response,
        "total_mentions": total_citations,
        "visibility_score": visibility_score,
        "citation_breakdown": citation_results,
    }


@app.post("/analyze_citation/")
async def analyze_citation(request: QueryRequest):
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    results = []


    for model in GROQ_MODELS:
        result = check_ai_citation(prompt, model)
        results.append(result)


    if GEMINI_API_KEY:
        try:
            results.append(check_gemini_citation(prompt))
        except Exception as e:
            print(f"Gemini analysis failed: {e}")

    return {"results": results}


@app.get("/")
async def root():
    return {"status": "ok", "message": "BioNexa GEO backend is running."}

