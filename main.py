import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from difflib import SequenceMatcher

# ================= Environment Setup =====================
load_dotenv()

# ================= API Clients ===========================
client = None
try:
    if os.getenv("GROQ_API_KEY"):
        client = Groq()
        print("Groq client initialized.")
    else:
        print("⚠️ Warning: GROQ_API_KEY not found.")
except Exception as e:
    print(f"❌ Error initializing Groq client: {e}")

try:
    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        print("Gemini API initialized.")
    else:
        print("⚠️ GEMINI_API_KEY not found.")
except Exception as e:
    print(f"❌ Gemini init failed: {e}")


def load_organizations(file_path: str = "organizations.json") -> List[Dict[str, str]]:
    """Load biotech organizations list."""
    try:
        if not os.path.exists(file_path):
            print(f"{file_path} not found — returning empty list.")
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("organizations.json must be a list of dicts.")
            return data
    except Exception as e:
        print(f"❌ Failed to load organizations.json: {e}")
        return []

ORGANIZATIONS = load_organizations()
TARGET_NAMES = [org["name"] for org in ORGANIZATIONS]


GROQ_MODELS = ["llama-3.3-70b-versatile"]


class QueryRequest(BaseModel):
    prompt: str


def merge_model_insights(groq_text: str, gemini_text: str) -> str:
    """
    Merge and clean Groq + Gemini outputs into a unified, de-duplicated summary.
    Produces a polished, human-like biotech insight.
    """
    if not groq_text and not gemini_text:
        return "No insights were generated for this query."

    groq_text = groq_text.strip().replace("\n", " ")
    gemini_text = gemini_text.strip().replace("\n", " ")

    def split_sentences(text):
        return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

    groq_sents = split_sentences(groq_text)
    gemini_sents = split_sentences(gemini_text)

    merged_sents = []
    for sent in groq_sents + gemini_sents:
        if not any(SequenceMatcher(None, sent.lower(), s.lower()).ratio() > 0.8 for s in merged_sents):
            merged_sents.append(sent)

    merged_text = " ".join(merged_sents)
    if groq_text and gemini_text:
        merged_text += " Together, these perspectives suggest a strong convergence between AI-driven insights and real-world biotech research trends."

    merged_text = re.sub(r"\s+", " ", merged_text).strip()
    merged_text = merged_text[0].upper() + merged_text[1:] if merged_text else merged_text

    return merged_text

def check_ai_citation(prompt: str, model: str) -> Dict[str, Any]:
    """Analyze biotech org mentions using Groq."""
    if not client:
        raise HTTPException(status_code=500, detail="Groq client not initialized.")

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise, factual biotech and AI industry expert. "
                        "Identify specific organizations, research institutes, biotech startups, "
                        "or academic journals relevant to the user's prompt. "
                        "Mention real, verifiable entities such as Moderna, BioNTech, CRISPR Therapeutics, "
                        "DeepMind, or Nature Biotechnology."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        ai_response = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API failed: {str(e)}")

    citation_results = {}
    total_citations = 0

    for org in ORGANIZATIONS:
        name = org["name"]
        category = org.get("category", "Unknown")
        pattern = re.compile(rf"\b{name.replace('&', 'and')}\b", re.IGNORECASE)
        found = bool(pattern.search(ai_response))
        confidence = 100 if found else 0

        citation_results[name] = {
            "cited": found,
            "category": category,
            "confidence": confidence,
        }
        if found:
            total_citations += 1

    total_targets = len(ORGANIZATIONS)
    visibility_score = int((total_citations / total_targets) * 100) if total_targets else 0

    return {
        "prompt_tested": prompt,
        "ai_response": ai_response,
        "total_mentions": total_citations,
        "citation_breakdown": citation_results,
        "visibility_score": visibility_score,
    }

def check_ai_citation_gemini(prompt: str) -> str:
    """Run Gemini analysis and return only the text response."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"""
            You are a biotech visibility and citation intelligence expert.
            Identify specific organizations, research institutes, biotech startups,
            or journals related to this topic. Be concise and factual.
            Query: {prompt}
            """
        )
        return response.text.strip() if response and response.text else ""
    except Exception as e:
        print(f"❌ Gemini API failed: {e}")
        return ""


app = FastAPI(
    title="BioNexa Backend API",
    description="Backend API for BioNexa — AI-driven biotech visibility analysis.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def root():
    return {"status": "ok", "message": "BioNexa API is live!"}

@app.get("/organizations")
def get_organizations():
    return {"count": len(ORGANIZATIONS), "data": ORGANIZATIONS}

@app.post("/analyze_citation/")
async def analyze_citation(request: QueryRequest):
    """Run both Groq and Gemini models and return merged, de-duplicated insights."""
    try:
    
        groq_result = check_ai_citation(request.prompt, GROQ_MODELS[0])
        groq_text = groq_result["ai_response"]

     
        gemini_text = check_ai_citation_gemini(request.prompt)

     
        merged_text = merge_model_insights(groq_text, gemini_text)

     
        groq_result["ai_response"] = merged_text 

        return {"results": [groq_result]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
