# main.py
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_DIR = BASE_DIR / "models"
PRIORITY_MODEL_PATH = MODEL_DIR / "priority_pipeline.pkl"
SENTIMENT_MODEL_PATH = MODEL_DIR / "sentiment_pipeline.pkl"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

ml_artifacts: dict[str, Any] = {}

# --- 2. DATA MODELS (STRUCTURED OUTPUT) ---

# Input from the Frontend
class TicketInput(BaseModel):
    ticket_text: str

# The STRUCTURED OUTPUT we demand from the LLM
class AIResponse(BaseModel):
    summary: str = Field(description="A very crisp, single-line summary of the issue (max 15 words).")
    final_priority: Literal['Low', 'Medium', 'High'] = Field(description="The final priority status determined by the LLM.")
    final_sentiment: Literal['Positive', 'Neutral', 'Negative'] = Field(description="The sentiment determined by the LLM.")
    
    # Message for the User (Chatbot)
    user_response: str = Field(description="Helpful, empathetic response to the customer. If Low priority, provide steps to solve. If High, assure them an agent is coming.")
    
    # Message for the Agent (Dashboard)
    agent_explanation: str = Field(description="Technical explanation for the support staff. Why is this priority? What is the suspected technical fault?")
    
    # Action Flag
    action: Literal['AUTO_RESOLVE', 'ESCALATE_TO_AGENT'] = Field(description="Decision on how to route the ticket.")

SUPERVISOR_TEMPLATE = """
You are AssistFlow AI, an advanced support automation engine.

INCOMING TICKET:
"{ticket}"

PRELIMINARY ANALYSIS (Fast ML Model):
- Predicted Priority: {ml_priority}
- Predicted Sentiment: {ml_sentiment}

YOUR INSTRUCTIONS:
1. Analyze the ticket content deeply.
2. Decide if the ML model's priority is correct.
   - IF the issue is simple (password reset, info request, simple config) -> Downgrade/Keep as LOW.
   - IF the issue is blocking business, data loss, or payment failure -> Upgrade to HIGH.
3. Determine the final action:
   - 'AUTO_RESOLVE': If Priority is LOW. Provide a step-by-step solution in 'user_response'.
   - 'ESCALATE_TO_AGENT': If Priority is MEDIUM or HIGH. Apologize to user in 'user_response' and tell them an agent is assigned.

OUTPUT FORMAT:
{format_instructions}
"""

# --- 3. LIFESPAN (LOAD MODELS ONCE) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if not PRIORITY_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file: {PRIORITY_MODEL_PATH}")
        if not SENTIMENT_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file: {SENTIMENT_MODEL_PATH}")

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is required.")

        ml_artifacts["priority_model"] = joblib.load(PRIORITY_MODEL_PATH)
        ml_artifacts["sentiment_model"] = joblib.load(SENTIMENT_MODEL_PATH)

        parser = PydanticOutputParser(pydantic_object=AIResponse)
        prompt = ChatPromptTemplate.from_template(SUPERVISOR_TEMPLATE)
        llm = ChatGroq(model=GROQ_MODEL, temperature=0)

        ml_artifacts["llm_chain"] = prompt | llm | parser
        ml_artifacts["format_instructions"] = parser.get_format_instructions()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        ml_artifacts.clear()
        raise RuntimeError(f"Startup failed: {exc}") from exc
    yield
    ml_artifacts.clear()

app = FastAPI(lifespan=lifespan)

# --- 4. THE CORE LOGIC CHAIN ---
def process_ticket_with_llm(ticket_text: str, ml_priority: str, ml_sentiment: str) -> AIResponse:
    chain = ml_artifacts.get("llm_chain")
    format_instructions = ml_artifacts.get("format_instructions")
    if chain is None or format_instructions is None:
        raise RuntimeError("LLM chain not initialized.")

    result = chain.invoke({
        "ticket": ticket_text,
        "ml_priority": ml_priority,
        "ml_sentiment": ml_sentiment,
        "format_instructions": format_instructions,
    })
    
    return result

# --- 5. API ENDPOINT ---
@app.post("/submit-ticket", response_model=AIResponse)
async def submit_ticket(input_data: TicketInput):
    if "priority_model" not in ml_artifacts or "sentiment_model" not in ml_artifacts:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    ticket_text = input_data.ticket_text.strip()
    if not ticket_text:
        raise HTTPException(status_code=422, detail="ticket_text cannot be empty.")

    # Step A: Fast ML Prediction (The "Triage")
    raw_priority_pred = str(ml_artifacts["priority_model"].predict([ticket_text])[0])
    raw_sentiment_pred = str(ml_artifacts["sentiment_model"].predict([ticket_text])[0])
    
    # Step B: LLM Reasoning (The "Brain")
    ai_decision = process_ticket_with_llm(
        ticket_text, 
        raw_priority_pred, 
        raw_sentiment_pred
    )
    
    # Step C: Business Logic Overrides (Optional Safety Net)
    # If the LLM says 'AUTO_RESOLVE' but the sentiment is 'Negative', 
    # we might force an escalation anyway (optional).
    if ai_decision.final_sentiment == "Negative" and ai_decision.final_priority == "Low":
        # Angry customers shouldn't deal with bots, even for simple issues.
        ai_decision.final_priority = "Medium"
        ai_decision.action = "ESCALATE_TO_AGENT"
        ai_decision.agent_explanation += " [AUTO-ESCALATED due to Negative Sentiment]"
    
    return ai_decision

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

# --- Run Command ---
# uvicorn main:app --reload
