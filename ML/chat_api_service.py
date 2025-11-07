import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types # <-- This import is used for explicit types


# Load environment variables (GEMINI_API_KEY)
load_dotenv() 

class Part(BaseModel):
    text: str

# Define the structure of a single message turn (user or model)
class Content(BaseModel):
    role: str = Field(..., pattern="^(user|model)$")
    parts: List[Part]

# Define the structure of the incoming request
class ChatRequest(BaseModel):
    history: List[Content] 
    new_message: str

# Define the structure of the outgoing response
class ChatResponse(BaseModel):
    reply: str
    updated_history: List[Content]

# --- FastAPI Setup and Client Initialization ---

app = FastAPI(
    title="Gemini Conversational API",
    description="A service for multi-turn chat using the Gemini API and persistent history."
)

# Initialize Gemini Client globally
try:
    gemini_client = genai.Client()
    GEMINI_MODEL = "gemini-2.5-flash"
except Exception as e:
    gemini_client = None
    print(f"CRITICAL: Failed to initialize Gemini Client. Check API Key. Error: {e}")


# --- API Endpoint: /chat ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receives chat history, gets a response from Gemini, and returns the updated history.
    """
    if not gemini_client:
        raise HTTPException(
            status_code=503, 
            detail="AI service is unavailable. Check server configuration."
        )

    conversation_contents = []
    
    # CORRECT: Convert Pydantic models to Gemini types.Content objects
    for message in request.history:
        # Map Pydantic structure to native Gemini types
        gemini_content = types.Content(
            role=message.role,
            parts=[types.Part.from_text(p.text) for p in message.parts]
        )
        conversation_contents.append(gemini_content)
    
    # CORRECT: Append the new user message as a Gemini types.Content object
    new_user_content_gemini = types.Content(
        role="user",
        parts=[types.Part.from_text(request.new_message)]
    )
    conversation_contents.append(new_user_content_gemini)

    try:
        # 2. Call the Gemini API with the correctly typed contents
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=conversation_contents # Now a list of types.Content objects
        )
        
        # Extract the model's reply
        model_reply = response.text
        
        # 3. Create the new model Content object using your Pydantic model for the response
        new_model_content_pydantic = Content(
            role="model",
            parts=[Part(text=model_reply)]
        )
        
        # Prepare the response payload: current history + new user message (Pydantic) + new model reply
        updated_history = request.history + [
            Content(role="user", parts=[Part(text=request.new_message)]),
            new_model_content_pydantic
        ]
        
        # 4. Return the response
        return ChatResponse(
            reply=model_reply,
            updated_history=updated_history
        )

    except Exception as e:
        # Use str(e) to get the error message
        raise HTTPException(
            status_code=500, 
            detail=f"Gemini API error: {e}"
        )

# --- Optional: Test Endpoint ---
@app.get("/")
def home():
    return {"message": "Conversational Chat API is running. Go to /docs to test the /chat endpoint."}