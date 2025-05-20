from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import os
import tempfile
import base64
import io
import logging
from pydantic import BaseModel
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Hen Feces Chatbot API")

# List of allowed frontend domains
allowed_origins = [
    "http://localhost:3000",  # For local frontend testing
    "https://your-frontend-domain.onrender.com",  # Replace with your actual frontend domain
]

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# API key for OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")  # Loaded from .env file

# Store conversation history
conversation_history = []

# Request model for chat input
class ChatRequest(BaseModel):
    user_message: str = ""
    user_reply: str = ""
    lang: str = "english"  # Default to English

def chat_with_vet(user_message: str, user_reply: str, image: Image.Image, lang: str):
    try:
        if not isinstance(image, Image.Image):
            error_msg = "Please upload a valid image of hen feces." if lang == "english" else "Da fatan za a loda hoton kaza mai inganci."
            logger.error(error_msg)
            return {"error": error_msg}

        if not openai.api_key:
            error_msg = "No valid API key provided." if lang == "english" else "Ba a bayar da maɓallin API mai inganci ba."
            logger.error(error_msg)
            return {"error": error_msg}

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Build conversation history for OpenAI
        messages = []
        if user_message:
            messages.append({"role": "user", "content": user_message})
        if user_reply:
            messages.append({"role": "user", "content": user_reply})
        conversation_history_messages = [msg["parts"][0] for msg in conversation_history]
        messages.extend([{"role": "assistant" if i % 2 == 1 else "user", "content": msg} for i, msg in enumerate(conversation_history_messages)])

        # Construct prompt
        prompt_text = (
            f"You are an intelligent veterinary chatbot specializing in poultry. An image of hen feces is uploaded. Analyze the image and user inputs to diagnose potential diseases and predict appropriate medications. "
            f"Provide brief, clear responses in a natural, conversational tone in {lang} ('english' or 'hausa'). If more information is needed, ask one concise, relevant follow-up question at a time, up to a maximum of three. Do not mention or list future questions. If sufficient information is gathered before three questions, provide a concise prediction listing only the likely disease(s) and specific medication(s) in {lang}. "
            "Note: Not all hens are layers."
        )
        messages.insert(0, {"role": "system", "content": prompt_text})

        # Add image to the last user message
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image of hen feces:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_str}"
                    },
                },
            ],
        })

        # Generate response using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
        )
        response_text = response.choices[0].message.content.strip()

        conversation_history.append({"role": "assistant", "parts": [response_text]})
        return {"response": response_text}

    except Exception as e:
        error_msg = f"Error: {str(e)}" if lang == "english" else f"Kuskure: {str(e)}"
        logger.error(f"Exception in chat_with_vet: {str(e)}")
        return {"error": error_msg}

def clear_conversation(lang: str):
    """Clear the conversation history and return a confirmation message."""
    conversation_history.clear()
    message = "Conversation history cleared. Ready for a new case." if lang == "english" else "An share tarihin tattaunawa. A shirye don sabon shari'a."
    return {"response": message}

# API endpoint for chatting with the vet
@app.post("/chat")
async def chat_endpoint(
    image: UploadFile = File(...),
    user_message: str = Form(default=""),
    user_reply: str = Form(default=""),
    lang: str = Form(default="english")
):
    try:
        # Validate language
        if lang.lower() not in ["english", "hausa"]:
            error_msg = "Invalid language. Use 'english' or 'hausa'." if lang.lower() == "english" else "Harshen da ba daidai ba. Yi amfani da 'english' ko 'hausa'."
            raise HTTPException(status_code=400, detail=error_msg)

        # Read and validate image
        image_data = await image.read()
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}" if lang.lower() == "english" else f"Kuskure wajen sarrafa hoto: {str(e)}"
            raise HTTPException(status_code=400, detail=error_msg)
        if image.format not in ["JPEG", "PNG"]:
            error_msg = "Only JPEG or PNG images are supported." if lang.lower() == "english" else "Hotunan JPEG ko PNG kawai ake tallafawa."
            raise HTTPException(status_code=400, detail=error_msg)

        # Call chat_with_vet function
        result = chat_with_vet(user_message, user_reply, image, lang.lower())
        return result

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}" if lang.lower() == "english" else f"Kuskure wajen sarrafa buƙata: {str(e)}"
        logger.error(f"Exception in chat_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=error_msg)

# API endpoint for clearing conversation history
@app.post("/clear")
async def clear_endpoint(lang: str = Form(default="english")):
    if lang.lower() not in ["english", "hausa"]:
        error_msg = "Invalid language. Use 'english' or 'hausa'." if lang.lower() == "english" else "Harshen da ba daidai ba. Yi amfani da 'english' ko 'hausa'."
        raise HTTPException(status_code=400, detail=error_msg)
    result = clear_conversation(lang.lower())
    return result

# Root endpoint for welcome message
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Hen Feces Chatbot API! Use POST /chat with an image, optional 'user_message', 'user_reply', and 'lang' ('english' or 'hausa'). Use POST /clear to reset conversation history.",
        "hausa_message": "Barka da zuwa API na Chatbot na Kaza! Yi amfani da POST /chat tare da hoto, zaɓin 'user_message', 'user_reply', da 'lang' ('english' ko 'hausa'). Yi amfani da POST /clear don sake saita tarihin tattaunawa."
    }

# Run the app (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)