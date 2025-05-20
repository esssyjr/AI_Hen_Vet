from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image
import os
import tempfile
from google import generativeai
import random
import io

# Initialize FastAPI app
app = FastAPI(title="Hen Feces Chatbot API")

# API keys for Google Generative AI
API_KEYS = [
    "AIzaSyBvtwP2ulNHPQexfPhhR13U30pvF2OswrU",
    "AIzaSyD0dLXPPrZmLbnHOj3f9twHmT_PZc15wMo",
]

# Store conversation history
conversation_history = []

def chat_with_vet(user_message: str, user_reply: str, image: Image.Image):
    try:
        if not isinstance(image, Image.Image):
            return {"error": "Please upload a valid image of hen feces."}

        # Save image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            image.save(temp_file.name, format="JPEG")
            image_path = temp_file.name

        # Select API key
        api_key = random.choice(API_KEYS)
        generativeai.configure(api_key=api_key)

        # Upload image
        uploaded_file = generativeai.upload_file(path=image_path, mime_type="image/jpeg")

        # Create model
        model = generativeai.GenerativeModel("gemini-1.5-flash")

        # Build conversation sequence
        prompt = (
            "You are an intelligent veterinary chatbot specializing in poultry. An image of hen feces is uploaded. Analyze the image and user inputs to diagnose potential diseases and predict appropriate medications. "
            "Provide brief, clear responses in a natural, conversational tone. If more information is needed, ask one concise, relevant follow-up question at a time, up to a maximum of three. Do not mention or list future questions. If sufficient information is gathered before three questions, provide a concise prediction listing only the likely disease(s) and specific medication(s). "
            "Note: Not all hens are layers."
        )
        
        # Update conversation
        if user_message:
            conversation_history.append({"role": "user", "parts": [user_message]})
        if user_reply:
            conversation_history.append({"role": "user", "parts": [user_reply]})

        # Add system prompt and image
        input_sequence = [uploaded_file, prompt] + [msg["parts"][0] for msg in conversation_history]

        # Generate response
        response = model.generate_content(input_sequence)
        conversation_history.append({"role": "assistant", "parts": [response.text]})
        os.unlink(image_path)

        return {"response": response.text}

    except Exception as e:
        return {"error": f"Error: {str(e)}"}

def clear_conversation():
    """Clear the conversation history and return a confirmation message."""
    conversation_history.clear()
    return {"response": "Conversation history cleared. Ready for a new case."}

# API endpoint for chatting with the vet
@app.post("/chat")
async def chat_endpoint(
    image: UploadFile = File(...),
    user_message: str = Form(default=""),
    user_reply: str = Form(default="")
):
    try:
        # Read and validate image
        image_data = await image.read()
        image = Image.open(io.BytesIO(image_data))
        if image.format not in ["JPEG", "PNG"]:
            raise HTTPException(status_code=400, detail="Only JPEG or PNG images are supported.")

        # Call chat_with_vet function
        result = chat_with_vet(user_message, user_reply, image)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# API endpoint for clearing conversation history
@app.post("/clear")
async def clear_endpoint():
    result = clear_conversation()
    return result

# Run the app (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
