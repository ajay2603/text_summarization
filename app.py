from text_summarizer import TextSummarization
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import torch
from dotenv import load_dotenv

load_dotenv()
import os

class DialogueRequest(BaseModel):
    text: str
    max_length: int = 64  # Optimal for dialogues
    min_length: int = 16
    temperature: float = 1.0  # For generation diversity

app = FastAPI(title="Pegasus Summarization API")

textSummarization = TextSummarization()

@app.post("/summarize")
async def summarize(request: DialogueRequest):
    try:
        if len(request.text) < 20:
            raise HTTPException(status_code=400, detail="Dialogue too short (min 20 chars)")

        result = textSummarization.pipe(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            temperature=request.temperature,
            do_sample=True if request.temperature != 1.0 else False
        )

        return {
            "summary": result[0]['generated_text'],
            "model": "pegasus-samsum",
            "parameters": {
                "max_length": request.max_length,
                "min_length": request.min_length,
                "temperature": request.temperature
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {
        "status": "active",
        "model": "transformersbook/pegasus-samsum",
        "optimal_for": "chat/dialogue summarization"
    }

# Ngrok setup
try:
    ngrok.set_auth_token(os.environ['NGROK_TOKEN'])  # Replace with your token
    nest_asyncio.apply()
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)

    # Configure server settings
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        timeout_keep_alive=60
    )
    server = uvicorn.Server(config)

    # Run server with graceful shutdown handling
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully...")
        ngrok.kill()  # Clean up ngrok tunnels
except Exception as e:
    print(f"Error starting server: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)