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

class TextRequest(BaseModel):
    text: str
    max_length: int = 130
    min_length: int = 30
    do_sample: bool = False

app = FastAPI(title="Pegasus Summarization API")

textSummarization = TextSummarization()

@app.post("/summarize")
async def summarize(request: TextRequest):
    try:
        if len(request.text) < 10:
            raise HTTPException(status_code=400, detail="Text too short for summarization")

        summary = textSummarization.summarizer(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=request.do_sample
        )
        return {"summary": summary[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "active", "model": "pegasus-cnn_dailymail"}

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
        port=8000,
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
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))