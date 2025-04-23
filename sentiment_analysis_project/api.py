from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import time
import os

import config
from predict import SentimentPredictor

# --- Pydantic Models ---
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The input text for sentiment analysis.")

class SentimentResponse(BaseModel):
    label: str = Field(..., description="Predicted sentiment label (Positive, Negative, Neutral).")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the prediction.")
    probabilities: dict[str, float] = Field(..., description="Dictionary of probabilities for each label.")
    processing_time_ms: float = Field(..., description="Time taken for prediction in milliseconds.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of customer feedback using a fine-tuned Transformer model.",
    version="1.0.0"
)

# --- Load Predictor ---
# Load the model when the API starts. This can take a few seconds.
# In a production scenario, consider health check endpoints.
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    print("API Startup: Loading sentiment predictor...")
    start_time = time.time()
    predictor = SentimentPredictor(model_path=config.MODEL_SAVE_PATH)
    if not predictor or not predictor.model:
        print("FATAL: Failed to load predictor during startup. API may not function.")
        # Depending on deployment strategy, you might want the app to fail startup here
    else:
        end_time = time.time()
        print(f"Predictor loaded successfully in {end_time - start_time:.2f} seconds.")

# --- API Endpoint ---
@app.post("/predict/",
          response_model=SentimentResponse,
          summary="Predict Sentiment",
          description="Takes a text input and returns the predicted sentiment, confidence score, and probabilities.",
          tags=["Sentiment Analysis"])
async def predict_sentiment(request: SentimentRequest):
    """
    Analyzes the sentiment of the provided text.
    """
    global predictor
    start_time = time.time()

    if predictor is None or not predictor.model:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable. Please try again later.")

    try:
        label, confidence, probabilities = predictor.predict_single(request.text)

        if label is None:
            # This might happen if predict_single encounters an internal error
            raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")

        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000

        return SentimentResponse(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            processing_time_ms=processing_time_ms
        )
    except Exception as e:
        # Catch unexpected errors during prediction
        print(f"Error during prediction endpoint: {e}") # Log the error server-side
        # import traceback
        # print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# --- Root Endpoint (Optional - for basic check) ---
@app.get("/", summary="Health Check", tags=["General"])
async def read_root():
    """Basic endpoint to check if the API is running."""
    status = "available" if (predictor and predictor.model) else "loading or failed"
    return {"message": f"Sentiment Analysis API is running. Predictor status: {status}"}


# --- Run the API (if file is executed directly) ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    # Check if model exists before starting to avoid immediate failure
    if not os.path.isdir(config.MODEL_SAVE_PATH):
         print(f"WARNING: Model directory '{config.MODEL_SAVE_PATH}' not found.")
         print("API will start, but prediction will fail until the model is available.")
         # Allow startup anyway, startup event will handle actual loading

    uvicorn.run(
        "api:app",          # 'api' is the filename, 'app' is the FastAPI instance
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True         # Enable auto-reload for development
    )