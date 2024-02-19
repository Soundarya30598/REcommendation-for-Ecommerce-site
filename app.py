from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from typing import List, Tuple
import pandas as pd
import uvicorn
from model import SentimentRecommender

app = FastAPI()
templates = Jinja2Templates(directory="templates")

sentiment_model = SentimentRecommender()

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def prediction(request: Request, userName: str = Form(...)):
    # convert text to lowercase
    user = userName.lower()
    items = sentiment_model.getSentimentRecommendations(user)

    if items is not None:
        print(f"retrieving items....{len(items)}")
        print(items)
        return templates.TemplateResponse("index.html", {"request": request, "column_names": items.columns.values, "row_data": items.values.tolist(), "zip": zip})
    else:
        return templates.TemplateResponse("index.html", {"request": request, "message": "User Name doesn't exist, No product recommendations at this point of time!"})

@app.post("/predictSentiment")
async def predict_sentiment(request: Request, reviewText: str = Form(...)):
    print(reviewText)
    pred_sentiment = sentiment_model.classify_sentiment(reviewText)
    print(pred_sentiment)
    return templates.TemplateResponse("index.html", {"request": request, "sentiment": pred_sentiment})

if __name__ == '__main__':
    uvicorn.run("app:app")