from typing import Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model import Model # do not remove unused import 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('model.pth').to(DEVICE)
model.eval()

app = FastAPI()

class Features(BaseModel):
    features: Tuple[float, ...]

@app.post("/predict/")
def predict(data: Features):
    if len(data.features) != 30:
        raise HTTPException(status_code=422, detail='Number of features must be 30')
    data = torch.tensor(data.features).to(DEVICE)
    pred = model(data)
    pred = torch.sigmoid(pred)
    pred = (pred >= 0.5).int()[0].item()

    if pred:
        pred = 'M'
    else:
        pred = 'B'

    return {"diagnosis": pred}