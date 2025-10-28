import os
import json
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from infer import InferenceModel


class PredictRequest(BaseModel):
    features: Optional[List[float]] = None
    news_id: Optional[str] = None


class PredictResponse(BaseModel):
    probabilities: List[float]
    pred: int


app = FastAPI(title='Fake News GNN Inference')

# Instantiate global inference model (lazy load)
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/gnn_model.pt')
_INF = InferenceModel(model_path=MODEL_PATH)


@app.on_event('startup')
def load_model():
    try:
        _INF.load()
        print('Model loaded for inference')
    except Exception as e:
        print('Warning: failed loading model at startup:', e)


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    # Accept precomputed feature vector
    if req.features is None and req.news_id is None:
        raise HTTPException(status_code=400, detail='Provide features or news_id')

    if req.features is not None:
        vec = req.features
    else:
        # For now, we don't implement news_id lookup. In future we can map id->features
        raise HTTPException(status_code=501, detail='news_id lookup not implemented; pass features')

    try:
        out = _INF.predict_from_vector(vec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(probabilities=out['probabilities'], pred=out['pred'])


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8000, log_level='info')
