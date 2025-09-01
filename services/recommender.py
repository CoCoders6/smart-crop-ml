import joblib
import os
from utils.preprocess import prepare_features

MODEL_PATH = os.getenv('MODEL_PATH', './models/crop_recommender.pkl')
model = joblib.load(MODEL_PATH)

def recommend_crop(soil, rain=100):
    X = prepare_features(soil, rain)
    pred = model.predict(X)[0]
    return pred
