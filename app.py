from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd

app = FastAPI()
model = pickle.load(open("catboost_model-2.pkl", "rb"))


def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])


@app.get('/')
async def root():
    return {'message': 'Prediction'}


@app.get('/predict')
async def predict(Age: int, RestingBP: int, Cholesterol: int, Oldpeak: float, FastingBS: int, MaxHR: int):

    incoming_data = {
        "Age": Age,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "Oldpeak": Oldpeak,
        "FastingBS": FastingBS,
        "MaxHR": MaxHR
    }

    prediction = model_pred(incoming_data)

    if prediction == 0:
        return {"You are well. No worries :)"}
    else:
        return {
            f"Prediction: {prediction}. Kindly make an appointment with the doctor!"}

if __name__ == '__main__':

    uvicorn.run(app, port=500, host='0.0.0.0')

# sample query:
# curl -X 'GET' \
# 'http://localhost:500/predict?Age=1&RestingBP=1&Cholesterol=1&Oldpeak=1&FastingBS=1&MaxHR=1' \
# -H 'accept: application/json'
