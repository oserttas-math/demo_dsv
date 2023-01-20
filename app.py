from fastapi import FastAPI
import uvicorn  # ?
import pickle


app = FastAPI()


@app.get('/')
async def root():
    return {'message': 'Prediction'}


@app.get('/predict')
async def predict(Age: int, RestingBP: int, Cholesterol: int, Oldpeak: float, FastingBS: int, MaxHR: int):
    model = pickle.load(open('catboost_model-2.pkl', 'rb'))

    prediction = model.predict(
        [[Age, RestingBP, Cholesterol, Oldpeak, FastingBS, MaxHR]]
    )

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
