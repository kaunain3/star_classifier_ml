from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
scaler = joblib.load("scaler.pkl")
model = joblib.load("svm.pkl")

FEATURE_COLUMNS = [
    'Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)',
    'Star color_blue white', 'Star color_orange', 'Star color_orange red',
    'Star color_pale yellow orange', 'Star color_red', 'Star color_white',
    'Star color_whitish', 'Star color_yellow white', 'Star color_yellowish',
    'Star color_yellowish white', 'Spectral Class_B', 'Spectral Class_F',
    'Spectral Class_G', 'Spectral Class_K', 'Spectral Class_M', 'Spectral Class_O'
]

class star_input(BaseModel):
    temperature: float
    luminosity: float
    radius: float
    absolute_magnitude: float
    star_color: str
    spectral_class: str

@app.post("/predict")
def predict(data: star_input):
    cleaned_color = data.star_color.lower().replace("-", " ").strip()
    cleaned_color = cleaned_color.replace("white yellow", "yellow white")
    cleaned_spectral = data.spectral_class.upper().strip()

    numeric = pd.DataFrame(
        [[data.temperature, data.luminosity, data.radius, data.absolute_magnitude]],
        columns=['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']
    )
    scaled = scaler.transform(numeric)

    input_dict = {col: 0 for col in FEATURE_COLUMNS}
    input_dict['Temperature (K)'] = scaled[0][0]
    input_dict['Luminosity(L/Lo)'] = scaled[0][1]
    input_dict['Radius(R/Ro)'] = scaled[0][2]
    input_dict['Absolute magnitude(Mv)'] = scaled[0][3]

    color_col = f"Star color_{cleaned_color}"
    spectral_col = f"Spectral Class_{cleaned_spectral}"

    if color_col in input_dict:
        input_dict[color_col] = 1
    if spectral_col in input_dict:
        input_dict[spectral_col] = 1

    input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
    prediction = model.predict(input_df)

    return {"star_type": int(prediction[0])}
