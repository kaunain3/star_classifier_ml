from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
app = FastAPI()


scaler = joblib.load("scaler.pkl")
model = joblib.load("svm.pkl")


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
    
    numeric = pd.DataFrame([[data.temperature, data.luminosity, data.radius, data.absolute_magnitude]],
                        columns=['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)'])
    scaled = scaler.transform(numeric)
    
    input_dict = {
        'Temperature (K)': scaled[0][0],
        'Luminosity(L/Lo)': scaled[0][1],   
        'Radius(R/Ro)': scaled[0][2],
        'Absolute magnitude(Mv)': scaled[0][3],
        'Star color_blue white' : 0,
        'Star color_orange' : 0,
        'Star color_orange red' : 0,
        'Star color_red' : 0,
        'Star color_pale yellow orange' : 0,
        'Star color_white' : 0,
        'Star color_whitish' : 0,
        'Star color_yellow white' : 0,
        'Star color_yellowish' : 0, 
        'Star color_yellowish white' : 0,
        'Spectral Class_B' : 0,
        'Spectral Class_F' : 0,
        'Spectral Class_G' : 0,
        'Spectral Class_K' : 0,
        'Spectral Class_M' : 0,
        'Spectral Class_O' : 0,
    }

    color_col = f"Star color_{cleaned_color}"
    spectral_col = f"Spectral Class_{cleaned_spectral}"

    if color_col in input_dict:
       input_dict[color_col] = 1
    if spectral_col in input_dict:
     input_dict[spectral_col] = 1

    
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)
    
    return {"star_type" : int(prediction[0])}
    pass