# Star Classifier

"Note: hosted on Render free tier — first request after inactivity may take ~30 seconds to wake up."

A machine learning API that classifies stars into one of 6 types based on their physical properties, using a Support Vector Machine (SVM) trained on astronomical data.

## Star Types
| Label | Type |
|-------|------|
| 0 | Brown Dwarf |
| 1 | Red Dwarf |
| 2 | White Dwarf |
| 3 | Main Sequence |
| 4 | Supergiant |
| 5 | Hypergiant |

## Project Structure
```
star-classifier/
├── app.py              # FastAPI app
├── train.py            # Training pipeline
├── scaler.pkl          # Fitted StandardScaler
├── svm.pkl             # Trained SVM model
├── star_data.csv       # Dataset
└── requirements.txt    # Dependencies
```

## Setup

**1. Clone the repo and install dependencies**
```bash
git clone https://github.com/yourusername/star-classifier.git
cd star-classifier
pip install -r requirements.txt
```

**2. Train the model (optional — pkl files are included)**
```bash
python train.py
```

**3. Run the API**
```bash
uvicorn app:app --reload
```

API will be available at `http://localhost:8000`

## Usage

Send a POST request to `/predict` with the star's physical measurements:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 3068,
    "luminosity": 0.0024,
    "radius": 0.17,
    "absolute_magnitude": 16.12,
    "star_color": "red",
    "spectral_class": "M"
  }'
```

**Response:**
```json
{"star_type": 0}
```

You can also test interactively via the auto-generated docs at `http://localhost:8000/docs`

## Model Performance
- Algorithm: Support Vector Machine (RBF kernel, C=10)
- Cross-validation accuracy: 99.4%
- Test accuracy: 100% (48 samples, stratified 80/20 split)
- Evaluated using precision, recall, F1-score across all 6 classes

## Pipeline
1. Data cleaning — normalized inconsistent Star color entries, removed duplicates
2. Feature scaling — StandardScaler on numeric columns
3. Encoding — one-hot encoding for categorical columns (Star color, Spectral Class)
4. Model selection — compared KNN, Decision Tree, SVM via GridSearchCV with 5-fold CV
5. Deployment — served via FastAPI
