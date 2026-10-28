1. Frontend: js/data.js me URL auto-switch

Local pe local backend, live pe Render wala:

javascript
const IS_LOCAL = ["localhost", "127.0.0.1"].includes(window.location.hostname);

export const API_BASE_URL = IS_LOCAL
  ? "http://127.0.0.1:8000"
  : "https://derma-co.onrender.com";

(Agar tumhari data.js me export nahi use ho raha, to const API_BASE_URL = ... hi rakho.)

2. Backend: app.py me CORS update

Jab frontend host hoga, uska URL bhi allow_origins me chahiye, warna browser block karega:

python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://prateek8683.github.io",   # GitHub Pages (ya jahan frontend host karo)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok", "service": "Derma-Scan API"}

Health route se Render ko pata chalta hai ki app zinda hai.

3. Model file ka masla (sabse zaroori)

Model aur .npy GitHub repo me nahi hain, to Render par app crash hogi. Model ko GitHub Release me upload karo aur startup par download karao:

python
import os, urllib.request

MODEL_PATH = "skin_classifier_new.pkl"
MODEL_URL = "https://github.com/Prateek8683/Derma.co/releases/download/v1.0/skin_classifier_new.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

Ye model load hone se pehle chalna chahiye. Release public repo me hi seedha download hota hai.

4. Render settings fix

Pichla error backend/requirements.txt not found ka tha. Render → Settings me:

Build Command: pip install -r backend/requirements.txt (agar GitHub root par backend/ dikhta hai)
Start Command: cd backend/app && uvicorn app:app --host 0.0.0.0 --port $PORT
Environment Variable: PYTHON_VERSION = 3.11.9
Health Check Path: /
