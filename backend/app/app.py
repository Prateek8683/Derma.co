import os
import pickle
from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import snapshot_download
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Derma.co | Derma-Scan AI Classifier API", version="2.2")

# --- Bulletproof CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sabhi origins (localhost, 127.0.0.1, custom ports) allow hain
    allow_credentials=False,  # allow_origins=["*"] ke sath credentials False rakhna standard rule hai
    allow_methods=["*"],     # GET, POST, OPTIONS, PUT, DELETE sab allow hain
    allow_headers=["*"],     # Sabhi custom headers allow hain
)

# Current folder ka exact path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "skin_classifier_model.pkl")

loaded_model = None
classifier_model = None
class_names = None


@app.on_event("startup")
def load_models():
    global loaded_model, classifier_model, class_names

    print("\n[1/3] Startup: Google Derm Foundation model load ho raha hai...")
    try:
        model_dir = snapshot_download(repo_id="google/derm-foundation")
        loaded_model = tf.saved_model.load(model_dir)
        print(" -> Google Derm Foundation model successfully load ho gaya!")
    except Exception as e:
        print(f" -> Error loading Derm Foundation model: {e}")

    print("\n[2/3] Startup: Trained Classifier Model (.pkl) check ho raha hai...")
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                classifier_model = pickle.load(f)
            print(f" -> Classifier Model ('{MODEL_PATH}') successfully load ho gaya!")
        except Exception as e:
            print(f" -> Error loading .pkl model: {e}")
    else:
        print(f" -> Warning: '{MODEL_PATH}' nahi mila. Classifier train karke file yahan rakhein.")

    print("\n[3/3] Startup: Class labels check ho rahe hain...")
    if classifier_model is not None and hasattr(classifier_model, "classes_"):
        class_names = [str(name) for name in classifier_model.classes_]
        print(f" -> Labels load ho gaye ({len(class_names)} classes)")
    else:
        print(" -> Note: Classifier labels available nahi hain.")


def get_skin_embedding(img: Image.Image) -> np.ndarray:
    if loaded_model is None:
        raise RuntimeError("Google Derm Foundation model is not loaded.")

    buf = BytesIO()
    img.convert("RGB").save(buf, "PNG")
    image_bytes = buf.getvalue()

    input_tensor = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/encoded": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image_bytes])
                )
            }
        )
    ).SerializeToString()

    infer = loaded_model.signatures["serving_default"]
    output = infer(inputs=tf.constant([input_tensor]))

    output_key = list(output.keys())[0]
    return output[output_key].numpy().flatten()


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Derma-Scan AI Backend (.pkl model) is running successfully!",
        "foundation_model_loaded": loaded_model is not None,
        "classifier_loaded": classifier_model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if classifier_model is None:
        raise HTTPException(status_code=500, detail="Classifier model (skin_classifier_model.pkl) load nahi hua hai.")
    if loaded_model is None:
        raise HTTPException(
            status_code=500,
            detail="Google Derm Foundation model load nahi ho paya hai."
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file mili hai.")

        img = Image.open(BytesIO(image_bytes))

        # 1. Image se Google Derm Foundation ka embedding nikalna
        embedding = get_skin_embedding(img)

        # 2. Embedding ko 2D batch array mein convert karna
        embedding_expanded = np.expand_dims(embedding, axis=0)

        # 3. Trained scikit-learn classifier se prediction lena
        predicted_label = classifier_model.predict(embedding_expanded)[0]
        probabilities = classifier_model.predict_proba(embedding_expanded)[0]
        predicted_class_idx = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities) * 100)

        response_data = {
            "status": "success",
            "class_index": predicted_class_idx,
            "confidence_percentage": round(confidence, 2),
        }

        # Agar label_encoder.pkl maujood hai toh condition ka naam bhi bhej do
        if class_names and 0 <= predicted_class_idx < len(class_names):
            response_data["condition"] = str(predicted_label)

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
