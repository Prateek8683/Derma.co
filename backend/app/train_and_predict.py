import os
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from PIL import Image
from io import BytesIO
from sklearn.linear_model import LogisticRegression
import pickle

print("1. Google Derm Foundation Model load ho raha hai...")
loaded_model = from_pretrained_keras("google/derm-foundation")
infer = loaded_model.signatures["serving_default"]

def get_skin_embedding(image_path):
    """Yeh function kisi bhi skin image ko 6144-d vector mein convert karta hai"""
    img = Image.open(image_path)
    buf = BytesIO()
    img.convert('RGB').save(buf, 'PNG')
    image_bytes = buf.getvalue()
    
    input_tensor = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image_bytes])
                )
            }
        )
    ).SerializeToString()
    
    output = infer(inputs=tf.constant([input_tensor]))
    embedding_vector = output['embedding'].numpy().flatten()
    return embedding_vector

# --- Step A: Training Data Setup ---
# Apne project folder mein kuch folders banayein jismein alag-alag bimaariyon ki photos ho, jaise:
# dataset/
#    ├── acne/
#    └── eczema/

# Yeh ek example function hai jo aapke folders se data uthayega aur train karega:
def train_skin_classifier(dataset_path):
    X = [] # Embeddings store karne ke liye
    y = [] # Labels (jaise 'acne', 'eczema') store karne ke liye
    
    print("2. Images se embeddings extract ki ja rahi hain...")
    for label_name in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, label_name)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                try:
                    # Embedding nikalein
                    embedding = get_skin_embedding(img_path)
                    X.append(embedding)
                    y.append(label_name)
                    print(f"Processed: {img_name} -> Class: {label_name}")
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")
                    
    if len(X) == 0:
        print("Koi training image nahi mili! Kripya 'dataset' folder mein classes ke mutabiq photos rakhein.")
        return

    X = np.array(X)
    y = np.array(y)
    
    print("3. Lightweight Classifier (Logistic Regression) train ho raha hai...")
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X, y)
    
    # Model ko save kar lein taki baar-baar train na karna pade
    with open('skin_classifier_model.pkl', 'wb') as f:
        pickle.dump(classifier, f)
        
    print("Training safaltapoorvak poori ho gayi! Model 'skin_classifier_model.pkl' ke roop mein save ho gaya hai.")

# --- Step B: Live Prediction Function ---
def predict_skin_condition(image_path):
    # Model load karein jo save kiya tha
    if not os.path.exists('skin_classifier_model.pkl'):
        print("Pehle classifier train karein!")
        return
        
    with open('skin_classifier_model.pkl', 'rb') as f:
        classifier = pickle.load(f)
        
    # Nayi image ka embedding nikalein
    print("Nayi image ka feature extract kiya ja raha hai...")
    live_embedding = get_skin_embedding(image_path)
    
    # Predict karein
    prediction = classifier.predict([live_embedding])
    probabilities = classifier.predict_proba([live_embedding])
    confidence = np.max(probabilities) * 100
    
    print(f"\n--- Prediction Result ---")
    print(f"Detected Condition: {prediction[0]}")
    print(f"Confidence: {confidence:.2f}%")

# --- Test karne ke liye ---
if __name__ == "__main__":
    # Agar aapko train karna hai toh yeh line un-comment karein (jab aapke paas dataset folder ho):
    # train_skin_classifier("dataset")
    
    # Agar model trained hai, toh kisi nayi photo par test karne ke liye yeh line chalayein:
    # predict_skin_condition("test_lesion.jpg")
    pass