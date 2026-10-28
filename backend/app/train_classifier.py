import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("1. Saved embeddings load kiye ja rahe hain...")
X = np.load('X_embeddings.npy')
y = np.load('y_labels.npy')

print(f"Total samples: {len(X)}, Feature size: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Classifier (Random Forest) train kiya ja raha hai...")
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Accuracy check
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Training Accuracy: {acc * 100:.2f}%")

# Trained model ko save karein
with open('skin_classifier_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

print("3. Success! 'skin_classifier_model.pkl' successfully save ho gaya hai.")