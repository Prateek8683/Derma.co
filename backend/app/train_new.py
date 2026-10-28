import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

# ---------- SETTINGS ----------
X_FILE = "X_embeddings_new.npy"
Y_FILE = "y_labels_new.npy"
MODEL_OUT = "skin_classifier_new.pkl"
MIN_SAMPLES = 20        # classes with fewer images are dropped
TEST_SIZE = 0.2
SEED = 42

# Optional: merge overlapping classes into one name. Example:
# MERGE = {"Eczema_Photos": "eczema", "Atopic_Dermatitis_Photos": "eczema",
#          "eczema_atopic_dermatitis": "eczema"}
MERGE = {}
# ------------------------------

X = np.load(X_FILE)
y = np.load(Y_FILE, allow_pickle=True)
y = np.array([MERGE.get(str(l), str(l)) for l in y])
print("Loaded:", X.shape, "| classes:", len(set(y)))

# Drop tiny classes
counts = Counter(y)
keep = np.array([counts[l] >= MIN_SAMPLES for l in y])
dropped = sorted(l for l, c in counts.items() if c < MIN_SAMPLES)
if dropped:
    print("Dropped (fewer than", MIN_SAMPLES, "images):", dropped)
X, y = X[keep], y[keep]
print("Training on:", X.shape, "| classes:", len(set(y)))

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"),
)

print("Training... (this can take a few minutes)")
model.fit(X_tr, y_tr)

pred = model.predict(X_te)
print("\nTest accuracy:", round(accuracy_score(y_te, pred), 4))

# Top-3 accuracy
proba = model.predict_proba(X_te)
classes = model.classes_
top3 = np.argsort(proba, axis=1)[:, -3:]
top3_acc = np.mean([y_te[i] in classes[top3[i]] for i in range(len(y_te))])
print("Top-3 accuracy:", round(float(top3_acc), 4))

print("\n", classification_report(y_te, pred, zero_division=0))

joblib.dump(model, MODEL_OUT)
print("Saved model to", MODEL_OUT)
