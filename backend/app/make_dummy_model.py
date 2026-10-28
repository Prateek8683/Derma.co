import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

print("Dummy classifier banaya ja raha hai testing ke liye...")

# Google model 6144-d embeddings deta hai, toh hum waisa hi dummy data bana rahe hain
X_dummy = np.random.rand(10, 6144) 
y_dummy = np.array(['Acne', 'Eczema', 'Acne', 'Eczema', 'Psoriasis', 'Acne', 'Psoriasis', 'Eczema', 'Acne', 'Psoriasis'])

classifier = LogisticRegression()
classifier.fit(X_dummy, y_dummy)

# Model ko save kar lein
with open('skin_classifier_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

print("Success! 'skin_classifier_model.pkl' tayar ho gaya hai.")