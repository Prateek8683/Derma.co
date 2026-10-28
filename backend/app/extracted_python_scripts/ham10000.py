!pip install --upgrade scikit-learn

# --- Cell Separator ---

import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print("GPU found!")
    !nvidia-smi
else:
    print("GPU not found")

# --- Cell Separator ---

import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ks
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                    Dropout, BatchNormalization, GlobalAveragePooling2D,
                                    Reshape, Layer, Input)
from tensorflow.keras.applications import ResNet50, EfficientNetB3 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping,
                                      TensorBoard, ModelCheckpoint)
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                           roc_auc_score, RocCurveDisplay,classification_report, accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from tqdm import tqdm
from tensorflow.keras import backend as K
from sklearn.utils import resample
import joblib

# --- Cell Separator ---

import sklearn
print(sklearn.__version__)
print(tf.__version__)
print(ks.__version__)
print(np.__version__)

# --- Cell Separator ---

# Load dataset
BASE_PATH = '/kaggle/input/skin-cancer-mnist-ham10000/'
metadata = pd.read_csv(BASE_PATH + 'HAM10000_metadata.csv')

print(f"Dataset shape: {metadata.shape[0]} rows × {metadata.shape[1]} columns")
print("Columns:", metadata.columns.tolist())

# --- Cell Separator ---

image_dir_1 = BASE_PATH + 'HAM10000_images_part_1/'
image_dir_2 = BASE_PATH + 'HAM10000_images_part_2/'

# --- Cell Separator ---

# Create a dictionary to map each image_id to its correct path
image_paths = {}
for img_id in metadata['image_id']:
    img_filename = img_id + '.jpg'
    if os.path.exists(image_dir_1 + img_filename):
        image_paths[img_id] = image_dir_1 + img_filename
    elif os.path.exists(image_dir_2 + img_filename):
        image_paths[img_id] = image_dir_2 + img_filename
    else:
        image_paths[img_id] = np.nan

# Now apply it to your metadata and re-split
metadata['path'] = metadata['image_id'].map(image_paths)


# --- Cell Separator ---

# Apply the mapping
metadata['path'] = metadata['image_id'].map(image_paths)

# --- Cell Separator ---

metadata = metadata.dropna(subset=['path']).reset_index(drop=True)

# --- Cell Separator ---

metadata['dx'].value_counts()

# --- Cell Separator ---

# Map to 4 classes: 3 cancer types and 1 "Not Cancer"
def map_to_4class(label):
    notcancer = 'not_cancer'
    if label == 'akiec':
        return label  # Actinic Keratoses
    elif label == 'bcc':
        return label  # Basal Cell Carcinoma
    elif label == 'mel':
        return label  # Melanoma
    else:
        return notcancer  # Not Cancer (nv, bkl, vasc, df)

# Apply mapping
metadata['label_4class'] = metadata['dx'].apply(map_to_4class)

# Print distributions to show the difference
print("Original (7-class) distribution:\n", metadata['dx'].value_counts(), "\n")
print("Mapped (4-class) distribution:\n", metadata['label_4class'].value_counts())

# --- Cell Separator ---

# 1. Original 7-class counts
orig_counts = metadata['dx'].value_counts().rename_axis('dx').reset_index(name='count')
orig_counts['label_4class'] = orig_counts['dx'].apply(map_to_4class)

# 2. Summarize into 4 classes
mapped_counts = orig_counts.groupby('label_4class')['count'].sum().reset_index()

# 3. Display both tables
print("Original (7 classes):")
print(orig_counts)
print("\nMapped (4 classes):")
print(mapped_counts)


# --- Cell Separator ---

metadata

# --- Cell Separator ---

metadata['label_4class'].value_counts()

# --- Cell Separator ---

# Visualize class distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=metadata, x='label_4class', order=metadata['label_4class'].value_counts().index)
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.show()

# --- Cell Separator ---

# Display sample images
plt.figure(figsize=(15, 10))
sample_images = metadata.groupby('label_4class').apply(lambda x: x.sample(1)).reset_index(drop=True)
for idx, row in enumerate(sample_images.itertuples(), 1):
    img = Image.open(row.path)
    plt.subplot(2, 4, idx)
    plt.imshow(img)
    plt.title(f"{row.dx}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# --- Cell Separator ---

# Resample to 3000 samples per class
resampled_df = []
for label in metadata['label_4class'].unique():
    class_subset = metadata[metadata['label_4class'] == label]
    resampled_subset = resample(class_subset,
                                replace=True,
                                n_samples=3000,
                                random_state=42)
    resampled_df.append(resampled_subset)

# --- Cell Separator ---

metadata = pd.concat(resampled_df).reset_index(drop=True)

# --- Cell Separator ---

# Display sample images
plt.figure(figsize=(15, 10))
sample_images = metadata.groupby('label_4class').apply(lambda x: x.sample(1)).reset_index(drop=True)
for idx, row in enumerate(sample_images.itertuples(), 1):
    img = Image.open(row.path)
    plt.subplot(2, 4, idx)
    plt.imshow(img)
    plt.title(f"{row.dx}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# --- Cell Separator ---

# Verify new distribution
print(metadata['label_4class'].value_counts())

# --- Cell Separator ---

# Split into train (70%), validation (15%), and test (15%)
train_df, test_df = train_test_split(metadata, test_size=0.15, stratify=metadata['label_4class'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1765, stratify=train_df['label_4class'], random_state=42)  # 0.1765 x 0.85 ≈ 0.15

# --- Cell Separator ---

f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}"

# --- Cell Separator ---

train_df['label_4class'].value_counts()

# --- Cell Separator ---

test_df['label_4class'].value_counts()

# --- Cell Separator ---

val_df['label_4class'].value_counts()

# --- Cell Separator ---

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,           # Reduce to 10-15 degrees
    width_shift_range=0.05,      # ~5% shift
    height_shift_range=0.05,
    zoom_range=0.1,              # ~10% zoom
    shear_range=0.05,            # Minimal shearing
    horizontal_flip=True,
    fill_mode='nearest'
)

# --- Cell Separator ---

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# --- Cell Separator ---

# Data Generators
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# --- Cell Separator ---

print(train_df.columns)


# --- Cell Separator ---

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='label_4class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)

# --- Cell Separator ---

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='path',
    y_col='label_4class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- Cell Separator ---

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='path',
    y_col='label_4class',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- Cell Separator ---

# Data Verification
# Check image normalization
sample_images, sample_labels = next(train_generator)
print("Image batch shape:", sample_images.shape)
print("Pixel range:", sample_images.min(), "-", sample_images.max())  # Should be 0-1

# --- Cell Separator ---

# Verify class mapping
print("Class indices:", train_generator.class_indices)
print("Sample labels (one-hot):", sample_labels[:2])

# --- Cell Separator ---

# Visualize samples
plt.figure(figsize=(12,6))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(sample_images[i])
    plt.title(list(train_generator.class_indices.keys())[np.argmax(sample_labels[i])])
    plt.axis('off')
plt.tight_layout()
plt.show()

# --- Cell Separator ---

# Convert dict_keys to a list
class_labels = list(train_generator.class_indices.keys())

print(class_labels)

# --- Cell Separator ---

# Get the unique class labels from the training data
unique_classes = np.unique(train_generator.classes)

print(unique_classes)

# --- Cell Separator ---

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=train_generator.classes
)

# --- Cell Separator ---

# Correct dictionary creation
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# --- Cell Separator ---

# To verify weights
class_weight_dict

# --- Cell Separator ---

aug_iter = train_datagen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='label_4class',
    target_size=(224, 224),
    batch_size=1,  # One image at a time for visualization
    class_mode='categorical',
    shuffle=True
)

# --- Cell Separator ---

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Models (no parallel jobs configured)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 1) Load & preprocess images
#    - Grayscale to reduce feature size (~1/3rd channels)
#    - Resize to 128x128 for better resolution
IMG_SIZE = (128, 128)

def load_images_flat(paths):
    """
    Load images in grayscale, resize, and flatten into vectors.
    Returns a 2D numpy array of shape (n_samples, IMG_SIZE[0]*IMG_SIZE[1]).
    """
    data = []
    for p in tqdm(paths, desc="Loading images", unit="img"):
        img = Image.open(p).convert('L')          # convert to grayscale
        img = img.resize(IMG_SIZE, Image.BILINEAR) # high-quality resize
        data.append(np.array(img).reshape(-1))    # flatten
    return np.vstack(data)

# Assume `metadata` DataFrame with 'path' and 'label_4class' exists
paths = metadata['path'].values
labels = metadata['label_4class'].values

# Encode labels into integers
y = LabelEncoder().fit_transform(labels)

# Split into train/test sets (80/20 stratified)
train_paths, test_paths, y_train, y_test = train_test_split(
    paths, y, test_size=0.2, stratify=y, random_state=42
)

# Load and flatten images
X_train = load_images_flat(train_paths)
X_test  = load_images_flat(test_paths)

# 2) Scale features only for linear models (LR & SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled  = scaler.transform(X_test.astype(np.float32))

# 3) Define models (no n_jobs, no parallelism)
models = [
    ("Logistic Regression", LogisticRegression(
         solver='saga', max_iter=500, random_state=42
     ), X_train_scaled, X_test_scaled),

    ("Support Vector Machine", SVC(
         kernel='rbf', probability=True, shrinking=False, random_state=42
     ), X_train_scaled, X_test_scaled),

    ("Decision Tree", DecisionTreeClassifier(
         max_depth=10, random_state=42
     ), X_train, X_test),

    ("Random Forest", RandomForestClassifier(
         n_estimators=50, random_state=42
     ), X_train, X_test),

    ("XGBoost", XGBClassifier(
         n_estimators=50, use_label_encoder=False,
         eval_metric='mlogloss', verbosity=0, random_state=42
     ), X_train, X_test)
]

# 4) Train, predict, and evaluate each model
for name, model, X_tr, X_te in tqdm(models, desc="Models", unit="model"):
    print(f"\n=== {name} ===")

    # Train the model on its designated data
    model.fit(X_tr, y_train)

    # Predict on test set
    y_pred = model.predict(X_te)

    # Print classification metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LabelEncoder().fit(labels).classes_))

    # Show confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=LabelEncoder().fit(labels).classes_,
                yticklabels=LabelEncoder().fit(labels).classes_,
                cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

# --- Cell Separator ---

# Mapping class index back to label
class_indices = aug_iter.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}

for _ in range(5):
    image_batch, label_batch = next(aug_iter)
    image = image_batch[0]
    label_idx = np.argmax(label_batch[0])
    label_name = inv_class_indices[label_idx]

    plt.imshow(image)
    plt.title(f"Label: {label_name}")
    plt.axis('off')
    plt.show()

# --- Cell Separator ---

# Callbacks
def get_callbacks(model_name):
    # Create a directory to save the models
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)  # Create if it doesn't exist

    # Set the filepath based on the model name
    if model_name == 'cnn':
        filepath = os.path.join(save_dir, 'best_cnn_model.keras') 
    elif model_name == 'resnet':
        filepath = os.path.join(save_dir, 'best_resnet_model.keras')
    elif model_name == "resnet_finetuned":
        filepath = os.path.join(save_dir, "best_resnet_finetuned.keras")
    elif model_name == "effnet":
        filepath = os.path.join(save_dir, 'best_effnet_model.keras')
    elif model_name == "effnet_finetuned":
        filepath = os.path.join(save_dir, 'best_effnet_finetuned.keras')
    else:
        filepath = os.path.join(save_dir, 'best_model.keras') 

    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6  
        ),
        ModelCheckpoint(
            filepath,  
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

# --- Cell Separator ---

cnn_model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 4 (deeper block for better performance)
    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Dense Layers
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# --- Cell Separator ---

# Compile CNN Model
cnn_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# --- Cell Separator ---

cnn_model.summary()

# --- Cell Separator ---

# Verify first batch predictions
# Initial predictions (should be ~0.25 per class)
print(cnn_model.predict(sample_images[:1]))

# --- Cell Separator ---

# Training CNN Model
print("Training CNN Model...")
cnn_history = cnn_model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks = get_callbacks('cnn') ,
    class_weight=class_weight_dict
)

# --- Cell Separator ---

# Predictions from the model
y_pred = cnn_model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# --- Cell Separator ---

# True labels from the generator
y_true = val_generator.classes

# --- Cell Separator ---

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# --- Cell Separator ---

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='magma')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# --- Cell Separator ---

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, digits=4))

# --- Cell Separator ---

cnn_test_loss, cnn_test_acc, cnn_test_auc = cnn_model.evaluate(test_generator)
print(f'CNN Test Accuracy: {cnn_test_acc:.4f}, AUC: {cnn_test_auc:.4f}')

# --- Cell Separator ---

from tensorflow.keras.applications import ResNet50


# --- Cell Separator ---

base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

inputs = Input(shape=(224, 224, 3))
x = base_resnet(inputs, training=False)  # Freeze ResNet
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x) 
x = Dropout(0.5)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x) 
x = Dropout(0.3)(x)
outputs = Dense(4, activation='softmax')(x)

# --- Cell Separator ---

resnet_model = Model(inputs, outputs)

# --- Cell Separator ---

resnet_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

# --- Cell Separator ---

resnet_model.summary()

# --- Cell Separator ---

# Training Resnet Model
print("Training ResNet Model...")
resnet_history = resnet_model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=get_callbacks('resnet'),
    class_weight=class_weight_dict
)

# --- Cell Separator ---

# Fine-tuning (Unfreeze some layers)
print("Fine-tuning ResNet...")
base_resnet.trainable = True
for layer in base_resnet.layers[:-20]:  # Unfreeze last 20 layers
    layer.trainable = False

# --- Cell Separator ---

# Debug: Print number of trainable layers
trainable_count = sum([1 for layer in base_resnet.layers if layer.trainable])
print(f"Fine-tuning ResNet: {trainable_count} layers are now trainable.")







# --- Cell Separator ---

# Recompile with lower learning rate
resnet_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# --- Cell Separator ---

resnet_finetune_history = resnet_model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=get_callbacks('resnet_finetuned'),
    class_weight=class_weight_dict
)

# --- Cell Separator ---

# Predictions from the model
y_pred = resnet_model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# --- Cell Separator ---

# True labels from the generator
y_true = val_generator.classes

# --- Cell Separator ---

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# --- Cell Separator ---

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='magma')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# --- Cell Separator ---

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, digits=4))

# --- Cell Separator ---

resnet_test_loss, resnet_test_acc, resnet_test_auc = resnet_model.evaluate(test_generator)
print(f'ResNet Test Accuracy: {resnet_test_acc:.4f}, AUC: {resnet_test_auc:.4f}')

# --- Cell Separator ---

# Evaluate DL Models
def plot_model_evaluation(history, model_name):
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title(f'{model_name} Accuracy', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o', linestyle='dashed')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o', linestyle='dashed')
    plt.title(f'{model_name} Loss', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# --- Cell Separator ---

plot_model_evaluation(cnn_history, 'CNN')
plot_model_evaluation(resnet_history, 'ResNet50 - Frozen')
plot_model_evaluation(resnet_finetune_history, 'ResNet50 - Fine-Tuned')

# --- Cell Separator ---

def evaluate_model(model, generator, model_name):
    print(f"\nEvaluating {model_name}...")
    results = model.evaluate(generator)
    print(f"Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}, AUC: {results[2]:.4f}")

    # Predictions
    y_pred = model.predict(generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = generator.classes

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels))

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # ROC AUC (OvR)
    y_true_bin = label_binarize(y_true, classes=range(len(class_labels)))
    roc_auc = roc_auc_score(y_true_bin, y_pred, multi_class='ovr')
    print(f"ROC AUC (OvR): {roc_auc:.4f}")

    # Plot ROC curves for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(class_labels)):
        RocCurveDisplay.from_predictions(
            y_true_bin[:, i],
            y_pred[:, i],
            name=f"{class_labels[i]} vs Rest",
        )
    plt.title(f'ROC Curves - {model_name}')
    plt.show()

# --- Cell Separator ---

evaluate_model(cnn_model, test_generator, "Custom CNN")
evaluate_model(resnet_model, test_generator, "Fine-tuned ResNet50")

# --- Cell Separator ---

# Plot CNN and ResNet50 training history
plt.figure(figsize=(10, 5))
plt.plot(cnn_history.history['accuracy'], label='CNN Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Val Accuracy')
plt.plot(resnet_history.history['accuracy'], label='ResNet50 Train Accuracy')
plt.plot(resnet_history.history['val_accuracy'], label='ResNet50 Val Accuracy')
plt.title("CNN vs ResNet Training History")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# --- Cell Separator ---

from tensorflow.keras.models import Model


# --- Cell Separator ---

# Feature Extraction for ML Models
def extract_features(generator, model, batch_size=32):
    features = []
    labels = []
    total_batches = len(generator)

    for i in tqdm(range(total_batches), desc="Extracting features"):
        batch_x, batch_y = generator[i]
        batch_features = model.predict(batch_x, verbose=0)
        features.append(batch_features)
        labels.append(batch_y)

        if (i + 1) * batch_size >= len(generator.filenames):
            break

    features = np.vstack(features)
    labels = np.vstack(labels)
    return features, labels

# --- Cell Separator ---

# Use ResNet for feature extraction
feature_extractor = Model(
    inputs=resnet_model.input,
    outputs=resnet_model.layers[-3].output  # Get features before final dense layer
)

# --- Cell Separator ---

# Extract features
train_features, train_labels = extract_features(train_generator, feature_extractor)
val_features, val_labels = extract_features(val_generator, feature_extractor)
test_features, test_labels = extract_features(test_generator, feature_extractor)

# --- Cell Separator ---

# Apply PCA for dimensionality reduction
pca = PCA(n_components=100)
train_features_pca = pca.fit_transform(train_features.reshape(train_features.shape[0], -1))
val_features_pca = pca.transform(val_features.reshape(val_features.shape[0], -1))
test_features_pca = pca.transform(test_features.reshape(test_features.shape[0], -1))

# --- Cell Separator ---

# Save PCA
joblib.dump(pca, 'pca_model.pkl')

# --- Cell Separator ---

# Get class labels
train_labels = np.argmax(train_labels, axis=1)
val_labels = np.argmax(val_labels, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# --- Cell Separator ---

def train_evaluate_ml_model(model, name, X_train, y_train, X_test, y_test):
    print(f"\nTraining and evaluating {name}...")
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=class_labels))

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# --- Cell Separator ---

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# --- Cell Separator ---

# Define ML models
ml_models = [
    ('Logistic Regression', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))),
    ('Decision Tree', DecisionTreeClassifier(random_state=42, max_depth=5)),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('SVM', make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
]

# --- Cell Separator ---

# Train and evaluate ML models
for name, model in ml_models:
    train_evaluate_ml_model(model, name, train_features_pca, train_labels, test_features_pca, test_labels)

# --- Cell Separator ---

import random
from PIL import Image
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import LabelEncoder

# Ensure consistent label encoding across models
label_encoder = LabelEncoder()
label_encoder.fit(metadata['label_4class'])

# Sample 5 random images
sample_df = metadata.sample(5, random_state=42).reset_index(drop=True)

# Store preprocessed data
cnn_inputs = []
ml_features = []
true_labels = []

print("\nDisplaying Sample Images and True Labels:")

for idx, row in sample_df.iterrows():
    img = Image.open(row['path']).resize((224, 224))
    img_array = img_to_array(img) / 255.0
    cnn_inputs.append(img_array)
    
    plt.figure()
    plt.imshow(img)
    plt.title(f"True: {row['label_4class']}")
    plt.axis('off')
    plt.show()
    
    # Feature extraction for ML model
    feature = feature_extractor.predict(np.expand_dims(img_array, axis=0), verbose=0)
    feature = feature.reshape(1, -1)
    pca_feature = pca.transform(feature)
    ml_features.append(pca_feature[0])
    
    true_labels.append(row['label_4class'])

cnn_inputs = np.array(cnn_inputs)
ml_features = np.array(ml_features)
true_label_indices = label_encoder.transform(true_labels)

# CNN Predictions
cnn_preds = cnn_model.predict(cnn_inputs, verbose=0)
cnn_classes = np.argmax(cnn_preds, axis=1)

# ResNet Predictions
resnet_preds = resnet_model.predict(cnn_inputs, verbose=0)
resnet_classes = np.argmax(resnet_preds, axis=1)

# Train and store ML models
trained_ml_models = []
for name, model in ml_models:
    model.fit(train_features_pca, train_labels)
    trained_ml_models.append((name, model))

ml_predictions = {}
for name, model in trained_ml_models:
    preds = model.predict(ml_features)
    ml_predictions[name] = preds


print("\nPrediction Results:")
for i in range(5):
    true_lbl = true_labels[i]
    cnn_lbl = label_encoder.inverse_transform([cnn_classes[i]])[0]
    resnet_lbl = label_encoder.inverse_transform([resnet_classes[i]])[0]

    print(f"Sample {i+1}:")
    print(f"  True Label     : {true_lbl}")
    print(f"  CNN Prediction : {cnn_lbl}")
    print(f"  ResNet Prediction : {resnet_lbl}")

    for name, preds in ml_predictions.items():
        ml_lbl = label_encoder.inverse_transform([preds[i]])[0]
        print(f"{name} Prediction : {ml_lbl}")
    
    print("-" * 40)


# --- Cell Separator ---



# --- Cell Separator ---

