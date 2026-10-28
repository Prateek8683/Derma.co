import os
import glob
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download

# ---------- SETTINGS ----------
DATASET_DIR = "dataset"
CACHE_DIR = "embeddings_cache"
BATCH_SIZE = 16
WORKERS = 4
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SKIP_FOLDERS = {"-1", "1", "2", "3", "4", "5", "6", "ipymg-folder", "PAD_UFES_Dataset"}
# ------------------------------

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        print(e)
print("GPUs visible:", gpus)

print("Loading Derm Foundation model...")
model_dir = snapshot_download(repo_id="google/derm-foundation")
infer = tf.saved_model.load(model_dir).signatures["serving_default"]
print("Model loaded.")


def to_example(path):
    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        buf = BytesIO()
        img.save(buf, "PNG")
        return tf.train.Example(features=tf.train.Features(feature={
            "image/encoded": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[buf.getvalue()]))
        })).SerializeToString()
    except Exception as e:
        print("SKIP:", path, e)
        return None


def embed_class(label, folder):
    paths = sorted(
        p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
        if p.lower().endswith(EXTS)
    )
    chunks = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for i in range(0, len(paths), BATCH_SIZE):
            batch = paths[i:i + BATCH_SIZE]
            data = [d for d in pool.map(to_example, batch) if d is not None]
            if not data:
                continue
            out = infer(inputs=tf.constant(data))
            key = "embedding" if "embedding" in out else list(out.keys())[0]
            chunks.append(out[key].numpy().reshape(len(data), -1))
            print(f"  {label}: {min(i + BATCH_SIZE, len(paths))}/{len(paths)}", end="\r")
    print()
    return np.concatenate(chunks) if chunks else None


def main():
    if not os.path.isdir(DATASET_DIR):
        print("Dataset folder not found. Run this from backend\\app.")
        return
    os.makedirs(CACHE_DIR, exist_ok=True)

    classes = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    for n, label in enumerate(classes, 1):
        if label in SKIP_FOLDERS:
            print(f"[{n}/{len(classes)}] {label}: skipped by setting")
            continue
        out_file = os.path.join(CACHE_DIR, f"{label}.npy")
        if os.path.exists(out_file):
            print(f"[{n}/{len(classes)}] {label}: already done")
            continue
        print(f"[{n}/{len(classes)}] {label}")
        X = embed_class(label, os.path.join(DATASET_DIR, label))
        if X is not None:
            np.save(out_file, X)

    Xs, ys = [], []
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, "*.npy"))):
        arr = np.load(f)
        Xs.append(arr)
        ys.extend([os.path.splitext(os.path.basename(f))[0]] * len(arr))
    if Xs:
        np.save("X_embeddings_new.npy", np.concatenate(Xs))
        np.save("y_labels_new.npy", np.array(ys))
        print("DONE. Total embeddings:", len(ys))


if __name__ == "__main__":
    main()
