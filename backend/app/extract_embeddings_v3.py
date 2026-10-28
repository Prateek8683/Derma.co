import os
import glob
import json
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download

# ---------- SETTINGS ----------
DATASET_DIR = "dataset"
CACHE_DIR = "embeddings_cache"        # finished classes (.npy)
PART_DIR = "embeddings_partial"       # in-progress classes
SKIP_LOG = "skipped_images.txt"
BATCH_SIZE = 16
SAVE_EVERY = 256                      # images between progress saves
WORKERS = 4
IMG_SIZE = 448
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SKIP_FOLDERS = {"-1", "1", "2", "3", "4", "5", "6", "ipymg-folder", "PAD_UFES_Dataset"}
# ------------------------------

for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        print(e)
print("GPUs visible:", tf.config.list_physical_devices("GPU"))

print("Loading Derm Foundation model...")
model_dir = snapshot_download(repo_id="google/derm-foundation")
infer = tf.saved_model.load(model_dir).signatures["serving_default"]
print("Model loaded.")


def log_skip(path, reason):
    with open(SKIP_LOG, "a", encoding="utf-8") as f:
        f.write(f"{path} | {reason}\n")
    print(f"\nSKIPPED: {path} ({reason})")


def to_example(path):
    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        buf = BytesIO()
        img.save(buf, "PNG")
        return tf.train.Example(features=tf.train.Features(feature={
            "image/encoded": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[buf.getvalue()]))
        })).SerializeToString()
    except Exception as e:
        log_skip(path, "cannot open: " + str(e)[:80])
        return None


def run_model(examples):
    out = infer(inputs=tf.constant(examples))
    key = "embedding" if "embedding" in out else list(out.keys())[0]
    return out[key].numpy().reshape(len(examples), -1)


def atomic_save(path, array):
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


def atomic_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def embed_class(label, folder):
    paths = sorted(
        p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
        if p.lower().endswith(EXTS)
    )
    part_x = os.path.join(PART_DIR, f"{label}.npy")
    part_meta = os.path.join(PART_DIR, f"{label}.json")

    # resume from partial progress if it exists
    done = 0
    rows = np.zeros((0, 0), dtype=np.float32)
    if os.path.exists(part_x) and os.path.exists(part_meta):
        try:
            rows = np.load(part_x)
            done = json.load(open(part_meta))["next_index"]
            print(f"  resuming at image {done}/{len(paths)}")
        except Exception:
            done, rows = 0, np.zeros((0, 0), dtype=np.float32)

    chunks = [rows] if rows.size else []
    since_save = 0

    with ThreadPoolExecutor(WORKERS) as pool:
        for i in range(done, len(paths), BATCH_SIZE):
            batch = paths[i:i + BATCH_SIZE]
            loaded = list(pool.map(to_example, batch))
            pairs = [(p, d) for p, d in zip(batch, loaded) if d is not None]

            if pairs:
                try:
                    chunks.append(run_model([d for _, d in pairs]))
                except Exception as e:
                    print("\n  batch failed, trying one by one:", str(e)[:100])
                    for p, d in pairs:
                        try:
                            chunks.append(run_model([d]))
                        except Exception as e2:
                            log_skip(p, "model error: " + str(e2)[:80])

            since_save += len(batch)
            print(f"  {label}: {min(i + BATCH_SIZE, len(paths))}/{len(paths)}", end="\r")

            if since_save >= SAVE_EVERY and chunks:
                os.makedirs(PART_DIR, exist_ok=True)
                atomic_save(part_x, np.concatenate(chunks))
                atomic_json(part_meta, {"next_index": i + BATCH_SIZE})
                since_save = 0
    print()

    result = np.concatenate(chunks) if chunks else None
    for f in (part_x, part_meta):
        if os.path.exists(f):
            os.remove(f)
    return result


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
            continue
        out_file = os.path.join(CACHE_DIR, f"{label}.npy")
        if os.path.exists(out_file):
            print(f"[{n}/{len(classes)}] {label}: already done")
            continue
        print(f"[{n}/{len(classes)}] {label}")
        try:
            X = embed_class(label, os.path.join(DATASET_DIR, label))
        except Exception as e:
            print(f"\nClass {label} failed: {e}. Moving on; run again to retry it.")
            continue
        if X is not None:
            atomic_save(out_file, X)

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
