import os
import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# 1. GPU Configuration & Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Bohot badhiya! GPU successfully lock ho gaya hai: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("Warning: Koi GPU detect nahi hua!")

print("\n1. Google Derm Foundation model load kiya ja raha hai (High-Performance Mode)...")
model_dir = snapshot_download(repo_id="google/derm-foundation")

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    loaded_model = tf.saved_model.load(model_dir)
    infer = loaded_model.signatures["serving_default"]
print("Model successfully load ho gaya hai!")

def process_single_image(img_path):
    """CPU par image load aur serialize karne ka helper function (Parallel chalega)"""
    try:
        img = Image.open(img_path)
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
        return input_tensor
    except Exception as e:
        return None

def process_dataset(dataset_path="dataset", batch_size=16):
    X = []
    y = []
    
    if not os.path.exists(dataset_path):
        print(f"Error: '{dataset_path}' folder nahi mila!")
        return

    print("\n2. Dataset embeddings extraction batch processing ke sath shuru ho rahi hai...")
    class_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    
    for idx, label_name in enumerate(class_folders):
        class_folder = os.path.join(dataset_path, label_name)
        images = [img for img in os.listdir(class_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
            
        print(f"\n[{idx+1}/{len(class_folders)}] Class: '{label_name}' (Total Images: {len(images)})")
        
        image_paths = [os.path.join(class_folder, img) for img in images]
        
        # CPU Multithreading use karke images ko fast load karenge
        with ThreadPoolExecutor(max_workers=8) as executor:
            serialized_tensors = list(executor.map(process_single_image, image_paths))
            
        # Valid tensors filter karein
        valid_data = [(path, tensor) for path, tensor in zip(image_paths, serialized_tensors) if tensor is not None]
        
        if not valid_data:
            continue
            
        valid_paths = [item[0] for item in valid_data]
        valid_tensors = [item[1] for item in valid_data]
        
        # Batches mein GPU par bhejen ge taaki GPU aur CPU dono full power pe hit hon
        for i in range(0, len(valid_tensors), batch_size):
            batch_tensors = valid_tensors[i:i+batch_size]
            
            with tf.device('/GPU:0' if gpus else '/CPU:0'):
                # Batch tensor prepare karein
                tensor_input = tf.constant(batch_tensors)
                output = infer(inputs=tensor_input)
                output_key = list(output.keys())[0]
                batch_embeddings = output[output_key].numpy()
                
            for emb, path in zip(batch_embeddings, valid_paths[i:i+batch_size]):
                X.append(emb.flatten())
                y.append(label_name)
                
    if len(X) == 0:
        print("Koi valid image nahi mili!")
        return

    X = np.array(X)
    y = np.array(y)
    
    np.save('X_embeddings.npy', X)
    np.save('y_labels.npy', y)
    print(f"\n Success! Total {len(X)} embeddings save ho gayi hain.")

if __name__ == "__main__":
    process_dataset("dataset")