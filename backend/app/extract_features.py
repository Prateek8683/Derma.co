import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from PIL import Image
from io import BytesIO

# Model load karein
loaded_model = from_pretrained_keras("google/derm-foundation")
infer = loaded_model.signatures["serving_default"]

def get_skin_embedding(image_path):
    # Image kholein
    img = Image.open(image_path)
    
    buf = BytesIO()
    img.convert('RGB').save(buf, 'PNG')
    image_bytes = buf.getvalue()
    
    # Model ke format ke mutabiq tensor banayein
    input_tensor = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image_bytes])
                )
            }
        )
    ).SerializeToString()
    
    # Model se output (embedding) nikalein
    output = infer(inputs=tf.constant([input_tensor]))
    embedding_vector = output['embedding'].numpy().flatten()
    
    return embedding_vector

# Test karne ke liye (apni kisi bhi skin photo ka path yahan dein):
# vector = get_skin_embedding("sample_skin.jpg")
# print("Embedding vector ka size:", vector.shape)  # Yeh (6144,) aana chahiye