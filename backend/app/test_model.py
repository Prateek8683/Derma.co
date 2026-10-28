import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from PIL import Image
from io import BytesIO

print("Model download aur load ho raha hai... (Phli baar mein thoda time lag sakta hai)")
loaded_model = from_pretrained_keras("google/derm-foundation")
infer = loaded_model.signatures["serving_default"]
print("Model safaltapoorvak load ho gaya hai!")