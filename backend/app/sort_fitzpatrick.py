import os
import csv
import shutil

# Paths
csv_path = r"C:\Users\Prateek\Downloads\archive\New folder\fitzpatrick17k (1).csv"
images_dir = r"C:\Users\Prateek\Downloads\archive\New folder\background removed"
target_dataset_dir = "dataset"

if not os.path.exists(csv_path):
    print(f"Error: CSV file nahi mili at {csv_path}")
elif not os.path.exists(images_dir):
    print(f"Error: Images folder nahi mila at {images_dir}")
else:
    os.makedirs(target_dataset_dir, exist_ok=True)
    copied_count = 0
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check available columns in CSV
        fieldnames = reader.fieldnames
        print(f"CSV Columns found: {fieldnames}")
        
        # Identify columns dynamically
        img_col = 'image' if 'image' in fieldnames else ('image_id' if 'image_id' in fieldnames else fieldnames[0])
        label_col = 'skin_condition' if 'skin_condition' in fieldnames else (fieldnames[1] if len(fieldnames) > 1 else None)
        
        print(f"Using image column: '{img_col}' and label column: '{label_col}'")
        
        if label_col:
            for row in reader:
                img_name = row.get(img_col, "").strip()
                if not img_name:
                    continue
                    
                if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_name += ".jpg"
                    
                condition = row.get(label_col, "Unknown").strip().replace(" ", "_").replace("/", "_")
                
                src_img_path = os.path.join(images_dir, img_name)
                dest_folder_path = os.path.join(target_dataset_dir, condition)
                
                if os.path.exists(src_img_path):
                    os.makedirs(dest_folder_path, exist_ok=True)
                    dst_img_path = os.path.join(dest_folder_path, f"fitz_{img_name}")
                    shutil.copy(src_img_path, dst_img_path)
                    copied_count += 1
                    
            print(f"\nSuccessfully sorted {copied_count} images into their respective dataset folders!")
        else:
            print("Label column identify nahi ho paya.")