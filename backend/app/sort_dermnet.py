import os
import shutil

# DermNet ka base path jahan train aur test folders hain
dermnet_base_dir = r"C:\Users\Prateek\Downloads\dermnet"
target_dataset_dir = "dataset"

# Process both train and test folders
splits = ["train", "test"]

for split in splits:
    split_dir = os.path.join(dermnet_base_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"Warning: '{split}' folder nahi mila, skip kiya ja raha hai.")
        continue
        
    print(f"\n--- Processing DermNet {split} folder ---")
    
    # Iske andar condition-wise folders hote hain (jaise Acne, Eczema, etc.)
    condition_folders = os.listdir(split_dir)
    
    for condition in condition_folders:
        src_condition_path = os.path.join(split_dir, condition)
        
        if os.path.isdir(src_condition_path):
            # Folder name ko clean kar lete hain (spaces ko underscore mein badalna)
            clean_condition_name = condition.strip().replace(" ", "_").replace("-", "_")
            target_folder_path = os.path.join(target_dataset_dir, clean_condition_name)
            
            # Agar project mein yeh folder nahi hai, toh naya bana do
            os.makedirs(target_folder_path, exist_ok=True)
            
            images = os.listdir(src_condition_path)
            print(f"Copying {len(images)} images from [{split}/{condition}] to [dataset/{clean_condition_name}]...")
            
            copied_count = 0
            for img in images:
                src_img_path = os.path.join(src_condition_path, img)
                # Name collision bachane ke liye prefix lagaya hai
                dst_img_path = os.path.join(target_folder_path, f"dermnet_{split}_{img}")
                
                if os.path.isfile(src_img_path):
                    shutil.copy(src_img_path, dst_img_path)
                    copied_count += 1
            print(f"-> Successfully copied {copied_count} images.")

print("\nDermNet ki saari train aur test images successfully dataset folders mein sort ho gayi hain!")