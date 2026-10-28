import os
import shutil

# PAD-UFES dataset ka base path
pad_ufes_base_dir = r"C:\Users\Prateek\Downloads\pad_ufes\images"
target_dataset_dir = "dataset"

if not os.path.exists(pad_ufes_base_dir):
    print(f"Error: Path nahi mila at {pad_ufes_base_dir}")
else:
    print("PAD-UFES dataset deep scanning aur sorting start ho rahi hai...")
    
    copied_total = 0
    # os.walk sabhi nested folders ke andar apne aap chala jata hai
    for root, dirs, files in os.walk(pad_ufes_base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                src_img_path = os.path.join(root, file)
                
                # Target folder decide karte hain (agar koi specific folder name hai toh wo use hoga)
                dest_folder_path = os.path.join(target_dataset_dir, "PAD_UFES_Dataset")
                os.makedirs(dest_folder_path, exist_ok=True)
                
                dst_img_path = os.path.join(dest_folder_path, f"pad_{file}")
                
                # Duplicate naam se bachne ke liye check karte hain
                if not os.path.exists(dst_img_path):
                    shutil.copy(src_img_path, dst_img_path)
                    copied_total += 1

    print(f"\nSuccessfully PAD-UFES ki {copied_total} images deep scan karke copy aur sort kar di gayi hain!")