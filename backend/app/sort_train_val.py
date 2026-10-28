import os
import shutil
import zipfile

# Pfad zum Downloads-Ordner an dem skin_lision Ordner
base_download_dir = r"C:\Users\Prateek\Downloads\skin_lision"
zip_path = r"C:\Users\Prateek\Downloads\skin_lision.zip"
target_dataset_dir = "dataset"

# Wann den Ordner net existiert, awer den .zip Fichier do ass, da entpaaken mir en direkt
if not os.path.exists(base_download_dir) and os.path.exists(zip_path):
    print("Entpaakt skin_lision.zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(r"C:\Users\Prateek\Downloads")
    print("Entpakt fäerdeg!")

# Mapping vun de Kuerzzeeche op déi richteg Ordner-Nummeren an Ärem Projet
folder_mapping = {
    "AK": "Actinic_Keratosis",
    "BCC": "Basal_Cell_Carcinoma_BCC",
    "BKL": "Seborrheic_Keratosis",
    "DF": "Dermatofibroma",
    "MEL": "Melanoma",
    "NV": "Melanocytic_Nevus",
    "SCC": "Squamous_Cell_Carcinoma_SCC",
    "VASC": "Vascular_Lesions",
    "UNK": "Unknown_Or_Unlabeled"
}

# Mir processéieren Train an Val Ordner
splits = ["Train", "Val"]

for split in splits:
    split_dir = os.path.join(base_download_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"Warnung: '{split}' Ordner net fonnt, gëtt iwwersprongen.")
        continue
        
    print(f"\n--- Veraarbechtung vum {split} Ordner ---")
    sub_folders = os.listdir(split_dir)
    
    for folder_name in sub_folders:
        src_folder_path = os.path.join(split_dir, folder_name)
        
        if os.path.isdir(src_folder_path):
            target_folder_name = folder_mapping.get(folder_name, folder_name)
            target_folder_path = os.path.join(target_dataset_dir, target_folder_name)
            
            os.makedirs(target_folder_path, exist_ok=True)
            
            images = os.listdir(src_folder_path)
            print(f"Kopieiert {len(images)} Biller vun [{split}/{folder_name}] op [dataset/{target_folder_name}]...")
            
            copied_count = 0
            for img in images:
                src_img_path = os.path.join(src_folder_path, img)
                dst_img_path = os.path.join(target_folder_path, f"{split}_{img}")
                
                if os.path.isfile(src_img_path):
                    shutil.copy(src_img_path, dst_img_path)
                    copied_count += 1
            print(f"-> Erfolleglech {copied_count} Biller kopiéiert.")

print("\nAll Biller aus Train an Val goufen erfolleglech an d'Dataset-Ordner sortéiert!")