import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BraTS2DDataset(Dataset):
    def __init__(self, data_dir, patient_ids=None, augmentations=None):
        """
        :param data_dir: Ruta a la carpeta processada.
        :param patient_ids: Llista d'IDs de pacients que aquest dataset pot carregar. 
                            Si és None, els carrega tots.
        """
        self.data_dir = data_dir
        self.augmentations = augmentations
        
        # Busquem totes les imatges i màscares
        all_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.npy")))
        all_masks = sorted(glob.glob(os.path.join(data_dir, "masks", "*.npy")))
        
        # Si ens passen una llista de pacients, filtrem els arxius per quedar-nos només amb ells
        if patient_ids is not None:
            self.images_paths = [f for f in all_images if any(pid in f for pid in patient_ids)]
            self.masks_paths = [f for f in all_masks if any(pid in f for pid in patient_ids)]
        else:
            self.images_paths = all_images
            self.masks_paths = all_masks

        assert len(self.images_paths) == len(self.masks_paths), "Desajust entre imatges i màscares."

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = np.load(self.images_paths[idx])
        mask = np.load(self.masks_paths[idx])

        if self.augmentations:
            image_hwc = np.transpose(image, (1, 2, 0))
            mask_hwc = np.transpose(mask, (1, 2, 0))
            augmented = self.augmentations(image=image_hwc, mask=mask_hwc)
            image = np.transpose(augmented['image'], (2, 0, 1))
            mask = np.transpose(augmented['mask'], (2, 0, 1))

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

def get_train_val_datasets(data_dir, train_ratio=0.8, seed=42):
    """
    Funció que llegeix la carpeta, extreu els pacients, els barreja i retorna
    dos Datasets (Train i Validation) separats correctament.
    """
    # 1. Obtenir tots els arxius
    all_images = glob.glob(os.path.join(data_dir, "images", "*.npy"))
    
    # 2. Extreure els IDs únics dels pacients (ex: 'BraTS20_Training_001')
    # Agafem el nom de l'arxiu i el tallem per '_slice'
    patient_ids = list(set([os.path.basename(f).split('_slice')[0] for f in all_images]))
    
    # 3. Barrejar la llista de pacients (amb llavor fixa per poder repetir l'experiment igual)
    random.seed(seed)
    random.shuffle(patient_ids)
    
    # 4. Tallar la llista
    split_idx = int(len(patient_ids) * train_ratio)
    train_patients = patient_ids[:split_idx]
    val_patients = patient_ids[split_idx:]
    
    print(f"Total pacients: {len(patient_ids)} | Train: {len(train_patients)} | Val: {len(val_patients)}")
    
    # 5. Instanciar els Datasets
    train_dataset = BraTS2DDataset(data_dir, patient_ids=train_patients)
    val_dataset = BraTS2DDataset(data_dir, patient_ids=val_patients)
    
    return train_dataset, val_dataset

def get_train_test_validation(data_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Funció que llegeix la carpeta, extreu els pacients, els barreja i retorna
    tres Datasets (Train, Validation i Test) separats correctament.
    """

    # 1. Obtenir tots els arxius
    all_images = glob.glob(os.path.join(data_dir, "images", "*.npy"))
    
    # 2. Extreure els IDs únics dels pacients (ex: 'BraTS20_Training_001')
    patient_ids = list(set([os.path.basename(f).split('_slice')[0] for f in all_images]))
    
    # 3. Barrejar la llista de pacients (amb llavor fixa)
    random.seed(seed)
    random.shuffle(patient_ids)
    
    # 4. Calcular els índexs de tall per a 3 blocs
    total_patients = len(patient_ids)
    split1 = int(total_patients * train_ratio)
    split2 = split1 + int(total_patients * val_ratio)
    
    # 5. Tallar la llista en 3 parts
    train_patients = patient_ids[:split1]               # 0% al 70%
    val_patients = patient_ids[split1:split2]           # 70% al 85%
    test_patients = patient_ids[split2:]                # 85% al final (100%)
    
    print(f"Total pacients: {total_patients}")
    print(f"Repartiment: Train: {len(train_patients)} | Val: {len(val_patients)} | Test: {len(test_patients)}")
    
    # 6. Instanciar els 3 Datasets
    train_dataset = BraTS2DDataset(data_dir, patient_ids=train_patients)
    val_dataset = BraTS2DDataset(data_dir, patient_ids=val_patients)
    test_dataset = BraTS2DDataset(data_dir, patient_ids=test_patients)
    
    return train_dataset, val_dataset, test_dataset

# --- PROVA ---
if __name__ == "__main__":
    DIR_PROCESSED = r"C:\Users\joanb\Documents\uab\3r\XN\projecte-deep-learning-12\data_processed"
    
    # Creem la divisió 80% / 20%
    train_data, val_data = get_train_val_datasets(DIR_PROCESSED, train_ratio=0.8)
    
    print(f"Imatges a Train: {len(train_data)}")
    print(f"Imatges a Validation: {len(val_data)}")