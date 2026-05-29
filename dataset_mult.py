import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2), 
    ])

class BraTS2DDataset(Dataset):
    def __init__(self, data_dir, patient_ids=None, augmentations=None):
        self.data_dir = data_dir
        self.augmentations = augmentations
        
        raw_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.npy")))
        
        self.images_paths = []
        self.masks_paths = []
        
        for img_path in raw_images:
            mask_path = img_path.replace("images", "masks_multiclass")
            
            if os.path.exists(mask_path):
                if patient_ids is None or any(pid in img_path for pid in patient_ids):
                    self.images_paths.append(img_path)
                    self.masks_paths.append(mask_path)
        
        assert len(self.images_paths) == len(self.masks_paths), "Desajuste crítico interno."
        if len(self.images_paths) == 0:
            print("⚠️ ALERTA: No se ha cargado ninguna pareja de imagen/máscara válida.")

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = np.load(self.images_paths[idx])
        mask = np.load(self.masks_paths[idx])

        # 1. Asegurar formato [H, W, C] para Albumentations
        if image.shape[0] in [1, 4]:
            image_hwc = np.transpose(image, (1, 2, 0))
        else:
            image_hwc = image

        # 2. Asegurar formato 2D [H, W] para la máscara
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask_hw = mask.squeeze(0)
        elif len(mask.shape) == 3 and mask.shape[-1] == 1:
            mask_hw = mask.squeeze(-1)
        else:
            mask_hw = mask

        # 3. Aplicar aumentaciones
        if self.augmentations:
            augmented = self.augmentations(image=image_hwc, mask=mask_hw)
            image_hwc = augmented['image']
            mask_hw = augmented['mask']

        # 4. 🚀 REMAPATGE DE CLASSES PER A MULTICLASSE CRÍTICA 🚀
        # Convertim els valors originals de BraTS (0, 1, 2, 4) a (0, 1, 2, 3) consecutius
        mask_multiclass = np.zeros_like(mask_hw)
        mask_multiclass[mask_hw == 1] = 1 # Necròtic es queda a 1
        mask_multiclass[mask_hw == 2] = 2 # Edema es queda a 2
        mask_multiclass[mask_hw == 4] = 3 # ⚠️ El tumor actiu (4) passa a ser la classe 3

        image_chw = np.transpose(image_hwc, (2, 0, 1))
        
        # Les màscares multiclasse de PyTorch han de ser de tipus enters (LongTensor)
        return torch.tensor(image_chw, dtype=torch.float32), torch.tensor(mask_multiclass, dtype=torch.long)

def get_train_val_test_datasets(data_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Divideix els pacients en 3 grups independents per evitar el Data Leaking."""
    all_images = glob.glob(os.path.join(data_dir, "images", "*.npy"))
    
    # Obtenir la llista de IDs únics de pacients
    patient_ids = list(set([os.path.basename(f).split('_slice')[0] for f in all_images]))
    
    random.seed(seed)
    random.shuffle(patient_ids)
    
    # Calcular els punts de tall matemàtics
    num_patients = len(patient_ids)
    train_end = int(num_patients * train_ratio)
    val_end = train_end + int(num_patients * val_ratio)
    
    train_patients = patient_ids[:train_end]
    val_patients = patient_ids[train_end:val_end]
    test_patients = patient_ids[val_end:]
    
    print("\n✂️ --- DIVISIÓ DEL DATASET SENSE DATA LEAKING ---")
    print(f"Total pacients : {num_patients}")
    print(f"👥 Train      : {len(train_patients)} pacients")
    print(f"🧪 Validation : {len(val_patients)} pacients")
    print(f"🔒 Test       : {len(test_patients)} pacients")
    print("------------------------------------------------\n")
    
    # Crear els 3 objectes Dataset independents
    train_dataset = BraTS2DDataset(data_dir, patient_ids=train_patients, augmentations=get_train_transforms())
    val_dataset = BraTS2DDataset(data_dir, patient_ids=val_patients, augmentations=None)
    test_dataset = BraTS2DDataset(data_dir, patient_ids=test_patients, augmentations=None)
    
    return train_dataset, val_dataset, test_dataset