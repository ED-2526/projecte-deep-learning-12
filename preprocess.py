import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

def normalize_slice(slice_2d):
    """Normalitza el teixit cerebral ignorant el fons."""
    brain_mask = slice_2d > 0
    if np.sum(brain_mask) > 0:
        mean = np.mean(slice_2d[brain_mask])
        std = np.std(slice_2d[brain_mask])
        slice_2d[brain_mask] = (slice_2d[brain_mask] - mean) / (std + 1e-8)
    return slice_2d

def process_brats_dataset(input_dir, output_dir, crop_size=160, threshold=0.01):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks_bin"), exist_ok=True)   # Carpeta binària
    os.makedirs(os.path.join(output_dir, "masks_multi"), exist_ok=True) # Carpeta multiclase

    patient_folders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    margin = (240 - crop_size) // 2

    for patient_path in tqdm(patient_folders, desc="Processant pacients"):
        patient_id = os.path.basename(patient_path)
        seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii")
        if not os.path.exists(seg_path): continue

        v_t1 = nib.load(os.path.join(patient_path, f"{patient_id}_t1.nii")).get_fdata()
        v_t1ce = nib.load(os.path.join(patient_path, f"{patient_id}_t1ce.nii")).get_fdata()
        v_t2 = nib.load(os.path.join(patient_path, f"{patient_id}_t2.nii")).get_fdata()
        v_flair = nib.load(os.path.join(patient_path, f"{patient_id}_flair.nii")).get_fdata()
        v_seg = nib.load(seg_path).get_fdata()

        for z in range(v_seg.shape[2]):
            flair_slice = v_flair[:, :, z]
            if np.count_nonzero(flair_slice) / flair_slice.size < threshold: continue 

            # 1. Imatge (4 canals)
            img = np.stack([
                normalize_slice(v_t1[margin:-margin, margin:-margin, z]),
                normalize_slice(v_t1ce[margin:-margin, margin:-margin, z]),
                normalize_slice(v_t2[margin:-margin, margin:-margin, z]),
                normalize_slice(flair_slice[margin:-margin, margin:-margin])
            ], axis=0)
            
            # 2. Màscara Multiclase (Valors originals: 0, 1, 2, 4)
            mask_multi = np.expand_dims(v_seg[margin:-margin, margin:-margin, z], axis=0)

            # 3. Màscara Binària (0 o 1)
            mask_bin = np.where(mask_multi > 0, 1.0, 0.0)

            # Guardar fitxers
            base_name = f"{patient_id}_slice{z:03d}.npy"
            np.save(os.path.join(output_dir, "images", base_name), img.astype(np.float32))
            np.save(os.path.join(output_dir, "masks_multi", base_name), mask_multi.astype(np.float32))
            np.save(os.path.join(output_dir, "masks_bin", base_name), mask_bin.astype(np.float32))

if __name__ == "__main__":
    # --- MODIFICA LES RUTES DE LA TEVA MV ---
    INPUT = "/home/datasets/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    OUTPUT = "/home/datasets/BraTS2020/data_processed/mask_multiclass"
    
    print("Iniciant preprocessament...")
    process_brats_dataset(INPUT, OUTPUT)
    print("Fet!")