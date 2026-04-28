import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

def normalize_slice(slice_2d):
    """Normalitza només el teixit cerebral (ignorant el fons negre)."""
    brain_mask = slice_2d > 0
    if np.sum(brain_mask) > 0:
        mean = np.mean(slice_2d[brain_mask])
        std = np.std(slice_2d[brain_mask])
        slice_2d[brain_mask] = (slice_2d[brain_mask] - mean) / (std + 1e-8)
    return slice_2d

def process_brats_dataset(input_dir, output_dir, crop_size=160, threshold=0.01):
    print("Creant directoris de sortida...")
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    # Agafem només les carpetes dels pacients
    patient_folders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    margin = (240 - crop_size) // 2

    for patient_path in tqdm(patient_folders, desc="Processant pacients"):
        patient_id = os.path.basename(patient_path)
        
        # Rutes als arxius .nii
        t1_path = os.path.join(patient_path, f"{patient_id}_t1.nii")
        t1ce_path = os.path.join(patient_path, f"{patient_id}_t1ce.nii")
        t2_path = os.path.join(patient_path, f"{patient_id}_t2.nii")
        flair_path = os.path.join(patient_path, f"{patient_id}_flair.nii")
        seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii")

        # Seguretat: Saltem si falta algun arxiu clau
        if not os.path.exists(seg_path):
            continue

        # Carregar els volums 3D
        vol_t1 = nib.load(t1_path).get_fdata()
        vol_t1ce = nib.load(t1ce_path).get_fdata()
        vol_t2 = nib.load(t2_path).get_fdata()
        vol_flair = nib.load(flair_path).get_fdata()
        vol_seg = nib.load(seg_path).get_fdata()

        # Processar cada tall de profunditat (Z)
        for z in range(vol_seg.shape[2]):
            flair_slice = vol_flair[:, :, z]
            
            # Filtratge: descartem si gairebé tot és fons fosc
            if np.count_nonzero(flair_slice) / flair_slice.size < threshold:
                continue 

            # Retall (Cropping) centrat
            s_t1 = vol_t1[margin:-margin, margin:-margin, z]
            s_t1ce = vol_t1ce[margin:-margin, margin:-margin, z]
            s_t2 = vol_t2[margin:-margin, margin:-margin, z]
            s_flair = flair_slice[margin:-margin, margin:-margin]
            s_mask = vol_seg[margin:-margin, margin:-margin, z]

            # Normalització intel·ligent
            s_t1 = normalize_slice(s_t1)
            s_t1ce = normalize_slice(s_t1ce)
            s_t2 = normalize_slice(s_t2)
            s_flair = normalize_slice(s_flair)

            # Binarització de la màscara (tumor=1, fons=0)
            s_mask = np.where(s_mask > 0, 1.0, 0.0)

            # Apilar canals en [4, H, W] i màscara en [1, H, W]
            img_stacked = np.stack([s_t1, s_t1ce, s_t2, s_flair], axis=0)
            mask_expanded = np.expand_dims(s_mask, axis=0)

            # Guardar com a arxius Numpy (.npy)
            img_filename = os.path.join(output_dir, "images", f"{patient_id}_slice{z:03d}.npy")
            mask_filename = os.path.join(output_dir, "masks", f"{patient_id}_slice{z:03d}.npy")

            np.save(img_filename, img_stacked.astype(np.float32))
            np.save(mask_filename, mask_expanded.astype(np.float32))

if __name__ == "__main__":
    # Rutes de configuració (Canvia-les si executes des del servidor de la UAB)
    INPUT_DIRECTORY = r"C:\Users\joanb\Documents\uab\3r\XN\data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData" 
    OUTPUT_DIRECTORY = r"C:\Users\joanb\Documents\uab\3r\XN\projecte-deep-learning-12\data_processed"
    
    print("Iniciant el preprocessament...")
    process_brats_dataset(INPUT_DIRECTORY, OUTPUT_DIRECTORY, crop_size=160)
    print("Preprocessament completat!")