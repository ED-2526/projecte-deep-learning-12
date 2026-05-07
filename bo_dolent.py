import os
import random
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import numpy as np
from tqdm import tqdm

from dataset import get_train_val_datasets

# Mètrica Dice ràpida per a numpy
def get_dice_numpy(pred, truth):
    intersection = np.logical_and(pred, truth).sum()
    dice = (2. * intersection) / (pred.sum() + truth.sum() + 1e-8)
    return dice

def main():
    # 1. Configuració de rutes i dispositiu
    DATA_DIR = "/home/edxnG12/data_processed_12"
    MODEL_PATH = "./checkpoints/unet_resnet34_best.pth"
    OUTPUT_IMAGE = "/home/edxnG12/Grafics/comparativa_bo_dolent.png"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Carregant model a {device} per a la comparativa clínica...")
    
    # 2. Carregar Model
    model = smp.Unet(encoder_name="resnet34", in_channels=4, classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 3. Carregar Datasets
    _, val_dataset = get_train_val_datasets(DATA_DIR)
    print(f"📊 Analitzant {len(val_dataset)} llesques de validació per trobar exemples...")

    # 4. Cercar casos de contrast (un bo, un dolent)
    # Cerquem casos que TINGUIN tumor per evitar zeros per llesques buides
    indices_amb_tumor = [i for i in tqdm(range(len(val_dataset)), desc="Cercant tumor") 
                        if val_dataset[i][1].sum() > 0]
    
    samples = []
    
    # Analitzem una submostra per anar més ràpid
    for i in random.sample(indices_amb_tumor, min(200, len(indices_amb_tumor))):
        image, true_mask = val_dataset[i]
        
        # Predicció
        image_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(image_tensor)
            pred_mask = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]
        
        truth = true_mask.cpu().numpy()[0]
        dice_val = get_dice_numpy(pred_mask, truth)
        
        samples.append((i, image.numpy(), truth, pred_mask, dice_val))
    
    # 5. Triar els millors i pitjors d'entre els que tenen tumor
    sorted_samples = sorted(samples, key=lambda x: x[4]) # Ordenem per Dice
    
    pitjor_cas = sorted_samples[0]   # El Dice més baix amb tumor
    millor_cas = sorted_samples[-1]  # El Dice més alt

    print(f"✅ Triats! Cas Bo (Idx {millor_cas[0]}, Dice: {millor_cas[4]:.4f}), Cas Dolent (Idx {pitjor_cas[0]}, Dice: {pitjor_cas[4]:.4f})")

    # 6. Visualització Clínca de 3 Columnes
    casos = [millor_cas, pitjor_cas]
    noms_casos = ["EXEMPLE D'ÈXIT (Dice Alt)", "EXEMPLE DE REPTES (Dice Baix/Zero)"]

    plt.figure(figsize=(16, 10))
    
    for i in range(2):
        cas = casos[i]
        titol = noms_casos[i]
        flair = cas[1][3] # Canal FLAIR per visualitzar
        truth = cas[2]
        pred = cas[3]
        
        # Columna 1: Original
        plt.subplot(2, 3, i*3 + 1)
        if i == 0: plt.title("1. FLAIR (Original)\nPacient de Prova", fontsize=14)
        plt.imshow(flair, cmap='gray')
        plt.ylabel(f"{titol}\nIdx: {cas[0]}", fontsize=12, fontweight='bold')
        plt.xticks([])
        plt.yticks([])

        # Columna 2: Ground Truth
        plt.subplot(2, 3, i*3 + 2)
        if i == 0: plt.title("2. Ground Truth (Metge)\nTumor real", fontsize=14)
        plt.imshow(flair, cmap='gray')
        # Pintem la màscara vermella on hi ha tumor
        plt.imshow(truth, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])

        # Columna 3: IA
        plt.subplot(2, 3, i*3 + 3)
        if i == 0: plt.title("3. Predicció (IA)\nSegmentació", fontsize=14)
        plt.imshow(flair, cmap='gray')
        # Pintem la màscara verda on prediu tumor
        plt.imshow(pred, cmap='Greens', alpha=0.5, vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        # Afegim el Dice Score final a la cantonada de la IA
        plt.text(10, 20, f"Dice: {cas[4]:.4f}", fontsize=14, color='white', fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round'))

    plt.tight_layout()
    plt.suptitle("Anàlisi Visual de Segmentació de Tumor Cerebral BraTS (UAB)\nComparativa per Separada", fontsize=18, y=1.02)
    
    plt.savefig(OUTPUT_IMAGE, bbox_inches='tight', dpi=150)
    print(f"✅ Comparativa clínica guardada com a '{OUTPUT_IMAGE}'")
    plt.show()

if __name__ == "__main__":
    main()