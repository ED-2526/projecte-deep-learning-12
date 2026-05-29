import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# DESCOMMETA AQUESTA LÍNIA si tens dataset.py a la mateixa carpeta
from dataset import get_train_test_validation
from train_wb_1 import MyUNet

def calculate_dice(preds, true_masks):
    inter = (preds * true_masks).sum()
    return (2. * inter + 1e-7) / (preds.sum() + true_masks.sum() + 1e-7)

# ==========================================
# 2. FUNCIÓ PER DIBUIXAR ELS RESULTATS EN UNA GRILLA
# ==========================================
def plot_case_on_grid(ax_row, img_flair, true_mask, pred_mask, dice_score, row_title):
    # Normalitzar el fons (FLAIR) perquè es vegi bé
    bg = (img_flair - img_flair.min()) / (img_flair.max() - img_flair.min() + 1e-8)

    # Imatge 1: Original
    ax_row[0].imshow(bg, cmap='gray')
    # Posem el títol global de la fila a la primera imatge per estalviar espai
    ax_row[0].set_title(f'{row_title}\nOriginal (FLAIR)')
    ax_row[0].axis('off')

    # Imatge 2: Ground Truth (Metge) - Farem servir el verd com a fons
    ax_row[1].imshow(bg, cmap='gray')
    # Ground truth (Metge) es mostra com una màscara verda
    # Creem un RGBA amb transparència per a la màscara verda
    gt_rgba = np.zeros((bg.shape[0], bg.shape[1], 4), dtype=np.float32)
    gt_rgba[..., 1] = true_mask * 0.8  # Canal Verd
    gt_rgba[..., 3] = true_mask * 0.5  # Alpha per la màscara
    ax_row[1].imshow(gt_rgba)
    ax_row[1].set_title('Ground Truth (Metge)')
    ax_row[1].axis('off')

    # Imatge 3: Predicció (IA) - Farem servir el vermell per a la màscara predita
    ax_row[2].imshow(bg, cmap='gray')
    # Creem un RGBA amb transparència per a la màscara vermella de la IA
    pred_rgba = np.zeros((bg.shape[0], bg.shape[1], 4), dtype=np.float32)
    pred_rgba[..., 0] = pred_mask * 0.8  # Canal Vermell
    pred_rgba[..., 3] = pred_mask * 0.5  # Alpha per la màscara
    ax_row[2].imshow(pred_rgba)
    ax_row[2].set_title(f'Predicció IA (Dice: {dice_score:.4f})')
    ax_row[2].axis('off')

    # Imatge 4: OVERLAY Professional amb RGBA
    ax_row[3].imshow(bg, cmap='gray')
    
    # Creem un canal RGBA: Vermell (IA Pred), Verd (Metge), Blau (Zero)
    overlay_rgb = np.zeros((bg.shape[0], bg.shape[1], 3), dtype=np.float32)
    overlay_rgb[..., 0] = pred_mask  # Canal Vermell
    overlay_rgb[..., 1] = true_mask  # Canal Verd
    
    # Creem un canal d'alpha que només és no-zero dins de les màscares
    overlay_alpha = np.zeros((bg.shape[0], bg.shape[1]), dtype=np.float32)
    overlay_alpha[(pred_mask > 0) | (true_mask > 0)] = 0.6 # Transparència per a la màscara
    
    # Concatenem per crear un RGBA on el vermell i el verd es solapen per donar Groc (solapament)
    overlay_rgba = np.concatenate([overlay_rgb, overlay_alpha[..., None]], axis=-1)
    ax_row[3].imshow(overlay_rgba)
    
    ax_row[3].set_title('Overlay (Groc = Solapament)')
    ax_row[3].axis('off')

# ==========================================
# 3. EXECUTAR AVALUACIÓ I GENERAR LA GALERIA
# ==========================================
def main():
    DATA_DIR = "/home/edxnG12/data_processed_12"
    # COMPROVA QUE LA RUTA DEL TEU MODEL SIGUI CORRECTA
    MODEL_PATH = "./checkpoints/prova_best_custom_unet.pth" 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎨 Generant galeria de resultats a: {device}")

    # 1. Carreguem el dataset de Test
    # REMEMBRA: descommeta la teva línia d'importació del dataset al principi
    _, _, test_ds = get_train_test_validation(DATA_DIR)
    
    # Utilitzem batch_size=1 per analitzar pacient a pacient amb detall
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    # 2. Carreguem l'esquelet i els pesos de la teva IA
    model = MyUNet(n_channels=4, n_classes=1, dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    results = []

    # 3. Recollim tots els casos de Test
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(tqdm(test_loader, desc="Recollint tots els casos")):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            dice = calculate_dice(preds.view(-1), masks.view(-1)).item()
            
            # Guardem tot el material en un diccionari (convertint tensors a numpy)
            results.append({
                'idx': i,
                'img': imgs[0, 0].cpu().numpy(), # Farem servir el FLAIR (canal 0) com a fons
                'true': masks[0, 0].cpu().numpy(),
                'pred': preds[0, 0].cpu().numpy(),
                'dice': dice,
                'tumor_size': masks[0, 0].cpu().numpy().sum() # Comptem píxels de tumor per a Ground Truth
            })

    # 4. Triem exactament els dos casos que demanes:
    
    # FILTRE CRUCIAL: Ens quedem només amb pacients que tinguin un tumor d'almenys 50 píxels
    # Així evitem que el "Millor cas" sigui una llesca buida on encerta que no hi ha res.
    cases_with_tumor = [r for r in results if r['tumor_size'] > 50]
    
    if not cases_with_tumor:
        print("Error: No hi ha cap tumor a les dades de Test.")
        return
        
    # Ordenem aquesta llista filtrada de pitjor a millor
    cases_with_tumor.sort(key=lambda x: x['dice'])

    # El pitjor cas és el primer de la llista filtrada
    worst_case = cases_with_tumor[0]
    
    # El millor cas és l'últim de la llista filtrada
    best_case = cases_with_tumor[-1]

    # 5. Generem la imatge resum
    # Creem una figura amb una grilla de 2 files (Millor, Pitjor) x 4 columnes
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Posem el títol global de la figura
    plt.suptitle('Resum d\'Avaluació Clínica: Millor i Pitjor Cas del Custom MyUNet', fontsize=16)

    # Dibuixem el millor cas a la primera fila
    print("\n🌟 Generant visualització del MILLOR cas...")
    plot_case_on_grid(axes[0, :], best_case['img'], best_case['true'], best_case['pred'], best_case['dice'], 'EL MILLOR CAS')
    
    # Dibuixem el pitjor cas a la segona fila
    print("⚠️ Generant visualització del PITJOR cas...")
    plot_case_on_grid(axes[1, :], worst_case['img'], worst_case['true'], worst_case['pred'], worst_case['dice'], 'EL PITJOR CAS')
    
    plt.tight_layout()
    # Guardem l'histograma del teu model Custom
    output_filename = 'millor_i_pitjor_cas.png'
    plt.savefig(output_filename)
    plt.close()
    
    print(f"✅ Galeria de resum guardada com a '{output_filename}' correctament a la teva carpeta!")

if __name__ == "__main__":
    main()