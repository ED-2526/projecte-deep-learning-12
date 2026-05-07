import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm

# Intentem importar les mètriques clíniques de medpy
try:
    from medpy.metric.binary import hd95, sensitivity, specificity, dc
except ImportError:
    print("⚠️ medpy no està instal·lat. Instal·lant...")
    os.system('pip install medpy')
    from medpy.metric.binary import hd95, sensitivity, specificity, dc

from dataset import get_train_val_datasets

def calculate_clinical_metrics(pred, truth):
    """Calcula mètriques evitant errors si no hi ha tumor"""
    # Si ambdues estan buides (encert perfecte de fons)
    if truth.sum() == 0 and pred.sum() == 0:
        return 1.0, 0.0, 1.0, 1.0 
    
    # Si una té tumor i l'altra no (error greu)
    if truth.sum() == 0 or pred.sum() == 0:
        return 0.0, 100.0, 0.0, 0.0 
    
    # Càlcul normal
    dice = dc(pred, truth)
    sens = sensitivity(pred, truth)
    spec = specificity(pred, truth)
    try:
        hausdorff = hd95(pred, truth)
    except:
        hausdorff = 100.0 
        
    return dice, hausdorff, sens, spec

def main():
    # 1. Rutes (Ajustades a la teva màquina)
    DATA_DIR = "/home/edxnG12/data_processed_12"
    MODEL_PATH = "./checkpoints/unet_resnet34_best.pth"
    WORST_CASES_DIR = "/home/edxnG12/Grafics/worst_cases_analysis"
    os.makedirs(WORST_CASES_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🩺 Iniciant Avaluació Clínica a: {device}")
    
    # 2. Carregar Model
    model = smp.Unet(encoder_name="resnet34", in_channels=4, classes=1).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Model carregat correctament.")
    else:
        print(f"❌ Error: No s'ha trobat el model a {MODEL_PATH}")
        return
        
    model.eval()

    # 3. Carregar Dades (Validació)
    _, val_dataset = get_train_val_datasets(DATA_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    results = []

    # 4. Avaluar cada llesca
    with torch.no_grad():
        for i, (image, true_mask) in enumerate(tqdm(val_loader, desc="Analitzant pacients")):
            img_tensor = image.to(device)
            
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).cpu().numpy().astype(bool)[0, 0]
            truth_mask = true_mask.cpu().numpy().astype(bool)[0, 0]
            
            d, hd, s, sp = calculate_clinical_metrics(pred_mask, truth_mask)
            
            results.append({
                "index": i,
                "dice": d,
                "hd95": hd,
                "sens": s,
                "spec": sp,
                "image": image[0].numpy(), 
                "pred": pred_mask,
                "truth": truth_mask
            })

    # 5. Resultats Finals
    avg_dice = np.mean([r["dice"] for r in results])
    avg_hd95 = np.mean([r["hd95"] for r in results if r["hd95"] < 100])
    avg_sens = np.mean([r["sens"] for r in results])
    avg_spec = np.mean([r["spec"] for r in results])
    
    print("\n" + "="*50)
    print(" 🏥 RESULTATS CLÍNICS FINALS 🏥 ")
    print("="*50)
    print(f"  Dice Score Mitjà : {avg_dice:.4f}")
    print(f"  Hausdorff (HD95) : {avg_hd95:.2f} px")
    print(f"  Sensibilitat     : {avg_sens:.4f}")
    print(f"  Especificitat    : {avg_spec:.4f}")
    print("="*50)

    # 6. Histograma
    print("📊 Generant histograma...")
    dice_scores = [r["dice"] for r in results if r["truth"].sum() > 0]
    plt.figure(figsize=(10, 5))
    plt.hist(dice_scores, bins=40, color='skyblue', edgecolor='black')
    plt.axvline(avg_dice, color='red', linestyle='dashed', label=f'Mitjana: {avg_dice:.2f}')
    plt.title("Distribució del Dice Score (Casos amb tumor)")
    plt.xlabel("Dice Score")
    plt.ylabel("Freqüència")
    plt.legend()
    plt.savefig(os.path.join(WORST_CASES_DIR, "histograma_dice.png"))
    plt.close()

    # 7. Els 30 Pitjors Casos
    casos_amb_tumor = [r for r in results if r["truth"].sum() > 0]
    pitjors = sorted(casos_amb_tumor, key=lambda x: x["dice"])[:30]
    
    print("📸 Guardant els 30 pitjors casos...")
    for idx, cas in enumerate(pitjors):
        plt.figure(figsize=(12, 4))
        flair = cas["image"][3] # Canal FLAIR
        
        plt.subplot(1, 3, 1)
        plt.title(f"FLAIR (Idx {cas['index']})")
        plt.imshow(flair, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth (Metge)")
        plt.imshow(flair, cmap='gray')
        plt.imshow(cas["truth"], cmap='Reds', alpha=0.4)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title(f"IA (Dice: {cas['dice']:.2f})")
        plt.imshow(flair, cmap='gray')
        plt.imshow(cas["pred"], cmap='Greens', alpha=0.4)
        plt.axis('off')
        
        plt.savefig(os.path.join(WORST_CASES_DIR, f"pitjor_cas_{idx+1}.png"))
        plt.close()
        
    print(f"✅ Fet! Mira la carpeta: {WORST_CASES_DIR}")

if __name__ == "__main__":
    main()