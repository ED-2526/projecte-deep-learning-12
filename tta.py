import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_wb_1 import MyUNet

# Intentem importar les mètriques clíniques de medpy
try:
    from medpy.metric.binary import hd95, sensitivity, specificity, dc
except ImportError:
    print("⚠️ medpy no està instal·lat. Instal·lant...")
    os.system('pip install medpy')
    from medpy.metric.binary import hd95, sensitivity, specificity, dc

from dataset import get_train_test_validation
from attention import AttentionUNet

def calculate_clinical_metrics(pred, truth):
    """Calcula mètriques evitant errors si no hi ha tumor"""
    if truth.sum() == 0 and pred.sum() == 0:
        return 1.0, 0.0, 1.0, 1.0 
    
    if truth.sum() == 0 or pred.sum() == 0:
        return 0.0, 100.0, 0.0, 0.0 
    
    dice = dc(pred, truth)
    sens = sensitivity(pred, truth)
    spec = specificity(pred, truth)
    try:
        hausdorff = hd95(pred, truth)
    except:
        hausdorff = 100.0 
        
    return dice, hausdorff, sens, spec

def main():
    DATA_DIR = "/home/edxnG12/data_processed_12"
    #MODEL_PATH = "./checkpoints/prova_best_custom_unet.pth"
    #MODEL_PATH = "./checkpoints/prova_unet_resnet34_best.pth"
    MODEL_PATH = "./checkpoints/best_attention_focaldice.pth"
    WORST_CASES_DIR = "./estudi_worst_cases/tta"
    os.makedirs(WORST_CASES_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🩺 Iniciant Avaluació Clínica amb TTA a: {device}")
    
    #model = smp.Unet(encoder_name="resnet34", in_channels=4, classes=1).to(device)
    #model = MyUNet(n_channels=4, n_classes=1, dropout_rate=0.3).to(device)
    model = AttentionUNet(n_channels=4, n_classes=1, dropout_rate=0.3).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Model carregat correctament.")
    else:
        print(f"❌ Error: No s'ha trobat el model a {MODEL_PATH}")
        return
        
    model.eval()

    # Carregar Dades (Test Set)
    _, _, test_dataset = get_train_test_validation(DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    results = []

    with torch.no_grad():
        for i, (image, true_mask) in enumerate(tqdm(test_loader, desc="Analitzant amb TTA")):
            img_tensor = image.to(device)
            
            # --- 🚀 INICI DEL TEST-TIME AUGMENTATION (TTA) 🚀 ---
            
            # 1. Predicció Original
            logits_orig = model(img_tensor)
            probs_orig = torch.sigmoid(logits_orig)
            
            # 2. Predicció Horizontal (Mirall)
            img_hflip = torch.flip(img_tensor, dims=[3]) # Girem l'amplada
            logits_hflip = model(img_hflip)
            probs_hflip = torch.sigmoid(logits_hflip)
            probs_hflip_back = torch.flip(probs_hflip, dims=[3]) # Tornem a girar la resposta
            
            # 3. Predicció Vertical (Cap per avall)
            img_vflip = torch.flip(img_tensor, dims=[2]) # Girem l'alçada
            logits_vflip = model(img_vflip)
            probs_vflip = torch.sigmoid(logits_vflip)
            probs_vflip_back = torch.flip(probs_vflip, dims=[2]) # Tornem a girar la resposta
            
            # 4. ENSAMBLATGE: Fem la mitjana de les 3 opinions
            probs_mean = (probs_orig + probs_hflip_back + probs_vflip_back) / 3.0
            
            # --- FINAL DEL TTA ---
            
            # Tornem al llindar òptim de 0.5, però ara aplicat a la mitjana!
            pred_mask = (probs_mean > 0.5).cpu().numpy().astype(bool)[0, 0]
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

    # Resultats Finals
    avg_dice = np.mean([r["dice"] for r in results])
    avg_hd95 = np.mean([r["hd95"] for r in results if r["hd95"] < 100])
    avg_sens = np.mean([r["sens"] for r in results])
    avg_spec = np.mean([r["spec"] for r in results])
    
    print("\n" + "="*50)
    print(" 🏥 RESULTATS CLÍNICS FINALS (AMB TTA) 🏥 ")
    print("="*50)
    print(f"  Dice Score Mitjà : {avg_dice:.4f}")
    print(f"  Hausdorff (HD95) : {avg_hd95:.2f} px")
    print(f"  Sensibilitat     : {avg_sens:.4f}")
    print(f"  Especificitat    : {avg_spec:.4f}")
    print("="*50)

    # Histograma
    dice_scores = [r["dice"] for r in results if r["truth"].sum() > 0]
    plt.figure(figsize=(10, 5))
    plt.hist(dice_scores, bins=40, color='lightcoral', edgecolor='black')
    plt.axvline(avg_dice, color='red', linestyle='dashed', label=f'Mitjana: {avg_dice:.2f}')
    plt.title("Distribució del Dice Score (Test-Time Augmentation)")
    plt.xlabel("Dice Score")
    plt.ylabel("Freqüència")
    plt.legend()
    plt.savefig(os.path.join(WORST_CASES_DIR, "histograma_dice_attention_tta.png"))
    plt.close()
        
    print(f"✅ Fet! Mira la carpeta: {WORST_CASES_DIR}")

if __name__ == "__main__":
    main()