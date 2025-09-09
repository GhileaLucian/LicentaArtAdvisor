# generate_visuals_final.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import re
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader

# Asigură-te că acest fișier există și a fost redenumit corespunzător
try:
    from model_emotional_licenta_definitiv import ArtDataset, EmotionEnsemble, get_transforms, collate_fn
except ImportError:
    print("EROARE: Nu am găsit fișierul 'model_licenta_definitiv.py'.")
    print("Asigură-te că ai redenumit scriptul tău final de antrenament în 'model_licenta_definitiv.py'")
    exit()

# --- CONFIGURARE PENTRU MODELUL CÂȘTIGĂTOR ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

# CORECȚIE 1: Folosim calea către modelul final cu 8 clase
MODEL_PATH = "models/model_final_best.pth" 
METADATA_PATH = "data/metadata_curat.csv"
TEST_DIR = "data/test"
# CORECȚIE 2: Folosim numele log-ului de la rularea cu 8 clase
LOG_FILE_PATH = "log_antrenament_final.log" 
OUTPUT_DIR = "vizualizari_licenta_final"
# CORECȚIE 3: Folosim lista cu 8 emoții
EMOTIONS = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Trust']
NUM_CLASSES = len(EMOTIONS)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Funcțiile de Generare Grafice (neschimbate, acum vor funcționa corect) ---
def plot_class_distribution(df):
    print("1/7: Se generează graficul de distribuție a claselor...")
    emotion_counts = (df[EMOTIONS] > 0).sum().sort_values(ascending=False)
    plt.figure(figsize=(14, 8)); bars = sns.barplot(x=emotion_counts.values, y=emotion_counts.index, palette="viridis")
    plt.title('Grafic 1: Distribuția Emoțiilor (8 Clase Principale)', fontsize=18, weight='bold')
    plt.xlabel('Număr de Imagini'); plt.ylabel('Emoție')
    for bar in bars.patches: bars.annotate(f'{int(bar.get_width())}', (bar.get_width() + 5, bar.get_y() + bar.get_height() / 2), ha='left', va='center')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "1_distributia_claselor.png"), dpi=300); plt.close()
    print("-> Salvat.")

def plot_learning_curves(log_path):
    print("2/7: Se generează curbele de învățare...")
    epochs, train_loss, val_loss, val_f1 = [], [], [], []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r"Epoca (\d+): Loss Antrenament: ([\d\.]+), Loss Validare: ([\d\.]+), F1-Macro Validare: ([\d\.]+).*", line)
                if match:
                    epochs.append(int(match.group(1))); train_loss.append(float(match.group(2)))
                    val_loss.append(float(match.group(3))); val_f1.append(float(match.group(4)))
    except FileNotFoundError: print(f"EROARE: Fișierul de log '{log_path}' nu a fost găsit. Verifică numele."); return
    if not epochs: print(f"AVERTISMENT: Nu am găsit date de epoci în '{log_path}'. Verifică dacă numele fișierului log este corect."); return

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(epochs, train_loss, 'o--', label='Loss Antrenament', color='cyan'); ax1.plot(epochs, val_loss, 'o-', label='Loss Validare', color='blue')
    ax1.set_xlabel("Epoci"); ax1.set_ylabel("Funcția de Cost (Loss)", color='blue'); ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left'); ax1.grid(True, which='both', linestyle='--')
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_f1, 'o-', label='F1-Macro Validare', color='red'); ax2.set_ylabel("F1-Macro Score", color='red'); ax2.tick_params(axis='y', labelcolor='red')
    best_f1_epoch = np.argmax(val_f1) + 1 if val_f1 else 1
    plt.axvline(x=best_f1_epoch, color='green', linestyle='--', label=f'Performanță Maximă (Epoca {best_f1_epoch})')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.title("Grafic 2: Curbele de Învățare (Loss vs. F1-Macro)", fontsize=18, weight='bold'); fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_curbele_de_invatare.png"), dpi=300); plt.close()
    print("-> Salvat.")

# ... (restul funcțiilor de plotare rămân la fel, ele se vor adapta automat la lista de 8 emoții) ...
def plot_per_class_f1(report):
    print("3/7: Se generează graficul F1 per clasă...")
    class_f1_scores = {k: v['f1-score'] for k, v in report.items() if k in EMOTIONS}
    class_f1_df = pd.DataFrame.from_dict(class_f1_scores, orient='index', columns=['f1-score']).sort_values('f1-score', ascending=False)
    plt.figure(figsize=(14, 8)); bars = sns.barplot(x=class_f1_df['f1-score'], y=class_f1_df.index)
    plt.title('Grafic 3: Performanța F1-Score Detaliată per Emoție', fontsize=18, weight='bold')
    plt.xlabel('F1-Score'); plt.ylabel('Emoție'); plt.xlim(0, 1)
    for bar in bars.patches: bars.annotate(f'{bar.get_width():.2f}', (bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2), ha='left', va='center')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "3_f1_per_clasa.png"), dpi=300); plt.close()
    print("-> Salvat.")
    
def plot_confusion_matrices(labels, preds):
    print("4/7: Se generează matricile de confuzie...")
    cols = 4; rows = (NUM_CLASSES + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    for i, emotion in enumerate(EMOTIONS):
        cm = confusion_matrix(labels[:, i], preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False, annot_kws={"size": 14})
        axes[i].set_title(f'Confuzie: {emotion}', fontsize=14); axes[i].set_xlabel('Prezis'); axes[i].set_ylabel('Real')
        axes[i].set_xticklabels(['Absent', 'Prezent']); axes[i].set_yticklabels(['Absent', 'Prezent'])
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.suptitle("Grafic 4: Matrici de Confuzie per Emoție", fontsize=22, weight='bold', y=1.0)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "4_matrici_confuzie.png"), dpi=300); plt.close()
    print("-> Salvat.")

def plot_precision_recall_curves(labels, outputs):
    print("5/7: Se generează curbele Precision-Recall...")
    plt.figure(figsize=(13, 9));
    colors = plt.cm.get_cmap('tab10', NUM_CLASSES)
    for i, (emotion, color) in enumerate(zip(EMOTIONS, colors.colors)):
        precision, recall, _ = precision_recall_curve(labels[:, i], outputs[:, i])
        ap = average_precision_score(labels[:, i], outputs[:, i])
        plt.plot(recall, precision, lw=2, color=color, label=f'{emotion} (AP = {ap:.2f})')
    plt.xlabel("Recall (Sensibilitate)"); plt.ylabel("Precision (Precizie)")
    plt.title("Grafic 5: Curbe Precision-Recall per Emoție", fontsize=18, weight='bold')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left"); plt.grid(True); plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, "5_curbe_pr.png"), dpi=300); plt.close()
    print("-> Salvat.")

def plot_prediction_examples(model, dataset, device, num_examples=5):
    print("6/7: Se generează exemple de predicții calitative...")
    model.eval()
    indices = random.sample(range(len(dataset)), num_examples)
    fig, axes = plt.subplots(num_examples, 2, figsize=(12, num_examples * 5))
    if num_examples == 1: axes = np.array([axes])
    for i, idx in enumerate(indices):
        image_data = dataset[idx]
        if image_data is None or image_data[0] is None: continue
        image, _ = image_data
        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        img_display = inv_normalize(image)
        axes[i, 0].imshow(np.clip(img_display.permute(1, 2, 0).numpy(), 0, 1))
        axes[i, 0].set_title(f"Imagine Test: {dataset.annotations.loc[idx, 'ImageName']}")
        axes[i, 0].axis('off')
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device)); probs = torch.sigmoid(output).cpu().squeeze().numpy()
        y_pos = np.arange(len(EMOTIONS))
        axes[i, 1].barh(y_pos, probs, align='center'); axes[i, 1].set_yticks(y_pos, labels=EMOTIONS)
        axes[i, 1].invert_yaxis(); axes[i, 1].set_xlabel('Probabilitate'); axes[i, 1].set_title('Probabilități Prezise'); axes[i, 1].set_xlim(0, 1)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "6_exemple_predictii.png"), dpi=300); plt.close()
    print("-> Salvat.")


# --- Funcția Principală ---
if __name__ == '__main__':
    full_df = pd.read_csv(METADATA_PATH, sep=',')
    plot_class_distribution(full_df)
    plot_learning_curves(LOG_FILE_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CORECȚIE: Inițiem modelul cu numărul corect de clase (8)
    model = EmotionEnsemble(num_classes=NUM_CLASSES)
    try: model.load_state_dict(torch.load(MODEL_PATH, weights_only=True)); model.to(device)
    except FileNotFoundError: print(f"EROARE: Fișierul modelului '{MODEL_PATH}' nu a fost găsit."); exit()
    except RuntimeError as e: print(f"EROARE la încărcarea modelului: {e}\nAsigură-te că modelul salvat are același număr de clase ({NUM_CLASSES}) ca cel definit aici."); exit()

    transforms_val = get_transforms()['val']
    test_dataset = ArtDataset(TEST_DIR, full_df, EMOTIONS, transforms_val)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    print("\nSe rulează predicțiile pe setul de test pentru a genera toate graficele...")
    all_labels, all_outputs = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluare pe setul de test"):
            if inputs.nelement() == 0: continue
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast(device_type="cuda"): outputs = model(inputs)
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy()); all_labels.append(labels.numpy())
    all_labels_np, all_outputs_np = np.vstack(all_labels), np.vstack(all_outputs)
    
    all_preds_np = all_outputs_np > 0.5

    report_dict = classification_report(all_labels_np, all_preds_np, target_names=EMOTIONS, zero_division=0, output_dict=True)
    plot_per_class_f1(report_dict)
    
    print("7/7: Se generează raportul de clasificare text...")
    report_text = classification_report(all_labels_np, all_preds_np, target_names=EMOTIONS, zero_division=0)
    with open(os.path.join(OUTPUT_DIR, "7_raport_clasificare.txt"), 'w', encoding='utf-8') as f: f.write(report_text)
    print("-> Salvat.")
    
    plot_confusion_matrices(all_labels_np, all_preds_np)
    plot_precision_recall_curves(all_labels_np, all_outputs_np)
    plot_prediction_examples(model, test_dataset, device)

    print(f"\n--- GATA! Toate vizualizările au fost generate în folderul '{OUTPUT_DIR}' ---")