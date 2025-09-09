# 3_antreneaza_model.py (Versiune Corectată - include num_workers)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import os
import random
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
from typing import List, Callable, Optional
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
from torch.optim.swa_utils import AveragedModel, SWALR
import torchvision.models.vision_transformer as vit
from torch.utils.tensorboard import SummaryWriter

# --- Configurare Logging și TensorBoard ---
log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("log_antrenament.log", mode='w', encoding='utf-8'),
                              logging.StreamHandler()])

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- Definiția Dataset-ului ---
class ArtDataset(Dataset):
    def __init__(self, data_dir: str, annotations_df: pd.DataFrame, emotions: List[str], transform: Optional[Callable] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.emotions = emotions
        
        existing_images = {f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        self.annotations = annotations_df[annotations_df['ImageName'].isin(existing_images)].reset_index(drop=True)
        
        logging.info(f"Pentru directorul {os.path.basename(data_dir)}, am găsit {len(self.annotations)} imagini corespunzătoare în metadata.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.loc[idx, 'ImageName']
        img_path = os.path.join(self.data_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"Eroare la deschiderea imaginii {img_path}: {e}")
            return None, None # Vom filtra aceste valori None in Dataloader

        emotion_values = pd.to_numeric(self.annotations.loc[idx, self.emotions], errors='coerce').fillna(0)
        labels = (emotion_values > 0).astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(labels.values, dtype=torch.float32)

def collate_fn(batch):
    """Funcție specială pentru a elimina imaginile care nu au putut fi încărcate."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)

# --- Transformări și Augmentare de Date ---
def get_transforms(img_size=224):
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

# --- Definirea Ansamblului de Modele ---
class EnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleModel, self).__init__()
        self.model_effnet = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        self.model_vit = vit.vit_b_16(weights=vit.ViT_B_16_Weights.DEFAULT)
        in_features_effnet = self.model_effnet.classifier[1].in_features
        self.model_effnet.classifier[1] = nn.Linear(in_features_effnet, num_classes)
        in_features_vit = self.model_vit.heads.head.in_features
        self.model_vit.heads.head = nn.Linear(in_features_vit, num_classes)

    def forward(self, x):
        return (self.model_effnet(x) + self.model_vit(x)) / 2

# --- Bucla de Antrenament și Validare ---
def run_training(model, train_loader, val_loader, criterion, optimizer, swa_model, swa_scheduler, num_epochs, patience, device, model_path):
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoca {epoch+1}/{num_epochs} [Antrenament]", unit="batch")
        for inputs, labels in progress_bar:
            if inputs.nelement() == 0: continue # Sarim peste batch-uri goale
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        
        if swa_model and epoch >= 10: # Începem SWA după 10 epoci
            swa_model.update_parameters(model)
            swa_scheduler.step()

        val_loss, val_f1_macro = evaluate(model, val_loader, criterion, device)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('F1_Macro/val', val_f1_macro, epoch)
        
        logging.info(f"Epoca {epoch+1}: Loss Antrenament: {epoch_loss:.4f}, Loss Validare: {val_loss:.4f}, F1-Macro Validare: {val_f1_macro:.4f}")
        
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            logging.info(f"Performanță îmbunătățită! Se salvează modelul la {model_path}")
        else:
            patience_counter += 1
            logging.info(f"Nicio îmbunătățire. Contor răbdare: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logging.info("Early stopping declanșat.")
            break

# --- Funcția de Evaluare ---
def evaluate(model, data_loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            if inputs.nelement() == 0: continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > threshold
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    f1_macro = f1_score(np.vstack(all_labels), np.vstack(all_preds), average='macro', zero_division=0)
    return avg_loss, f1_macro

# --- Funcția Principală ---
def main():
    parser = argparse.ArgumentParser(description="Script de Antrenament pentru Recunoașterea Emoțiilor")
    parser.add_argument("--annotations_file", type=str, default="data/metadata_curat.csv")
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/validation")
    parser.add_argument("--test_dir", type=str, default="data/test")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=15)
    # AICI ESTE LINIA ADĂUGATĂ:
    parser.add_argument("--num_workers", type=int, default=0, help="Numărul de procese paralele pentru încărcarea datelor.")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Se folosește dispozitivul: {device}")

    EMOTIONS = ['Agreeableness', 'Anger', 'Anticipation', 'Arrogance', 'Disagreeableness', 'Disgust', 
                'Fear', 'Gratitude', 'Happiness', 'Humility', 'Love', 'Optimism', 'Pessimism', 
                'Regret', 'Sadness', 'Shame', 'Shyness', 'Surprise', 'Trust', 'Neutral']
    NUM_CLASSES = len(EMOTIONS)

    full_df = pd.read_csv(args.annotations_file, sep=',')
    transforms = get_transforms()
    
    train_dataset = ArtDataset(args.train_dir, full_df, EMOTIONS, transforms['train'])
    val_dataset = ArtDataset(args.val_dir, full_df, EMOTIONS, transforms['val'])
    
    class_counts = train_dataset.annotations[EMOTIONS].gt(0).sum()
    class_weights = 1. / class_counts[class_counts > 0]
    sample_weights = np.zeros(len(train_dataset))
    for i, row in train_dataset.annotations.iterrows():
        labels = (row[EMOTIONS] > 0).astype(int)
        active_weights = class_weights[labels[labels > 0].index]
        if not active_weights.empty:
            sample_weights[i] = active_weights.sum()
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(train_dataset), replacement=True)
    
    # AICI ESTE A DOUA MODIFICARE: Folosim `args.num_workers`
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    model = EnsembleModel(NUM_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.learning_rate / 2)

    model_path = os.path.join(args.models_dir, "model_ansamblu_best.pth")
    run_training(model, train_loader, val_loader, criterion, optimizer, swa_model, swa_scheduler, 
                 args.num_epochs, args.patience, device, model_path)
    
    logging.info("Antrenament finalizat. Se evaluează cel mai bun model pe setul de test...")
    model.load_state_dict(torch.load(model_path))
    test_dataset = ArtDataset(args.test_dir, full_df, EMOTIONS, transforms['val'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loss, test_f1 = evaluate(model, test_loader, criterion, device)
    logging.info(f"Performanță Finală (Model Standard) - Loss Test: {test_loss:.4f}, F1-Macro Test: {test_f1:.4f}")

    if swa_model:
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_model_path = os.path.join(args.models_dir, "model_ansamblu_swa.pth")
        torch.save(swa_model.state_dict(), swa_model_path)
        test_loss_swa, test_f1_swa = evaluate(swa_model, test_loader, criterion, device)
        logging.info(f"Performanță Finală (Model SWA) - Loss Test: {test_loss_swa:.4f}, F1-Macro Test: {test_f1_swa:.4f}")

if __name__ == '__main__':
    main()