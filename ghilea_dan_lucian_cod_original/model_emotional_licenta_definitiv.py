# model_licenta_final.py
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
from sklearn.metrics import f1_score, precision_recall_curve
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models.vision_transformer as vit

# --- Configurare ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("log_final.log", mode='w', encoding='utf-8'),
                              logging.StreamHandler()])
writer = SummaryWriter("logs_final/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
def set_seed(seed=42): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)

# --- Dataset ---
class ArtDataset(Dataset):
    def __init__(self, data_dir, annotations_df, emotions, transform=None):
        self.data_dir = data_dir; self.transform = transform; self.emotions = emotions
        existing_images = {f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        self.annotations = annotations_df[annotations_df['ImageName'].isin(existing_images)].reset_index(drop=True)
    def __len__(self): return len(self.annotations)
    def __getitem__(self, idx):
        img_name = self.annotations.loc[idx, 'ImageName']
        img_path = os.path.join(self.data_dir, img_name)
        try: image = Image.open(img_path).convert('RGB')
        except Exception: return None, None
        emotion_values = pd.to_numeric(self.annotations.loc[idx, self.emotions], errors='coerce').fillna(0)
        labels = (emotion_values > 0).astype(np.float32)
        if self.transform: image = self.transform(image)
        return image, torch.tensor(labels.values, dtype=torch.float32)

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)

# --- Transformări ---
def get_transforms(img_size=224):
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

# --- Modelul Ansamblu ---
class EmotionEnsemble(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(EmotionEnsemble, self).__init__()
        self.effnet = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        in_features_effnet = self.effnet.classifier[1].in_features
        self.effnet.classifier = nn.Identity()
        self.vit = vit.vit_b_16(weights=vit.ViT_B_16_Weights.DEFAULT)
        in_features_vit = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features_effnet + in_features_vit),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features_effnet + in_features_vit, num_classes)
        )
    def forward(self, x):
        with torch.no_grad():
            eff_features = self.effnet(x)
            vit_features = self.vit(x)
        combined_features = torch.cat((eff_features, vit_features), dim=1)
        output = self.classifier(combined_features)
        return output

# --- Funcțiile de Antrenament și Evaluare ---
def run_training(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, num_epochs, patience, device, model_path):
    best_f1 = 0.0; patience_counter = 0
    for epoch in range(num_epochs):
        model.train(); running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoca {epoch+1}/{num_epochs}", unit="batch")
        for inputs, labels in progress_bar:
            if inputs.nelement() == 0: continue
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(inputs); loss = criterion(outputs, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss += loss.item(); progress_bar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        val_loss, val_f1_macro, _, _ = evaluate(model, val_loader, criterion, device)
        writer.add_scalar('Loss/val', val_loss, epoch); writer.add_scalar('F1_Macro/val', val_f1_macro, epoch)
        scheduler.step(val_f1_macro); current_lr = optimizer.param_groups[0]['lr']; writer.add_scalar('LearningRate', current_lr, epoch)
        logging.info(f"Epoca {epoch+1}: Loss Antrenament: {epoch_loss:.4f}, Loss Validare: {val_loss:.4f}, F1-Macro Validare: {val_f1_macro:.4f}, LR: {current_lr:.1e}")
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro; patience_counter = 0
            torch.save(model.state_dict(), model_path)
            logging.info(f"Performanță îmbunătățită! Se salvează modelul (F1: {best_f1:.4f}).")
        else:
            patience_counter += 1; logging.info(f"Nicio îmbunătățire. Contor răbdare: {patience_counter}/{patience}")
            if patience_counter >= patience: logging.info("Early stopping declanșat."); break

def evaluate(model, data_loader, criterion, device, thresholds=None):
    model.eval(); total_loss = 0.0; all_preds, all_labels, all_outputs = [], [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            if inputs.nelement() == 0: continue
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(inputs); loss = criterion(outputs, labels)
            total_loss += loss.item()
            sigmoid_outputs = torch.sigmoid(outputs)
            if thresholds is not None: preds = sigmoid_outputs > thresholds.to(device)
            else: preds = sigmoid_outputs > 0.5
            all_preds.append(preds.cpu().numpy()); all_labels.append(labels.cpu().numpy()); all_outputs.append(sigmoid_outputs.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    all_preds_np, all_labels_np = np.vstack(all_preds), np.vstack(all_labels)
    f1_macro = f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
    return avg_loss, f1_macro, all_labels_np, np.vstack(all_outputs)

def find_best_thresholds(model, val_loader, device):
    logging.info("Se caută pragurile optime de decizie pe setul de validare...")
    _, _, all_labels_np, all_outputs_np = evaluate(model, val_loader, nn.BCEWithLogitsLoss(), device)
    best_thresholds = np.zeros(all_outputs_np.shape[1])
    for i in range(all_outputs_np.shape[1]):
        best_f1 = 0; best_thresh = 0.5
        for threshold in np.linspace(0.1, 0.9, 41):
            preds = all_outputs_np[:, i] > threshold
            f1 = f1_score(all_labels_np[:, i], preds, zero_division=0)
            if f1 > best_f1: best_f1 = f1; best_thresh = threshold
        best_thresholds[i] = best_thresh
    logging.info(f"Praguri optime găsite: {[f'{t:.2f}' for t in best_thresholds]}")
    return torch.tensor(best_thresholds)

def main():
    parser = argparse.ArgumentParser(description="Script Final de Licență")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--annotations_file", type=str, default="data/metadata_curat.csv")
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/validation")
    parser.add_argument("--test_dir", type=str, default="data/test")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Se folosește dispozitivul: {device}")
    
    EMOTIONS = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Trust']
    
    full_df = pd.read_csv(args.annotations_file, sep=',')
    transforms = get_transforms()
    train_dataset = ArtDataset(args.train_dir, full_df, EMOTIONS, transforms['train'])
    val_dataset = ArtDataset(args.val_dir, full_df, EMOTIONS, transforms['val'])
    
    class_counts = train_dataset.annotations[EMOTIONS].gt(0).sum()
    class_weights = 1. / class_counts[class_counts > 0]
    sample_weights = np.zeros(len(train_dataset))
    for i, (_, row) in enumerate(train_dataset.annotations.iterrows()):
        labels = (row[EMOTIONS] > 0).astype(int)
        active_weights = class_weights[labels[labels > 0].index]
        if not active_weights.empty: sample_weights[i] = active_weights.sum()
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = EmotionEnsemble(len(EMOTIONS)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True)
    scaler = torch.amp.GradScaler()

    model_path = os.path.join(args.models_dir, "model_licenta_final.pth")
    run_training(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, args.num_epochs, args.patience, device, model_path)
    
    logging.info("Antrenament finalizat. Se încarcă cel mai bun model...")
    model.load_state_dict(torch.load(model_path))
    best_thresholds = find_best_thresholds(model, val_loader, device)

    logging.info("Evaluare finală pe setul de test cu praguri optimizate...")
    test_dataset = ArtDataset(args.test_dir, full_df, EMOTIONS, transforms['val'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_loss, test_f1, _, _ = evaluate(model, test_loader, criterion, device, thresholds=best_thresholds)
    logging.info(f"Performanță Finală de Licență - Loss Test: {test_loss:.4f}, F1-Macro Test: {test_f1:.4f}")

if __name__ == '__main__':
    main()