# make_ch6_plots.py
# Script "one-stop" pentru figurile Cap. 6 (testare & validare) – multi-label emoții artă
# Necesită: matplotlib, numpy, pandas, scikit-learn, torch, tensorboard (opțional pentru citit event files)

import os, re, json, argparse, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from itertools import combinations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ==== 1) Importă din codul tău existent ====
from train_thesis_modelV2 import (
    ArtDataset, EmotionEnsemble, get_transforms, collate_fn,
    evaluate, find_best_thresholds
)

# ---- utilitare pentru output
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

# ---- încearcă să citești TensorBoard; dacă nu, parsează log-ul text
def read_tb_scalars(tb_root):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        # ia ultimul run (cel mai recent folder)
        runs = sorted(glob.glob(os.path.join(tb_root, "*")))
        if not runs: return None
        ea = EventAccumulator(runs[-1]); ea.Reload()
        scalars = {}
        for tag in ["Loss/train", "Loss/val", "F1_Macro/val", "LearningRate"]:
            if tag in ea.Tags().get('scalars', []):
                steps, vals = zip(*[(s.step, s.value) for s in ea.Scalars(tag)])
                scalars[tag] = (np.array(steps), np.array(vals))
        return scalars
    except Exception:
        return None

def read_text_log(log_path):
    # caută linii "Loss Antrenament: X, Loss Validare: Y, F1-Macro Validare: Z"
    if not os.path.isfile(log_path): return None
    ep, loss_tr, loss_val, f1_val = [], [], [], []
    pat = re.compile(r"Loss Antrenament:\s*([0-9.]+),\s*Loss Validare:\s*([0-9.]+),\s*F1-Macro Validare:\s*([0-9.]+)")
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            m = pat.search(line)
            if m:
                loss_tr.append(float(m.group(1)))
                loss_val.append(float(m.group(2)))
                f1_val.append(float(m.group(3)))
                ep.append(len(ep))
    if not ep: return None
    return {
        "Loss/train": (np.arange(len(ep)), np.array(loss_tr)),
        "Loss/val": (np.arange(len(ep)), np.array(loss_val)),
        "F1_Macro/val": (np.arange(len(ep)), np.array(f1_val)),
    }

# ---- frecvențe și co-ocurențe
def class_frequency(df, emotions):
    # procent de imagini cu eticheta activă (>0 în CSV) – conceptual
    frec = (df[emotions] > 0).sum(axis=0).astype(int)
    return frec.sort_index()

def cooccurrence_matrix(df, emotions):
    k = len(emotions)
    M = np.zeros((k, k), dtype=int)
    B = (df[emotions] > 0).astype(int).values
    for i in range(k):
        for j in range(k):
            M[i, j] = np.sum((B[:, i] == 1) & (B[:, j] == 1))
    return M

# ---- F1 la sweep de praguri per clasă
def f1_sweep_per_class(labels, probs, emotions, thresholds=np.linspace(0.1, 0.9, 41)):
    out = {}
    for i, emo in enumerate(emotions):
        y = labels[:, i].astype(int)
        p = probs[:, i]
        f1s = []
        for t in thresholds:
            yhat = (p > t).astype(int)
            f1s.append(f1_score(y, yhat, zero_division=0))
        out[emo] = np.array(f1s)
    return thresholds, out

# ---- desen: bar simple
def barh_with_vals(ax, labels, vals, title, xlabel):
    y = np.arange(len(labels))
    ax.barh(y, vals)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title); ax.set_xlabel(xlabel)

# ==== 2) main ====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations_file", default="data/metadata_curat.csv")
    ap.add_argument("--train_dir", default="data/train")
    ap.add_argument("--val_dir", default="data/validation")
    ap.add_argument("--test_dir", default="data/test")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--tb_root", default="logs_final_definitiv")
    ap.add_argument("--text_log", default="log_final_definitiv.log")
    ap.add_argument("--out_dir", default="vizualizari_licenta/ch6")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--emotions", nargs="*", default=[
        "Anger","Anticipation","Disgust","Fear","Happiness","Sadness","Surprise","Trust"
    ])
    ap.add_argument("--tta", type=int, default=0, help="Număr de vederi TTA (0=off, 2=hflip, 3=+vflip, 4=+hvflip)")
    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # 2.1 Citește CSV
    df = pd.read_csv(args.annotations_file)
    EMO = [e for e in args.emotions if e in df.columns]
    if not EMO:
        raise ValueError("Nu am găsit coloane de emoții în CSV. Verifică --emotions și headerul CSV.")

    # 2.2 Desenează frecvențe și co-ocurențe din CSV (pe tot corpul)
    frec_all = class_frequency(df, EMO)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    barh_with_vals(ax, frec_all.index, frec_all.values, "Frecvență totală etichete (CSV)", "Număr imagini pozitive")
    savefig(os.path.join(args.out_dir, "Fig6_01_frecvente_totale.png"))

    M = cooccurrence_matrix(df, EMO)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(M, interpolation="nearest")
    ax.set_xticks(np.arange(len(EMO))); ax.set_yticks(np.arange(len(EMO)))
    ax.set_xticklabels(EMO, rotation=45, ha="right"); ax.set_yticklabels(EMO)
    ax.set_title("Co-ocurențe emoții (CSV)"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(os.path.join(args.out_dir, "Fig6_02_coocurente_csv.png"))

    # 2.3 Curbe de învățare (TensorBoard sau log)
    scalars = read_tb_scalars(args.tb_root)
    if scalars is None:
        scalars = read_text_log(args.text_log)

    if scalars is not None:
        if "Loss/train" in scalars and "Loss/val" in scalars:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(*scalars["Loss/train"], label="train")
            ax.plot(*scalars["Loss/val"], label="val")
            ax.set_title("Loss train/val pe epoci"); ax.set_xlabel("Epocă"); ax.set_ylabel("Loss"); ax.legend()
            savefig(os.path.join(args.out_dir, "Fig6_03_loss_curves.png"))
        if "F1_Macro/val" in scalars:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(*scalars["F1_Macro/val"], label="F1-Macro val")
            ax.set_title("F1-Macro pe validare"); ax.set_xlabel("Epocă"); ax.set_ylabel("F1-Macro"); ax.legend()
            savefig(os.path.join(args.out_dir, "Fig6_04_f1_val_curve.png"))

    # 2.4 Încarcă modelul, calculează praguri pe validare și evaluează pe test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = get_transforms()
    full_df = df

    val_ds  = ArtDataset(args.val_dir,  full_df, EMO, transforms['val'])
    test_ds = ArtDataset(args.test_dir, full_df, EMO, transforms['val'])
    val_ld  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = EmotionEnsemble(len(EMO)).to(device)
    # ia cel mai nou .pth din models dir, dacă nu specifici exact
    ckpts = sorted(glob.glob(os.path.join(args.models_dir, "*.pth")), key=os.path.getmtime)
    if not ckpts: raise FileNotFoundError("Nu am găsit niciun .pth în models/.")
    model.load_state_dict(torch.load(ckpts[-1], map_location=device, weights_only=True))

    # praguri optime pe validare (funcția e în codul tău)
    best_th = find_best_thresholds(model, val_ld, device)  # vector praguri per emoție
    # evaluare pe test (returnează și scorurile sigmoid dacă o chemăm fără praguri)
    criterion = nn.BCEWithLogitsLoss()
    _, _, y_val, p_val, _ = evaluate(model, val_ld, criterion, device)      # fără prag => scoruri
    _, _, y_test, p_test, _ = evaluate(model, test_ld, criterion, device)   # fără prag => scoruri

    # TTA opțional pe test
    if args.tta and args.tta > 0:
        print(f"[Info] Rulez TTA cu {min(args.tta,4)} vederi pe test...")
        views = ["id", "hflip", "vflip", "hvflip"][:min(args.tta, 4)]
        with torch.no_grad():
            p_aggr = []
            for v in views:
                pv = []
                for xb, _ in test_ld:
                    xb = xb.to(device)
                    if v == "hflip": xb_v = torch.flip(xb, dims=[3])
                    elif v == "vflip": xb_v = torch.flip(xb, dims=[2])
                    elif v == "hvflip": xb_v = torch.flip(torch.flip(xb, dims=[3]), dims=[2])
                    else: xb_v = xb
                    with torch.amp.autocast(device_type=("cuda" if device.type=="cuda" else "cpu")):
                        logits = model(xb_v)
                    pv.append(torch.sigmoid(logits).cpu().numpy())
                p_aggr.append(np.vstack(pv))
            p_test = np.mean(np.stack(p_aggr, axis=0), axis=0)

    # 2.5 PR-curves + AP per clasă (test)
    for i, emo in enumerate(EMO):
        prec, rec, _ = precision_recall_curve(y_test[:, i], p_test[:, i])
        ap = average_precision_score(y_test[:, i], p_test[:, i])
        fig, ax = plt.subplots(figsize=(4.2,3.2))
        ax.plot(rec, prec)
        ax.set_title(f"PR {emo} (AP={ap:.3f})"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        savefig(os.path.join(args.out_dir, f"Fig6_PR_{i:02d}_{emo}.png"))

    # 2.6 F1 vs prag per emoție (val) + marcaj pe pragul optim calculat
    thr_grid, f1s_map = f1_sweep_per_class(y_val, p_val, EMO)
    for i, emo in enumerate(EMO):
        fig, ax = plt.subplots(figsize=(4.2,3.0))
        ax.plot(thr_grid, f1s_map[emo], label="F1(val) vs threshold")
        ax.axvline(float(best_th[i]), linestyle="--", label=f"th*={float(best_th[i]):.2f}")
        ax.set_title(f"F1 vs prag — {emo}"); ax.set_xlabel("Prag"); ax.set_ylabel("F1"); ax.legend()
        savefig(os.path.join(args.out_dir, f"Fig6_F1sweep_{i:02d}_{emo}.png"))

    # 2.7 Histogramă de scoruri (pozitive vs negative) – test
    for i, emo in enumerate(EMO):
        fig, ax = plt.subplots(figsize=(4.2,3.0))
        ax.hist(p_test[y_test[:, i]==1, i], bins=30, alpha=0.7, label="pozitive")
        ax.hist(p_test[y_test[:, i]==0, i], bins=30, alpha=0.7, label="negative")
        ax.set_title(f"Histogramă scoruri — {emo} (test)"); ax.set_xlabel("Sigmoid score"); ax.set_ylabel("Count"); ax.legend()
        savefig(os.path.join(args.out_dir, f"Fig6_Hist_{i:02d}_{emo}.png"))

    # 2.8 Tabel cu metrici per emoție (test) + F1-Macro
    yhat_test = (p_test > best_th.numpy().reshape(1, -1)).astype(int)
    rows = []
    for i, emo in enumerate(EMO):
        f1 = f1_score(y_test[:, i], yhat_test[:, i], zero_division=0)
        prec, rec, _ = precision_recall_curve(y_test[:, i], p_test[:, i])
        ap = average_precision_score(y_test[:, i], p_test[:, i])
        rows.append({"Emotion": emo, "F1_test": f1, "AP_test": ap, "Threshold*": float(best_th[i])})
    macro = f1_score(y_test, yhat_test, average="macro", zero_division=0)
    micro = f1_score(y_test, yhat_test, average="micro", zero_division=0)
    samples = f1_score(y_test, yhat_test, average="samples", zero_division=0)
    exact_match = float((y_test == yhat_test).all(axis=1).mean())
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "Tab6_metrics_test.csv"), index=False)
    with open(os.path.join(args.out_dir, "macro_f1_test.txt"), "w", encoding="utf-8") as f:
        f.write(f"F1-Macro (test) = {macro:.4f}\n")
    # rezumat extins
    with open(os.path.join(args.out_dir, "summary_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"F1-Macro (test): {macro:.4f}\n")
        f.write(f"F1-Micro (test): {micro:.4f}\n")
        f.write(f"F1-Samples (test): {samples:.4f}\n")
        f.write(f"Exact-Match (subset accuracy): {exact_match:.4f}\n")
    # salvăm pragurile
    thr_map = {emo: float(best_th[i]) for i, emo in enumerate(EMO)}
    import json
    with open(os.path.join(args.out_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thr_map, f, ensure_ascii=False, indent=2)

    # 2.9 Bar charts pentru F1/AP per clasă
    df_metrics = pd.DataFrame(rows)
    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=df_metrics.sort_values("F1_test", ascending=False), x="F1_test", y="Emotion", palette="viridis")
    plt.title("F1 (test) per emoție"); plt.xlim(0,1); plt.tight_layout()
    savefig(os.path.join(args.out_dir, "Fig6_05_f1_bars.png"))

    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=df_metrics.sort_values("AP_test", ascending=False), x="AP_test", y="Emotion", palette="mako")
    plt.title("Average Precision (test) per emoție"); plt.xlim(0,1); plt.tight_layout()
    savefig(os.path.join(args.out_dir, "Fig6_06_ap_bars.png"))

    # 2.10 Matrici de confuzie per emoție
    cols = 4
    rows_grid = int(np.ceil(len(EMO)/cols))
    fig, axes = plt.subplots(rows_grid, cols, figsize=(cols*4.2, rows_grid*3.6))
    axes = np.array(axes).reshape(-1)
    for i, emo in enumerate(EMO):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test[:, i], yhat_test[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(emo); axes[i].set_xlabel('Pred'); axes[i].set_ylabel('Real')
        axes[i].set_xticklabels(['0','1']); axes[i].set_yticklabels(['0','1'])
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.suptitle("Matrici de confuzie per emoție (test)", y=1.02)
    plt.tight_layout()
    savefig(os.path.join(args.out_dir, "Fig6_07_confusion_grids.png"))

    print(f"[OK] Gata. Vezi imaginile/fișierele în: {args.out_dir}")

if __name__ == "__main__":
    main()
