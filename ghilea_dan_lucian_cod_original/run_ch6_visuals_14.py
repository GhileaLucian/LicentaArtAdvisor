# run_ch6_visuals_14.py
# Rulează vizualurile Cap. 6 pentru modelul cu 14 emoții antrenat prin train_thesis_modelV2.py
# Folosește make_ch6_plots.py cu importurile corecte și setări implicite pentru 14 emoții.

import sys
import subprocess

if __name__ == "__main__":
    cmd = [
        sys.executable, "make_ch6_plots.py",
        "--annotations_file", "data/metadata_curat.csv",
        "--train_dir", "data/train",
        "--val_dir", "data/validation",
        "--test_dir", "data/test",
        "--models_dir", "models",
        "--tb_root", "logs_final_definitiv",
        "--text_log", "log_final_definitiv.log",
        "--out_dir", "vizualizari_licenta/ch6v2_14emo",
        "--batch_size", "8",
        "--num_workers", "4",
        "--emotions",
        "Sadness","Trust","Fear","Disgust","Anger","Anticipation","Happiness",
        "Love","Surprise","Optimism","Gratitude","Pessimism","Regret","Agreeableness"
    ]
    print("Rulăm:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))
