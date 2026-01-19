import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, balanced_accuracy_score, 
    matthews_corrcoef, cohen_kappa_score, average_precision_score
)

from model import get_densenet121

def predict_patient(patient_dir, model, device='cuda', log_txt_file=None):
    """
    Predicts patient class using Soft Voting (Averaging Probabilities).
    Returns: (predicted_label, probability_of_positive_class)
    """
    if log_txt_file is not None:
        with open(log_txt_file, 'a') as f:
            f.write(f"Predicting patient {patient_dir}\n")
    
    outputs_probs = []
    
    # 1. Collect probabilities for all chunks/slices
    for npy_file in os.listdir(patient_dir):
        if npy_file.endswith('.npy'):
            npy_path = os.path.join(patient_dir, npy_file)
            npy_data = np.load(npy_path)

            torch_data = torch.from_numpy(npy_data).float().to(device)
            
            with torch.no_grad():
                logits = model(torch_data)
                # Apply Softmax to get probabilities (Shape: 1, 2)
                probs = F.softmax(logits, dim=1) 
                
            out = probs.detach().cpu().numpy()
            outputs_probs.append(out)
            if log_txt_file is not None:
                with open(log_txt_file, 'a') as f:
                    f.write(f"Output for {npy_file}: {out}\n")

    if len(outputs_probs) == 0:
        return None, None
    
    # 2. Soft Voting: Average the probabilities across all samples
    outputs_probs = np.concatenate(outputs_probs, axis=0) # Shape: (num_samples, 2)
    avg_probs = np.mean(outputs_probs, axis=0)            # Shape: (2,)
    
    # 3. Determine label and positive class probability
    predicted_label = int(np.argmax(avg_probs))
    prob_positive = avg_probs[1] # Probability of class 1 (AD)

    if log_txt_file is not None:
        with open(log_txt_file, 'a') as f:
            f.write(f"Mean Probs: {avg_probs} | Pred: {predicted_label}\n")

    return predicted_label, prob_positive

# --- Setup ---
map_labels = {'CN': 0, 'AD': 1}
inference_dir = '/home/user01/aiotlab/htien/3D_LBM/3D_training/results/inferences/3D_LBM_uniform_noise0.005_step20'
test_df = pd.read_csv('./test_label_CN_pMCI.csv')
ckpt_path = './Classification-models/from_scratch_DenseNet121_train+val.pth'
device = 'cuda'
log_txt_file = '/home/user01/aiotlab/htien/3D_LBM/3D_training/results/inferences/3D_LBM_uniform_noise0.005_step20/classification_log.txt'

# Load Model
model = get_densenet121(ckpt_path, device)
model.eval() # Ensure model is in eval mode

test_df['int_label'] = test_df['DX_B'].map(map_labels)

# --- Inference Loop ---
predictions = []
probabilities = []
ground_truths = []

print("Starting Inference...")
for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    patient_dir = os.path.join(inference_dir, row['ID'])
    
    # Check if dir exists before predicting
    if os.path.exists(patient_dir):
        pred_label, prob_pos = predict_patient(patient_dir, model, device, log_txt_file)
        
        if pred_label is not None:
            predictions.append(pred_label)
            probabilities.append(prob_pos)
            ground_truths.append(row['int_label'])

# Convert to arrays
y_true = np.array(ground_truths)
y_pred = np.array(predictions)
y_prob = np.array(probabilities)

# --- Metric Calculations ---

# Basic Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
f1 = f1_score(y_true, y_pred, zero_division=0)

# Advanced Medical Metrics
# Specificity & NPV requires Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0 # Negative Predictive Value

# Balanced Accuracy (Arithmetic mean of Sensitivity and Specificity)
balanced_acc = balanced_accuracy_score(y_true, y_pred)

# Matthews Correlation Coefficient (Better than F1 for imbalanced datasets)
mcc = matthews_corrcoef(y_true, y_pred)

# Cohen's Kappa (Agreement relative to chance)
kappa = cohen_kappa_score(y_true, y_pred)

# ROC AUC and PR AUC (Using Probabilities, NOT labels)
try:
    roc_auc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
except ValueError:
    roc_auc = 0.0
    auprc = 0.0

# --- Print Results ---
print("\n" + "=" * 60)
print(f"{'MEDICAL CLASSIFICATION REPORT':^60}")
print("=" * 60)
print(f"{'Metric':<30} | {'Value':<10}")
print("-" * 60)
print(f"{'Accuracy':<30} | {accuracy:.4f}")
print(f"{'Balanced Accuracy':<30} | {balanced_acc:.4f}")
print("-" * 60)
print(f"{'Sensitivity (Recall)':<30} | {recall:.4f}")
print(f"{'Specificity':<30} | {specificity:.4f}")
print(f"{'Precision (PPV)':<30} | {precision:.4f}")
print(f"{'NPV (Neg. Pred. Value)':<30} | {npv:.4f}")
print("-" * 60)
print(f"{'F1 Score':<30} | {f1:.4f}")
print(f"{'MCC (Matthews)':<30} | {mcc:.4f}")
print(f"{'Cohen Kappa':<30} | {kappa:.4f}")
print("-" * 60)
print(f"{'ROC AUC':<30} | {roc_auc:.4f}")
print(f"{'PR AUC (AUPRC)':<30} | {auprc:.4f}")
print("=" * 60)
print("\nConfusion Matrix:")
print(f"      Pred 0 (CN)   Pred 1 (AD)")
print(f"True 0   {tn:<10}    {fp:<10}")
print(f"True 1   {fn:<10}    {tp:<10}")
print("=" * 60)

# --- Log to File ---
with open(log_txt_file, 'a') as f:
    f.write("\n" + "=" * 60 + "\n")
    f.write(f"FINAL EVALUATION RESULTS\n")
    f.write("=" * 60 + "\n")
    f.write(f"Accuracy:          {accuracy:.4f}\n")
    f.write(f"Balanced Acc:      {balanced_acc:.4f}\n")
    f.write(f"Sensitivity:       {recall:.4f}\n")
    f.write(f"Specificity:       {specificity:.4f}\n")
    f.write(f"PPV (Precision):   {precision:.4f}\n")
    f.write(f"NPV:               {npv:.4f}\n")
    f.write(f"F1 Score:          {f1:.4f}\n")
    f.write(f"MCC:               {mcc:.4f}\n")
    f.write(f"ROC AUC:           {roc_auc:.4f}\n")
    f.write(f"PR AUC:            {auprc:.4f}\n")
    f.write(f"Confusion Matrix:  TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")
    f.write("=" * 60 + "\n")