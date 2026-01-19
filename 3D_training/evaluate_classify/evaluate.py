import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from model import get_densenet121

def predict_patient(patient_dir, model, device='cuda'):
    outputs = []
    for npy_file in os.listdir(patient_dir):
        if npy_file.endswith('.npy'):
            npy_path = os.path.join(patient_dir, npy_file)
            npy_data = np.load(npy_path)

            torch_data = torch.from_numpy(npy_data).float().to(device)
            output = model(torch_data)
            # output = torch.argmax(output, dim=1)
            outputs.append(output.detach().cpu().numpy())

    # Convert outputs to predicted labels and use majority voting
    if len(outputs) == 0:
        return None
    
    outputs = np.concatenate(outputs, axis=0)  # Shape: (num_samples, 2)
    predicted_labels = np.argmax(outputs, axis=1)  # Get predicted class for each sample
    
    # Return the label with the highest number of votes
    votes = np.bincount(predicted_labels, minlength=2)
    return int(np.argmax(votes))


map_labels = {'CN': 0, 'AD': 1}
inference_dir = ''
test_df = pd.read_csv('./test_label_CN_pMCI.csv')
ckpt_path = './Classification-models/from_scratch_DenseNet121_train+val.pth'
device = 'cuda'


model = get_densenet121(ckpt_path, device)
test_df['int_label'] = test_df['DX_B'].map(map_labels)

for index, row in tqdm(test_df.iterrows()):
    patient_dir = os.path.join(inference_dir, row['ID'])
    output = predict_patient(patient_dir)
    test_df.at[index, 'prediction'] = output

test_df['prediction'] = test_df['prediction'].astype(int)
valid_df = test_df.dropna(subset=['prediction', 'int_label'])
y_true = valid_df['int_label'].values
y_pred = valid_df['prediction'].astype(int).values

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)  # Sensitivity
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

# Confusion matrix: [[TN, FP], [FN, TP]]
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Specificity = TN / (TN + FP)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Print results
print("=" * 50)
print("Evaluation Results")
print("=" * 50)
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1 Score:    {f1:.4f}")
print(f"ROC AUC:     {roc_auc:.4f}")
print(f"Specificity: {specificity:.4f}")
print("\nConfusion Matrix:")
print(f"  TN: {tn}  FP: {fp}")
print(f"  FN: {fn}  TP: {tp}")
print("=" * 50)
