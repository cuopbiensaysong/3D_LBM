
train_csv_path = './fold_1_train.csv'

import pandas as pd

df = pd.read_csv(train_csv_path)

print(df.head())


print(df['DX_B'].value_counts())

print('9' * 100)

import numpy as np

def subsample_labels(df, label_col='label', labels_to_limit=['CN', 'MCI'], max_per_label=40, random_state=42):
    np.random.seed(random_state)
    keep_indices = []
    
    # For specified labels, keep at most max_per_label rows
    for lbl in labels_to_limit:
        label_indices = df[df[label_col] == lbl].index.tolist()
        if len(label_indices) > max_per_label:
            selected = np.random.choice(label_indices, size=max_per_label, replace=False).tolist()
        else:
            selected = label_indices
        keep_indices.extend(selected)
    
    # For other labels, keep all rows
    other_indices = df[~df[label_col].isin(labels_to_limit)].index.tolist()
    keep_indices.extend(other_indices)
    
    # Sort indices to maintain original order
    keep_indices = sorted(set(keep_indices))
    return df.loc[keep_indices].reset_index(drop=True)


df_subsampled = subsample_labels(df, label_col= 'DX_B', labels_to_limit=['CN', 'MCI'], max_per_label=36)


print(df_subsampled['DX_B'].value_counts())
# save the subsampled dataframe
df_subsampled.to_csv('fold_1_train_balanced.csv', index=False)



