
import os
import numpy as np
import pandas as pd

def convert_csv_to_npy(csv_file_path, npy_file_path, num_classes):
    data = []
    df = pd.read_csv(csv_file_path)

    if 'Scores' not in df.columns or 'True Label' not in df.columns:
        print(f"Error: Missing required columns in {csv_file_path}")
        return

    line_count = 0
    for _, row in df.iterrows():
        try:
            true_label = int(row['True Label'])
            scores = eval(row['Scores'])
            if len(scores) != num_classes:
                raise ValueError(f"Expected {num_classes} scores, but got {len(scores)}")
            data.append((np.array(scores), true_label))
            line_count += 1
        except ValueError as e:
            print(f"Error processing row: {row}")
            print(f"Error: {e}")
            continue

    data = np.array(data, dtype=object)
    np.save(npy_file_path, data)
    print(f"Converted {csv_file_path} to {npy_file_path} with {line_count} rows")

def process_all_csv_files(base_path, num_classes):
    files = os.listdir(base_path)
    csv_files = [f for f in files if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_file_path = os.path.join(base_path, csv_file)
        if csv_file.startswith('val_'):
            npy_file_name = 'validation_with_scores_' + csv_file.split('_')[1].replace('.csv', '')
        elif csv_file.startswith('test_'):
            npy_file_name = 'test_with_scores_' + csv_file.split('_')[1].replace('.csv', '')
        else:
            print(f"Skipping unrecognized file format: {csv_file}")
            continue

        npy_file_path = os.path.join(base_path, npy_file_name + '.npy')
        convert_csv_to_npy(csv_file_path, npy_file_path, num_classes)



# # Example usage for CIFAR-100
base_path_cifar100 = '/Users/nitinbisht/imb_20/raw_files'
process_all_csv_files(base_path_cifar100, num_classes=100)
