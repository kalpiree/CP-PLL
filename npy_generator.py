
import os
import numpy as np
import pandas as pd
import argparse


def process_txt_file(txt_file_path, num_classes):
    data = []
    line_count = 0
    with open(txt_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                true_label, scores_str = line.split(',', 1)
                true_label = int(true_label.strip())
                scores = eval(scores_str.strip())
                if len(scores) != num_classes:
                    raise ValueError(f"Expected {num_classes} scores, but got {len(scores)}")
                data.append((np.array(scores), true_label))
                line_count += 1
            except ValueError as e:
                print(f"Error processing line: {line}")
                print(f"Error: {e}")
                continue
    return data, line_count


def process_csv_file(csv_file_path, num_classes):
    data = []
    df = pd.read_csv(csv_file_path)
    columns = list(df.columns)

    # Check for the two types of CSV formats
    if 'Scores' in columns and 'True Label' in columns:
        # Format 1: 'Scores' and 'True Label' columns
        for _, row in df.iterrows():
            try:
                true_label = int(row['True Label'])
                scores = eval(row['Scores'])
                if len(scores) != num_classes:
                    raise ValueError(f"Expected {num_classes} scores, but got {len(scores)}")
                data.append((np.array(scores), true_label))
            except ValueError as e:
                print(f"Error processing row: {row}")
                print(f"Error: {e}")
                continue
    elif 'label' in columns and all(col.isdigit() for col in columns if col != 'label'):
        # Format 2: Columns are '0', '1', ..., 'num_classes-1' and 'label'
        for _, row in df.iterrows():
            try:
                true_label = int(row['label'])
                scores = row[columns[:-1]].values.astype(float)
                if len(scores) != num_classes:
                    raise ValueError(f"Expected {num_classes} scores, but got {len(scores)}")
                data.append((np.array(scores), true_label))
            except ValueError as e:
                print(f"Error processing row: {row}")
                print(f"Error: {e}")
                continue
    else:
        print(f"Error: Unrecognized format in {csv_file_path}")
        print(f"Found columns: {columns}")
        return data, 0

    return data, len(data)


def convert_to_npy(file_path, npy_file_path, num_classes):
    if file_path.endswith('.txt'):
        data, line_count = process_txt_file(file_path, num_classes)
    elif file_path.endswith('.csv'):
        data, line_count = process_csv_file(file_path, num_classes)
    else:
        print(f"Unsupported file type: {file_path}")
        return

    if data:
        data = np.array(data, dtype=object)
        np.save(npy_file_path, data)
        print(f"Converted {file_path} to {npy_file_path} with {line_count} entries")
    else:
        print(f"No data converted for file: {file_path}")


def process_all_files(base_path, num_classes):
    files = os.listdir(base_path)
    for file_name in files:
        file_path = os.path.join(base_path, file_name)
        if file_name.endswith('.txt'):
            npy_file_path = os.path.join(base_path, file_name.replace('.txt', '.npy'))
        elif file_name.startswith('val_') or file_name.startswith('test_'):
            if file_name.startswith('val_'):
                npy_file_name = 'validation_with_scores_' + file_name.split('_')[1].replace('.csv', '')
            else:
                npy_file_name = 'test_with_scores_' + file_name.split('_')[1].replace('.csv', '')
            npy_file_path = os.path.join(base_path, npy_file_name + '.npy')
        else:
            print(f"Skipping unrecognized file format: {file_name}")
            continue

        convert_to_npy(file_path, npy_file_path, num_classes)


def main():
    parser = argparse.ArgumentParser(
        description='Convert various input files to .npy format for CIFAR-10 or CIFAR-100 datasets.')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], required=True,
                        help='Specify the dataset: "cifar10" or "cifar100".')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Specify the base path containing the files to process.')

    args = parser.parse_args()

    num_classes = 10 if args.dataset == 'cifar10' else 100

    process_all_files(args.base_path, num_classes)


if __name__ == '__main__':
    main()
