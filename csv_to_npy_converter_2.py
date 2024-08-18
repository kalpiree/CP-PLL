# import os
# import numpy as np
# import pandas as pd
#
#
# def convert_csv_to_npy_format2(csv_file_path, npy_file_path, num_classes):
#     # Read the CSV file as comma-separated, skip the first row
#     df = pd.read_csv(csv_file_path, delimiter=',')
#
#     # The columns should match what we expect: 0, 1, ..., num_classes-1, label
#     expected_columns = [str(i) for i in range(num_classes)] + ['label']
#
#     if list(df.columns) != expected_columns:
#         print(f"Error: Column names do not match expected names in {csv_file_path}")
#         print(f"Found columns: {list(df.columns)}")
#         return
#
#     data = []
#     for _, row in df.iterrows():
#         try:
#             true_label = int(row['label'])
#             scores = row[expected_columns[:-1]].values  # Exclude the label column
#             if len(scores) != num_classes:
#                 raise ValueError(f"Expected {num_classes} scores, but got {len(scores)}")
#             data.append((np.array(scores), true_label))
#         except ValueError as e:
#             print(f"Error processing row: {row}")
#             print(f"Error: {e}")
#             continue
#
#     data = np.array(data, dtype=object)
#     np.save(npy_file_path, data)
#     print(f"Converted {csv_file_path} to {npy_file_path} with {len(data)} rows")
#
#
# def process_all_csv_files(base_path, num_classes):
#     files = os.listdir(base_path)
#     csv_files = [f for f in files if f.endswith('.csv')]
#
#     for csv_file in csv_files:
#         csv_file_path = os.path.join(base_path, csv_file)
#         if csv_file.startswith('val_'):
#             npy_file_name = 'validation_with_scores_' + csv_file.split('_')[1].replace('.csv', '')
#         elif csv_file.startswith('test_'):
#             npy_file_name = 'test_with_scores_' + csv_file.split('_')[1].replace('.csv', '')
#         else:
#             print(f"Skipping unrecognized file format: {csv_file}")
#             continue
#
#         npy_file_path = os.path.join(base_path, npy_file_name + '.npy')
#         convert_csv_to_npy_format2(csv_file_path, npy_file_path, num_classes)
#
#
# # Example usage
# base_path_cifar10 = '/Users/nitinbisht/PycharmProjects/ct-pll/exp_alice/records/unbalanced/imbalance_cifar10/imbalance_cifar_10/at_200'
# process_all_csv_files(base_path_cifar10,
#                       num_classes=10)  # Adjust num_classes according to your dataset (10 for CIFAR-10, 100 for CIFAR-100)

import os
import numpy as np
import pandas as pd


def convert_csv_to_npy_format2(csv_file_path, npy_file_path, num_classes):
    # Read the CSV file as comma-separated
    df = pd.read_csv(csv_file_path, delimiter=',')

    # Adjust the expected columns to match the new header format
    expected_columns = ['True Label'] + [f'Label_{i}' for i in range(num_classes)]

    if list(df.columns) != expected_columns:
        print(f"Error: Column names do not match expected names in {csv_file_path}")
        print(f"Found columns: {list(df.columns)}")
        return

    data = []
    for _, row in df.iterrows():
        try:
            true_label = int(row['True Label'])
            scores = row[expected_columns[1:]].values  # Get all the Label_ columns
            if len(scores) != num_classes:
                raise ValueError(f"Expected {num_classes} scores, but got {len(scores)}")
            data.append((np.array(scores), true_label))
        except ValueError as e:
            print(f"Error processing row: {row}")
            print(f"Error: {e}")
            continue

    data = np.array(data, dtype=object)
    np.save(npy_file_path, data)
    print(f"Converted {csv_file_path} to {npy_file_path} with {len(data)} rows")


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
        convert_csv_to_npy_format2(csv_file_path, npy_file_path, num_classes)


# Example usage
base_path_cifar10 = '/Users/nitinbisht/unbalanced_cifar_100/at_50'
process_all_csv_files(base_path_cifar10,
                      num_classes=100)  # Adjust num_classes according to your dataset (10 for CIFAR-10, 100 for CIFAR-100)
