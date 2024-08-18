import os
import numpy as np


def convert_txt_to_npy(txt_file_path, npy_file_path):
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
                if len(scores) != 100:
                    raise ValueError(f"Expected 100 scores, but got {len(scores)}")
                data.append((np.array(scores), true_label))
                line_count += 1
            except ValueError as e:
                print(f"Error processing line: {line}")
                print(f"Error: {e}")
                continue
    data = np.array(data, dtype=object)
    np.save(npy_file_path, data)
    print(f"Converted {txt_file_path} to {npy_file_path} with {line_count} lines")


def process_all_txt_files(base_path):
    files = os.listdir(base_path)
    txt_files = [f for f in files if f.endswith('.txt')]

    for txt_file in txt_files:
        txt_file_path = os.path.join(base_path, txt_file)
        npy_file_path = os.path.join(base_path, txt_file.replace('.txt', '.npy'))
        convert_txt_to_npy(txt_file_path, npy_file_path)




# Example usage
base_path = '/Users/cifar_100/folder_'
process_all_txt_files(base_path)
