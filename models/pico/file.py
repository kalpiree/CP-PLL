##import pandas as pd
##
### Replace 'your_file.csv' with the path to your actual CSV file
##file_path = ''
##
### Read the CSV file
##data = pd.read_csv(file_path)
##
### Get the number of rows
##num_rows = len(data)
##
### Print the number of rows
##print(f"The number of rows in the file is: {num_rows}")

import pandas as pd

# The name of your CSV file
file_path = 'final_val_results.csv'

# Read the CSV file
data = pd.read_csv(file_path)

# Function to print True Label values for specified rows
def print_true_labels(data, start, step, num_lines):
    for i in range(num_lines):
        index = start + i * step
        if index < len(data):
            print(f"True Label at row {index}: {data.at[index, 'True Label']}")
        else:
            print(f"Index {index} is out of bounds for the dataframe with {len(data)} rows")

# Print the True Label for the 0th row and the 10000th row
print_true_labels(data, 0, 10000, 10)
