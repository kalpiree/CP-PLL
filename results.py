
import os
import numpy as np
import pandas as pd
import re

def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data


def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    train_labels = np.array(train_labels)

    if np.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif np.min(train_labels) == 1:
        train_labels -= 1

    K = int(np.max(train_labels) - np.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = np.zeros((n, K))
    partialY[np.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix = np.eye(K)
    transition_matrix[np.where(~np.eye(K, dtype=bool))] = p_1
    print('==> Transition Matrix:')
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = (random_n[j, :] < transition_matrix[train_labels[j], :]).astype(float)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY


def compute_partial_label_scores(scores, candidate_labels):
    y_score = np.max(scores[candidate_labels == 1])  # Use the maximum score from candidate labels
    S_PLL = np.sum(scores[scores >= y_score])  # Sum of scores greater than or equal to y_score
    return S_PLL


def compute_quantile(S_PLL_scores, epsilon):
    n = len(S_PLL_scores)
    quantile_index = int(np.ceil((n + 1) * (1 - epsilon)))
    Q_PLL = np.sort(S_PLL_scores)[quantile_index - 1]
    return Q_PLL


def generate_prediction_set(scores, candidate_labels, Q_PLL):
    y_score = np.max(scores[candidate_labels == 1])  # Use the maximum score from candidate labels
    filtered_indices = np.where(scores >= y_score)[0]  # Get the original indices of the filtered scores
    filtered_scores = scores[filtered_indices]  # Get the filtered scores based on the original indices

    sorted_indices = np.argsort(filtered_scores)  # Sort indices of the filtered scores in ascending order
    sorted_filtered_scores = filtered_scores[sorted_indices]
    sorted_filtered_indices = filtered_indices[
        sorted_indices]  # Get the sorted original indices based on the filtered scores

    # Compute cumulative sum starting with the highest scores included
    cumulative_sum = np.cumsum(sorted_filtered_scores[::-1])  # Start with all scores summed in descending order
    C_epsilon_PLL = list(sorted_filtered_indices[::-1])  # Reverse to match cumulative sum order

    while len(cumulative_sum) > 1 and cumulative_sum[-1] > Q_PLL:
        cumulative_sum = cumulative_sum[:-1]
        C_epsilon_PLL = C_epsilon_PLL[:-1]

    return C_epsilon_PLL


def compute_accuracy(test_data, prediction_sets):
    correct_predictions = 0
    total_predictions = len(test_data)

    for i, (scores, y_true) in enumerate(test_data):
        if y_true in prediction_sets[i]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy


def CP_PLL_algorithm(validation_data, test_data, epsilon, partial_labels):
    # Compute partial label scores for validation data
    S_PLL_scores = [compute_partial_label_scores(scores, partial_labels[i]) for i, (scores, y_true) in
                    enumerate(validation_data)]

    # Compute quantile value
    Q_PLL = compute_quantile(S_PLL_scores, epsilon)

    # Generate prediction set for test data
    prediction_sets = [generate_prediction_set(scores, partial_labels[i], Q_PLL) for i, (scores, y_true) in
                       enumerate(test_data)]

    # Compute accuracy
    accuracy = compute_accuracy(test_data, prediction_sets)

    avg_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])

    return Q_PLL, avg_set_size, accuracy


def process_datasets(base_path, epsilon, partial_rate=0.1):
    results = []
    files = os.listdir(base_path)
    validation_files = [f for f in files if 'validation' in f]
    test_files = [f for f in files if 'test' in f]

    for validation_file in validation_files:
        base_name = validation_file.replace('validation_with_scores_', '')
        corresponding_test_file = 'test_with_scores_' + base_name

        if corresponding_test_file in test_files:
            print(f"Processing {validation_file} and {corresponding_test_file}...")
            validation_data = load_data(os.path.join(base_path, validation_file))
            test_data = load_data(os.path.join(base_path, corresponding_test_file))

            # Extract partial_rate from file name
            partial_rate_match = re.search(r'(\d+(\.\d+)?)', base_name)
            if partial_rate_match:
                partial_rate = float(partial_rate_match.group(0))
            else:
                raise ValueError(f"Could not extract partial rate from file name: {base_name}")

            print(f"Using partial rate: {partial_rate}")

            # Generate partial labels
            train_labels = [data[1] for data in validation_data]
            partial_labels = generate_uniform_cv_candidate_labels(train_labels, partial_rate)
            print("Partial Labels", partial_labels)

            Q_PLL, avg_set_size, accuracy = CP_PLL_algorithm(validation_data, test_data, epsilon, partial_labels)

            results.append({
                'Test File': corresponding_test_file,
                'Quantile Value': Q_PLL,
                'Average Prediction Set Size': avg_set_size,
                'Accuracy': accuracy
            })

    # Write results to Excel file
    output_dir = os.path.join(base_path, 'evaluation_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_output = pd.DataFrame(results)
    output_file_path = os.path.join(output_dir, 'evaluation_results.xlsx')
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        df_output.to_excel(writer, index=False)

    print(f"All results saved to '{output_file_path}'.")


# Example usage
base_path = '/Users/ct-pll/table_cifar_100_lt'
epsilon = 0.1  # Example epsilon value
process_datasets(base_path, epsilon)
