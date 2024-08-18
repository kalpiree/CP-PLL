import os
import numpy as np
import pandas as pd


def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data


def compute_partial_label_scores(scores, candidate_labels):
    y_score = np.max(scores[candidate_labels == 1])
    S_PLL = np.sum(scores[scores >= y_score])
    return S_PLL


def compute_quantile(S_PLL_scores, epsilon):
    n = len(S_PLL_scores)
    quantile_index = int(np.ceil((n + 1) * (1 - epsilon)))
    Q_PLL = np.sort(S_PLL_scores)[quantile_index - 1]
    return Q_PLL


def generate_prediction_set(scores, candidate_labels, Q_PLL):
    y_score = np.max(scores[candidate_labels == 1])
    filtered_indices = np.where(scores >= y_score)[0]
    filtered_scores = scores[filtered_indices]
    sorted_indices = np.argsort(filtered_scores)
    sorted_filtered_scores = filtered_scores[sorted_indices]
    sorted_filtered_indices = filtered_indices[sorted_indices]

    cumulative_sum = np.cumsum(sorted_filtered_scores[::-1])
    C_epsilon_PLL = list(sorted_filtered_indices[::-1])

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
    S_PLL_scores = [compute_partial_label_scores(scores, partial_labels[i]) for i, (scores, y_true) in
                    enumerate(validation_data)]
    Q_PLL = compute_quantile(S_PLL_scores, epsilon)
    prediction_sets = [generate_prediction_set(scores, partial_labels[i], Q_PLL) for i, (scores, y_true) in
                       enumerate(test_data)]
    avg_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])
    return avg_set_size


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

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):
        partialY[j, :] = (random_n[j, :] < transition_matrix[train_labels[j], :]).astype(float)

    return partialY


def process_datasets(base_path, epsilons, models, partial_rate=0.1):
    results = {epsilon: [] for epsilon in epsilons}

    for model in models:
        validation_file = f'validation_with_scores_{partial_rate}_{model}.npy'
        test_file = f'test_with_scores_{partial_rate}_{model}.npy'

        validation_data = load_data(os.path.join(base_path, validation_file))
        test_data = load_data(os.path.join(base_path, test_file))

        train_labels = [data[1] for data in validation_data]
        partial_labels = generate_uniform_cv_candidate_labels(train_labels, partial_rate)

        for epsilon in epsilons:
            avg_set_size = CP_PLL_algorithm(validation_data, test_data, epsilon, partial_labels)
            results[epsilon].append(avg_set_size)

    df_output = pd.DataFrame(results, index=models)
    df_output = df_output.T  # Transpose to have epsilons as rows and models as columns

    output_dir = os.path.join(base_path, 'evaluation_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_path = os.path.join(output_dir, 'average_prediction_set_size.xlsx')
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        df_output.to_excel(writer, index=True)

    print(f"Results saved to '{output_file_path}'.")


# Example usage
base_path = '/Users/ct-pll/table_cifar_100_lt'
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
models = ['proden', 'htc', 'pico', 'records', 'solar']

process_datasets(base_path, epsilons, models)
