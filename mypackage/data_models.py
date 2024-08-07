import numpy as np
import pandas as pd

def generate_data_uniform_plus_normal(K, n_in_inliers, n_out, dim, means, radius, a_signal):
    """
    Generates synthetic data with K classes of inliers and 1 class of outliers.
    """
    n = np.sum(n_in_inliers) + n_out
    assert (n > 0)
    assert len(n_in_inliers) == K, "Length of n_in_inliers should be equal to K"
    assert len(means) == K, "Length of means should be equal to K"

    X = np.zeros((0, dim))
    Y = np.zeros((0,))

    for i in range(K):
        if n_in_inliers[i] > 0:
            inliers = np.random.uniform(low=means[i] - radius, high=means[i] + radius, size=(n_in_inliers[i], dim)) \
                      + np.random.normal(loc=0, scale=1, size=(n_in_inliers[i], dim))
            X = np.concatenate([X, inliers], 0)
            Y = np.concatenate([Y, (i + 1) * np.ones((n_in_inliers[i],))], 0)

    if n_out > 0:
        outliers = []
        for i in range(K):
            outlier = np.random.uniform(low=means[i] - radius, high=means[i] + radius, size=(n_out, dim)) \
                      + a_signal * np.random.normal(loc=0, scale=1, size=(n_out, dim))
            outliers.append(outlier)

        outliers = np.vstack(outliers)
        np.random.shuffle(outliers)
        outliers = outliers[:n_out]
        X = np.concatenate([X, outliers], 0)
        Y = np.concatenate([Y, np.zeros((n_out,))], 0)

    return X, Y.astype(int)


def sample_from_real_data(data_source, in_class_labels, n_in, n_out=0):
    """
    Sample data from a NumPy array based on in_class_labels, ensuring no intersection and sampling outliers
    from data that does not belong to in_class_labels. Also return the remaining data that was not sampled.

    Parameters:
    - data_source (np.ndarray): The array containing the dataset where the first column is labels.
    - in_class_labels (list): List of class labels to sample from.
    - n_in (list): Number of inliers to sample for each class.
    - n_out (int): Number of outliers to sample. If 0, no outliers are sampled.

    Returns:
    - tuple: (X, Y, remaining_data) where X is the sampled data without the 'label' column, Y is the corresponding labels,
      and remaining_data is the data that was not sampled.
    """
    # Separate labels and features
    labels = data_source[:, 0]
    features = data_source[:, 1:]

    # Initialize lists to store sampled data and labels
    sampled_features = []
    sampled_labels = []

    # Create a mask to keep track of sampled indices
    sampled_indices = np.zeros(len(data_source), dtype=bool)

    # Create a mapping for new labels
    label_mapping = {label: idx + 1 for idx, label in enumerate(in_class_labels)}

    i = 0
    for label in in_class_labels:
        class_indices = np.where(labels == label)[0]
        # Sample inliers without replacement
        sampled_class_indices = np.random.choice(class_indices, size=n_in[i], replace=False)
        sampled_features.append(features[sampled_class_indices])
        sampled_labels.append(np.full(n_in[i], label_mapping[label]))  # Map old label to new label

        # Mark these indices as sampled
        sampled_indices[sampled_class_indices] = True
        i += 1

    # Concatenate all inliers into one array
    X = np.vstack(sampled_features)
    Y = np.concatenate(sampled_labels)

    # Determine the remaining data after sampling inliers
    remaining_indices = ~sampled_indices

    if n_out > 0:
        # Get remaining data that does not belong to in_class_labels
        remaining_data = data_source[remaining_indices]
        remaining_labels = remaining_data[:, 0]
        remaining_features = remaining_data[:, 1:]

        # Filter remaining data to exclude in_class_labels
        remaining_data_excluding_labels = remaining_data[~np.isin(remaining_labels, in_class_labels)]

        # Sample outliers from the remaining data
        outlier_indices = np.random.choice(len(remaining_data_excluding_labels), size=n_out, replace=False)
        outliers = remaining_data_excluding_labels[outlier_indices]

        # Assign outlier labels as 0
        X_outliers = outliers[:, 1:]
        Y_outliers = np.zeros(len(outliers), dtype=int)  # All outliers are labeled as 0

        # Combine inliers and outliers
        X = np.vstack([X, X_outliers])
        Y = np.concatenate([Y, Y_outliers])

        # Update remaining data after sampling outliers
        remaining_indices &= ~np.isin(np.arange(len(data_source)), outliers[:, 0])

    # Final remaining data
    remaining_data = data_source[remaining_indices]

    return X, Y, remaining_data