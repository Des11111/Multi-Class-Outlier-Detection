import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.mixture import GaussianMixture
import itertools

def compute_pu_scores(K, X_train, Y_train, X_cal, Y_cal, X_test, binary_classifier, occ_list = None, multi_step=True):

    X_unlabeled = np.vstack((X_cal, X_test))

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled = scaler.transform(X_cal)
    X_test_scaled = scaler.transform(X_test)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)

    if multi_step:
        assert occ_list is not None, "One-class classifier must be provided for two-step or multi-step method"

        # Step 1: Apply the one-class classifiers to the unlabeled data
        pred_unlabeled_list = []
        for occ in occ_list:
            if occ is not None:
                pred_unlabeled_list.append(occ.predict(X_unlabeled_scaled))
            else:
                # For absent classifiers, assume all predictions are -1
                pred_unlabeled_list.append(np.full(X_unlabeled_scaled.shape[0], -1))

        # Identify reliable negatives
        pred_unlabeled_array = np.array(pred_unlabeled_list)
        reliable_negatives_idx = np.where(np.all(pred_unlabeled_array == -1, axis=0))[0]
        X_reliable_negatives = X_unlabeled_scaled[reliable_negatives_idx]

        # Combine the positive samples with reliable negatives
        X_combined = np.vstack((X_train_scaled, X_reliable_negatives))
        y_combined = np.hstack((np.zeros(len(X_train_scaled)), np.ones(len(X_reliable_negatives))))
    else:
        # Combine the positive samples with the mixed samples
        X_combined = np.vstack((X_train_scaled, X_unlabeled_scaled))
        y_combined = np.hstack((np.zeros(len(X_train_scaled)), np.ones(len(X_unlabeled_scaled))))

    # Step 2: Train a binary classifier on the selected positives and reliable negatives
    binary_classifier.fit(X_combined, y_combined)

    # Predict probabilities
    scores_cal = binary_classifier.predict_proba(X_cal_scaled)[:, 1]
    scores_test = binary_classifier.predict_proba(X_test_scaled)[:, 1]

    return scores_cal, scores_test, X_reliable_negatives if multi_step else None

def prepare_pu_score_matrices(K, n_in_cal, n_test, scores_cal, scores_test):
    # 1. Assertions to check the lengths of scores_cal and scores_test
    assert len(scores_cal) == sum(n_in_cal), f"scores_cal length is {len(scores_cal)}, expected {sum(n_in_cal)}"
    assert len(scores_test) == n_test, f"scores_test length is {len(scores_test)}, expected {n_test}"

    # 2. Transform scores_cal into a n_in_cal * K matrix
    scores_cal_mat = np.zeros((sum(n_in_cal), K))
    cum_n_in_cal = np.cumsum([0] + n_in_cal)  # Cumulative sum to get start and end indices for each class
    for k in range(K):
        start = cum_n_in_cal[k]
        end = cum_n_in_cal[k + 1]
        scores_cal_mat[start:end, k] = scores_cal[start:end]

    # 3. Transform scores_test into a n_test * K matrix
    scores_test_mat = np.tile(scores_test.reshape(-1, 1), K)

    # 4. Return scores_cal_mat and scores_test_mat
    return scores_cal_mat, scores_test_mat

def combine_proportions(prop_in_1_train, prop_in_1_cal, prop_in_1_test, prop_out):
    proportion_combinations = []
    for p1_train in prop_in_1_train:
        for p1_cal in prop_in_1_cal:
            for p1_test in prop_in_1_test:

                p2_train = 1 - p1_train
                p2_cal = 1 - p1_cal
                p2_test = (1 - prop_out) * (1 - p1_test)
                p1_test = (1 - prop_out) * p1_test

                proportion_combinations.append((p1_train, p2_train, p1_cal, p2_cal, p1_test, p2_test, prop_out))

    return proportion_combinations

# Calculate the proportions within the predicted outliers
def calculate_proportions_within_predictions(Y_test, outlier_predictions):
    total_outliers = np.sum(outlier_predictions == 0)
    if total_outliers == 0:
        return {'type_1_inliers': 0, 'type_2_inliers': 0, 'true_outliers': 0}

    type_1_inliers = np.sum((Y_test == 1) & (outlier_predictions == 0)) / total_outliers
    type_2_inliers = np.sum((Y_test == 2) & (outlier_predictions == 0)) / total_outliers
    true_outliers = np.sum((Y_test == 0) & (outlier_predictions == 0)) / total_outliers

    return {
        'type_1_inliers': type_1_inliers,
        'type_2_inliers': type_2_inliers,
        'true_outliers': true_outliers
    }