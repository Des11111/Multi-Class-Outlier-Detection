import numpy as np
from statsmodels.stats.multitest import multipletests

def compute_standard_conformal_scores(X_train, Y_train, X_cal, Y_cal, X_test, classifier_callable=True):
    # Identify unique classes in the training labels
    unique_classes = np.unique(Y_train)

    # Dictionary to store classifiers for each class
    classifiers = {}

    # Step 1: Train a one-class classifier on each type of inliers separately
    for cls in unique_classes:
        # Select data points belonging to the current class
        X_train_cls = X_train[Y_train == cls]

        # Create a new instance of the classifier for each class
        clf = classifier_callable()

        # Fit the classifier on the current class data
        clf.fit(X_train_cls)

        # Store the trained classifier
        classifiers[cls] = clf

    # Step 2: Computing conformal scores for X_cal and X_test
    scores_cal = np.zeros((X_cal.shape[0], len(unique_classes)))
    scores_test = np.zeros((X_test.shape[0], len(unique_classes)))

    for idx, cls in enumerate(unique_classes):
        clf = classifiers[cls]
        scores_cal[:, idx] = clf.decision_function(X_cal)
        scores_test[:, idx] = clf.decision_function(X_test)

    return scores_cal, scores_test

def compute_MAMCOD_conformal_pv(K, n_in_cal, scores_cal, scores_test, is_high_score_inlier = True):
    # Calculate cumulative sums of n_in_cal for indexing
    cum_n_in_cal = np.cumsum(np.concatenate(([0], n_in_cal)))

    # Initialize the pv_test matrix
    pv_test = np.zeros((scores_test.shape[0], K))

    if is_high_score_inlier:
    # Compute conformal p-values
        for k in range(K):
            # Get the specific range from scores_cal
            cal_scores_range = scores_cal[cum_n_in_cal[k]:cum_n_in_cal[k + 1], k]

            # Use broadcasting and vectorized operations to compute p-values
            pv_test[:, k] = (np.sum(cal_scores_range <= scores_test[:, k].reshape(-1, 1), axis=1) + 1) / (
                        cal_scores_range.size + 1)
    else:
        for k in range(K):
            # Get the specific range from scores_cal
            cal_scores_range = scores_cal[cum_n_in_cal[k]:cum_n_in_cal[k + 1], k]

            # Use broadcasting and vectorized operations to compute p-values
            pv_test[:, k] = (np.sum(cal_scores_range > scores_test[:, k].reshape(-1, 1), axis=1) + 1) / (
                    cal_scores_range.size + 1)

    # Return the maximum of each row of pv_test
    max_pv_test = np.max(pv_test, axis=1)

    return max_pv_test

def compute_fdr_power(MAMCOD_pv, Y_test, alpha=0.10):
    MAMCOD_pv = np.array(MAMCOD_pv)
    Y_test = np.array(Y_test)

    # Apply BH procedure
    reject, pvals_corrected, _, _ = multipletests(MAMCOD_pv, alpha=alpha, method='fdr_bh')

    # Calculate FDR and Power
    true_positives = (reject & (Y_test == 0)).sum()
    false_positives = (reject & (Y_test != 0)).sum()
    total_positives = reject.sum()
    total_outliers = (Y_test == 0).sum()

    fdr = false_positives / total_positives if total_positives > 0 else 0
    power = true_positives / total_outliers if total_outliers > 0 else 0

    return fdr, power