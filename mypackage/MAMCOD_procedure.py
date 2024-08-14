import numpy as np
from statsmodels.stats.multitest import multipletests
import copy
from sklearn.preprocessing import StandardScaler

def train_occ(K, X_train, Y_train, oneclass_classifier):
    # In some cases X_train has to be scaled
    occ_list = [copy.deepcopy(oneclass_classifier) for _ in range(K)]
    for i in range(K):
        X_train_i = X_train[Y_train == (i + 1)]
        if len(X_train_i) > 0:
            occ_list[i].fit(X_train_i)
        else:
            occ_list[i] = None
    return occ_list

def scale_data(X_train, X_external):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_external_scaled = scaler.transform(X_external)
    return X_external_scaled

def compute_standard_conformal_scores(K, X_train, Y_train, X_cal, Y_cal, X_test, oneclass_classifier):

    # Scaling
    X_train_scaled = scale_data(X_train, X_train)
    X_cal_scaled = scale_data(X_train, X_cal)
    X_test_scaled = scale_data(X_train, X_test)

    # Step 1: Train a one-class classifier on each type of inliers separately
    occ_list = train_occ(K, X_train_scaled, Y_train, oneclass_classifier)

    # Step 2: Computing conformal scores for X_cal and X_test
    scores_cal = np.zeros((X_cal_scaled.shape[0], K))
    scores_test = np.zeros((X_test_scaled.shape[0], K))

    for idx in range(K):
        scores_cal[:, idx] =  occ_list[idx].score_samples(X_cal_scaled)
        scores_test[:, idx] = occ_list[idx].score_samples(X_test_scaled)

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