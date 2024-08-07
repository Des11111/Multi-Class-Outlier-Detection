import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
import torch
import pomegranate
from pomegranate.distributions import *
from pomegranate.gmm import GeneralMixtureModel

# Set random seed and print options
np.set_printoptions(suppress=True)

def compute_mean_distance(K,scores_test, Y_test):
    outliers = scores_test[Y_test == 0]
    min_mean_distance = float('inf')
    for k in range(K):
        inliers = scores_test[Y_test == k+1]
        mean_distance = np.abs(outliers[:, None] - inliers).mean()
        if mean_distance < min_mean_distance:
            min_mean_distance = mean_distance
    min_mean_distance = min_mean_distance - np.abs(1 - outliers[:, None]).mean()
    return min_mean_distance


import numpy as np

def compute_proportion_error(proportions, estimated_proportions, metric='mse'):
    """
    Compute various estimation errors between true and estimated proportions.

    Parameters:
    - proportions (array-like): True proportions.
    - estimated_proportions (array-like): Estimated proportions.
    - metric (str): The error metric to use ('mae', 'mse', 'rmse', 'mape', 'smape').

    Returns:
    - float: Computed error based on the selected metric.
    """
    proportions = np.array(proportions)
    estimated_proportions = np.array(estimated_proportions)

    if metric == 'mae':
        error = np.mean(np.abs(proportions - estimated_proportions))
    elif metric == 'mse':
        error = np.mean((proportions - estimated_proportions) ** 2)
    elif metric == 'rmse':
        mse = np.mean((proportions - estimated_proportions) ** 2)
        error = np.sqrt(mse)
    elif metric == 'mape':
        error = np.mean(np.abs((proportions - estimated_proportions) / proportions)) * 100
    elif metric == 'smape':
        error = np.mean(np.abs(proportions - estimated_proportions) / (
                    (np.abs(proportions) + np.abs(estimated_proportions)) / 2)) * 100
    else:
        raise ValueError("Unknown metric. Choose from 'mae', 'mse', 'rmse', 'mape', 'smape'.")

    return error

def subsample_training_data(X_train, Y_train, target_proportions):
    available_populations = [np.sum(Y_train == (i + 1)) for i in range(len(target_proportions))]
    target_sizes = [int(target_proportions[i] * len(Y_train)) for i in range(len(target_proportions))]

    # Scale down target sizes proportionally if any target size exceeds the available population
    scaling_factor = min(1.0, min(available_populations[i] / target_sizes[i] for i in range(len(target_sizes)) if target_sizes[i] > 0))
    scaled_target_sizes = [int(scaling_factor * target_sizes[i]) for i in range(len(target_sizes))]

    subsampled_X_train = []
    subsampled_Y_train = []

    for i in range(len(scaled_target_sizes)):
        inlier_indices = np.where(Y_train == (i + 1))[0]
        subsampled_indices = np.random.choice(inlier_indices, scaled_target_sizes[i], replace=False)
        subsampled_X_train.append(X_train[subsampled_indices])
        subsampled_Y_train.append(Y_train[subsampled_indices])

    return np.vstack(subsampled_X_train), np.hstack(subsampled_Y_train)


def bootstrap_training_data(X_train, Y_train, target_proportions):
    bootstrapped_X_train = []
    bootstrapped_Y_train = []

    for i in range(len(target_proportions)):
        inlier_indices = np.where(Y_train == (i + 1))[0]
        target_size = int(max(0,target_proportions[i] * len(Y_train)))
        bootstrapped_indices = np.random.choice(inlier_indices, target_size, replace=True)
        bootstrapped_X_train.append(X_train[bootstrapped_indices])
        bootstrapped_Y_train.append(Y_train[bootstrapped_indices])

    return np.vstack(bootstrapped_X_train), np.hstack(bootstrapped_Y_train)

def estimate_test_proportions(X_train, Y_train, X_unlabeled, oneclass_classifier, K):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)

    occ_list = [copy.deepcopy(oneclass_classifier) for _ in range(K)]
    for i in range(K):
        X_train_i = X_train_scaled[Y_train == (i + 1)]
        if len(X_train_i) > 0:
            occ_list[i].fit(X_train_i)
        else:
            occ_list[i] = None  # Mark as None if no data to fit

    # Apply the one-class classifiers to the unlabeled data
    pred_unlabeled_list = []
    for occ in occ_list:
        if occ is not None:
            pred_unlabeled_list.append(occ.predict(X_unlabeled_scaled))
        else:
            # For absent classifiers, assume all predictions are -1
            pred_unlabeled_list.append(np.full(X_unlabeled_scaled.shape[0], -1))

    # Estimate proportions of inliers in the test set
    estimated_proportions = np.mean(np.array(pred_unlabeled_list) == 1, axis=1)
    return estimated_proportions, occ_list

def adjust_proportions(X_train, Y_train, X_cal, X_test, oneclass_classifier, K, method='subsample'):
    X_unlabeled = np.vstack((X_cal, X_test))

    # Estimate test proportions
    estimated_proportions, trained_occ_list = estimate_test_proportions(X_train, Y_train, X_unlabeled, oneclass_classifier, K)

    if method == 'subsample':
        X_train_adjusted, Y_train_adjusted = subsample_training_data(X_train, Y_train,
                                                                            estimated_proportions)
    elif method == 'bootstrap':
        X_train_adjusted, Y_train_adjusted = bootstrap_training_data(X_train, Y_train,
                                                                            estimated_proportions)
    else:
        raise ValueError("Method should be either 'subsample' or 'bootstrap'")

    return X_train_adjusted, Y_train_adjusted, trained_occ_list

def adjust_proportions_mixture_model(K, X_train, Y_train, X_cal, X_test, oneclass_classifier, sample_method='subsample', mixture_distributions = 'normal'):
    X_unlabeled = np.vstack((X_cal, X_test))

    # Estimate unlabeled proportions
    estimated_proportions, trained_occ_list = estimate_unlabeled_proportions(K, X_train, Y_train, X_unlabeled, oneclass_classifier, mixture_distributions)

    if sample_method == 'subsample':
        X_train_adjusted, Y_train_adjusted = subsample_training_data(X_train, Y_train,
                                                                            estimated_proportions)
    elif sample_method == 'bootstrap':
        X_train_adjusted, Y_train_adjusted = bootstrap_training_data(X_train, Y_train,
                                                                            estimated_proportions)
    else:
        raise ValueError("Method should be either 'subsample' or 'bootstrap'")

    return X_train_adjusted, Y_train_adjusted, trained_occ_list

def estimate_unlabeled_proportions(K, X_train, Y_train, X_unlabeled, oneclass_classifier, mixture_distributions):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)

    # Train one-class classifiers for each inlier type
    occ_list = [copy.deepcopy(oneclass_classifier) for _ in range(K)]
    for i in range(K):
        X_train_i = X_train_scaled[Y_train == (i + 1)]
        if len(X_train_i) > 0:
            occ_list[i].fit(X_train_i)
        else:
            occ_list[i] = None  # Mark as None if no data to fit

    # Create a 2D NumPy array to store scores
    scores_train = np.empty((K, K), dtype=object)
    scores_unlabeled = np.zeros((K, len(X_unlabeled_scaled)))

    for k in range(K):
        for j in range(K):
            X_train_k = X_train_scaled[Y_train == (k + 1)]
            if occ_list[j] is not None and len(X_train_k) > 0:
                scores_train[k, j] = occ_list[j].score_samples(X_train_k)
            else:
                scores_train[k, j] = np.full(len(X_train_k), np.nan)  # Use NaN for missing scores

    # Get scores for unlabeled data
    for k in range(K):
        if occ_list[k] is not None:
            scores_unlabeled[k, :] = occ_list[k].score_samples(X_unlabeled_scaled)

    if mixture_distributions == 'normal':

        # Initialize models and fit them
        models = []
        for k in range(K):
            # Flatten the scores into a 2D array, ignoring NaNs
            scores_flat = np.column_stack([scores_train[k, j] for j in range(K) if np.any(~np.isnan(scores_train[k, j]))])
            if scores_flat.size > 0:
                model = Normal(covariance_type='diag').fit(scores_flat)
                means = model.means
                covs = model.covs
                model = Normal(means=means, covs=covs, covariance_type='diag', frozen=True)
                models.append(model)
            else:
                models.append(Normal())  # Append an empty, unfitted Normal model

        # Append an empty, unfitted Normal model to the list
        models.append(Normal())

    elif mixture_distributions == 'gamma':
        # Ensure that scores_train and scores_unlabeled are all positive
        scores_train_flat = np.concatenate([s[~np.isnan(s)] for s in scores_train.flat if s is not None])
        scores_unlabeled_flat = scores_unlabeled.flatten()

        # Find the smallest negative value efficiently
        min_negative_train = np.min(scores_train_flat[scores_train_flat < 0], initial=0)
        min_negative_unlabeled = np.min(scores_unlabeled_flat[scores_unlabeled_flat < 0], initial=0)
        min_negative = min(min_negative_train, min_negative_unlabeled)

        if min_negative < 0:
            adjustment = -min_negative
            # Add adjustment to all scores efficiently
            for k in range(K):
                for j in range(K):
                    if scores_train[k, j] is not None:
                        scores_train[k, j] += adjustment + 1
            scores_unlabeled += adjustment + 1

        # Initialize models and fit them
        models = []
        for k in range(K):
            scores_flat = np.column_stack([scores_train[k, j] for j in range(K) if np.any(~np.isnan(scores_train[k, j]))])
            if scores_flat.size > 0:
                model = Gamma().fit(scores_flat)
                shapes = model.shapes
                rates = model.rates
                model = Gamma(shapes=shapes, rates=rates, frozen=True)
                models.append(model)
            else:
                models.append(Gamma())  # Append an empty, unfitted Gamma model

        # Append an empty, unfitted Normal model to the list
        models.append(Gamma())

    elif mixture_distributions == 'uniform':

        # Initialize models and fit them
        models = []
        for k in range(K):
            scores_flat = np.column_stack([scores_train[k, j] for j in range(K) if np.any(~np.isnan(scores_train[k, j]))])
            if scores_flat.size > 0:
                model = Uniform().fit(scores_flat)
                mins = model.mins
                maxs = model.maxs
                model = Uniform(mins=mins, maxs=maxs, frozen=True)
                models.append(model)
            else:
                models.append(Uniform())  # Append an empty, unfitted Uniform model

        # Append an empty, unfitted Normal model to the list
        models.append(Uniform())

    else:
        raise ValueError("mixture distributions should be either 'normal', 'gamma', 'uniform' or 'beta'")

    # Initialize the GeneralMixtureModel with K normal distributions
    model_final = GeneralMixtureModel(models, verbose=True).fit(scores_unlabeled.T)

    proportions = model_final.priors
    proportions = proportions[0:K]/(1 - proportions[-1])

    return proportions, occ_list