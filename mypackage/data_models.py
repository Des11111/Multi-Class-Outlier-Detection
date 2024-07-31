import numpy as np

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

