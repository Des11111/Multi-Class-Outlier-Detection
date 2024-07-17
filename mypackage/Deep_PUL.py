import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
import itertools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

def train_deep_model(input_dim, X_train, y_train, epochs=10, batch_size=32):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def compute_deep_pu_scores(K, dim, X_train, Y_train, X_cal, Y_cal, X_test_part1, X_test_part2,
                           train_model_fn, two_step=True, oneclass_classifier=None):
    assert callable(train_model_fn), "train_model_fn must be a callable function"

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled = scaler.transform(X_cal)
    X_test_part1_scaled = scaler.transform(X_test_part1)
    X_test_part2_scaled = scaler.transform(X_test_part2)

    if two_step:
        assert oneclass_classifier is not None, "One-class classifier must be provided for two-step method"

        # Step 1: Train one-class classifiers for each inlier type
        occ_list = [copy.deepcopy(oneclass_classifier) for _ in range(K)]
        for i in range(K):
            X_train_scaled_i = X_train_scaled[Y_train == (i + 1)]
            if len(X_train_scaled_i) > 0:
                occ_list[i].fit(X_train_scaled_i)

        # Apply the one-class classifiers to the unlabeled data
        pred_unlabeled_list = [occ.predict(X_test_part1_scaled) for occ in occ_list]

        # Identify reliable negatives
        reliable_negatives_idx = np.where(np.all(np.array(pred_unlabeled_list) == -1, axis=0))[0]
        X_reliable_negatives = X_test_part1_scaled[reliable_negatives_idx]

        # Combine the positive samples with reliable negatives
        X_combined = np.vstack((X_train_scaled, X_reliable_negatives))
        y_combined = np.hstack((np.zeros(len(X_train_scaled)), np.ones(len(X_reliable_negatives))))

    else:
        # Combine the positive samples with the mixed samples
        X_combined = np.vstack((X_train_scaled, X_test_part1_scaled))
        y_combined = np.hstack((np.zeros(len(X_train_scaled)), np.ones(len(X_test_part1_scaled))))

    # Step 2: Train the model on the combined data using the provided train_model_fn
    model = train_model_fn(dim, X_combined, y_combined)

    # After fitting the model, predict probabilities
    scores_cal = model.predict(X_cal_scaled)
    scores_test = model.predict(X_test_part2_scaled)

    return scores_cal, scores_test


# Example usage
# scores_cal, scores_test = compute_deep_pu_scores(K, dim, X_train, Y_train, X_cal, Y_cal, X_test_part1, X_test_part2,
#                                                  binary_classifier, train_deep_model, two_step=True, oneclass_classifier=your_oneclass_classifier)


# def compute_deep_pu_scores(K, dim, X_train, Y_train, X_cal, Y_cal, X_test_part1, X_test_part2, binary_classifier,
#                                 two_step=True, oneclass_classifier=None, deep_learning = True):
#     assert deep_learning is True
#     # Scaling
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_cal_scaled = scaler.transform(X_cal)
#     X_test_part1_scaled = scaler.transform(X_test_part1)
#     X_test_part2_scaled = scaler.transform(X_test_part2)
#
#     if two_step:
#         assert oneclass_classifier is not None, "One-class classifier must be provided for two-step method"
#
#         # Step 1: Train one-class classifiers for each inlier type
#         occ_list = [copy.deepcopy(oneclass_classifier) for _ in range(K)]
#         for i in range(K):
#             X_train_scaled_i = X_train_scaled[Y_train == (i + 1)]
#             if len(X_train_scaled_i) > 0:
#                 occ_list[i].fit(X_train_scaled_i)
#
#         # Apply the one-class classifiers to the unlabeled data
#         pred_unlabeled_list = [occ.predict(X_test_part1_scaled) for occ in occ_list]
#
#         # Identify reliable negatives
#         reliable_negatives_idx = np.where(np.all(np.array(pred_unlabeled_list) == -1, axis=0))[0]
#         X_reliable_negatives = X_test_part1_scaled[reliable_negatives_idx]
#
#         # Combine the positive samples with reliable negatives
#         X_combined = np.vstack((X_train_scaled, X_reliable_negatives))
#         y_combined = np.hstack((np.zeros(len(X_train_scaled)), np.ones(len(X_reliable_negatives))))
#
#     else:
#         # Combine the positive samples with the mixed samples
#         X_combined = np.vstack((X_train_scaled, X_test_part1_scaled))
#         y_combined = np.hstack((np.zeros(len(X_train_scaled)), np.ones(len(X_test_part1_scaled))))
#
#     # Step 2: Train a DNN on the combined data
#     dnn_model = Sequential([
#         Dense(128, activation='relu', input_shape=(dim,)),
#         Dropout(0.5),
#         Dense(64, activation='relu'),
#         Dropout(0.5),
#         Dense(1, activation='sigmoid')
#     ])
#     dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     dnn_model.fit(X_combined, y_combined, epochs=10, batch_size=32, verbose=0)
#
#     # After fitting dnn_model, predict probabilities
#     scores_cal = dnn_model.predict(X_cal_scaled)
#     scores_test = dnn_model.predict(X_test_part2_scaled)
#
#     return scores_cal, scores_test

def compute_deep_pu_scores_intersection_two_step(K, dim, X_train, Y_train, X_cal, Y_cal, X_test_part1, X_test_part2,
                                                     binary_classifier, oneclass_classifier, deep_learning = True):
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled = scaler.transform(X_cal)
    X_test_part1_scaled = scaler.transform(X_test_part1)
    X_test_part2_scaled = scaler.transform(X_test_part2)

    # Step 1: Train a one-class SVM for each inlier type
    occ_list = [copy.deepcopy(oneclass_classifier) for _ in range(K)]

    for i in range(K):
        X_train_scaled_i = X_train_scaled[np.where(Y_train == i + 1)[0]]
        if len(X_train_scaled_i) > 0:
            occ_list[i].fit(X_train_scaled_i)

    # Apply the one-class SVMs to the unlabeled data
    pred_unlabeled_list = [occ.predict(X_test_part1_scaled) for occ in occ_list]

    # Identify reliable negatives
    reliable_negatives_idx = np.where(np.all(np.array(pred_unlabeled_list) == -1, axis=0))[0]
    X_reliable_negatives = X_test_part1_scaled[reliable_negatives_idx]

    # Combine the positive samples with reliable negatives
    X_combined = np.vstack((X_train_scaled, X_reliable_negatives))
    y_combined = np.hstack((np.zeros(len(X_train_scaled)), np.ones(len(X_reliable_negatives))))

    # Step 2: Train a DNN on the combined data
    dnn_model = Sequential([
        Dense(128, activation='relu', input_shape=(dim,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    dnn_model.fit(X_combined, y_combined, epochs=10, batch_size=32, verbose=0)

    # After fitting dnn_model, predict probabilities
    scores_cal = dnn_model.predict(X_cal_scaled)
    scores_test = dnn_model.predict(X_test_part2_scaled)

    return scores_cal, scores_test


