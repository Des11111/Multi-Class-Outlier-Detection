import numpy as np
from sklearn.preprocessing import StandardScaler
import copy

def compute_pu_scores(X_train, Y_train, X_cal, Y_cal, X_test, binary_classifier, two_step=True, oneclass_classifier=None):   
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  
    X_cal_scaled = scaler.transform(X_cal)
    X_test_scaled = scaler.transform(X_test)
    
    # Define unlabeled set
    X_unlabeled_scaled = np.concatenate([X_cal_scaled, X_test_scaled], 0)
    
    if two_step:
        assert (oneclass_classifier is not None)
        
        # Step 1: Train a one-class SVM on positive samples
        oneclass_classifier.fit(X_train_scaled)
        # Apply the one-class SVM to the unlabeled data
        pred_unlabeled = oneclass_classifier.predict(X_unlabeled_scaled)
        reliable_negatives_idx = np.where(pred_unlabeled == -1)[0]  # Select reliable negatives
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

    # After fitting final_clf, predict probabilities
    scores_cal = binary_classifier.predict_proba(X_cal_scaled)[:, 1]
    scores_test = binary_classifier.predict_proba(X_test_scaled)[:, 1]
   
    return scores_cal, scores_test


def compute_pu_scores_multiclass(X_train, Y_train, X_cal, Y_cal, X_test, binary_classifier, oneclass_classifier):  
    # TASK: extend this to K classes
    # TASK: think about whether this can be improved more
    # TASK: compare to existing methods
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  
    X_cal_scaled = scaler.transform(X_cal)
    X_test_scaled = scaler.transform(X_test)
    
    # Define unlabeled set
    X_unlabeled_scaled = np.concatenate([X_cal_scaled, X_test_scaled], 0)
   
    # Make two copies of the occ model
    occ_1 = copy.deepcopy(oneclass_classifier)
    occ_2 = copy.deepcopy(oneclass_classifier)
        
    # Step 1: Train a one-class SVM on positive samples
    # Make the code more robust, make sure it won't crash if there is only one type of inlier in the data
    X_train_scaled_1 = X_train_scaled[np.where(Y_train==1)[0]]
    X_train_scaled_2 = X_train_scaled[np.where(Y_train==2)[0]]
    occ_1.fit(X_train_scaled_1)
    occ_2.fit(X_train_scaled_2)
    

    # Apply the one-class SVM to the unlabeled data
    # This could be improved:
    # Maybe predict (uses an arbitrary threshold) is not optimal
    pred_unlabeled_1 = occ_1.predict(X_unlabeled_scaled)
    pred_unlabeled_2 = occ_2.predict(X_unlabeled_scaled)
    reliable_negatives_idx = np.where( (pred_unlabeled_1 == -1) * (pred_unlabeled_2 == -1) ==1 ) [0]  # Select reliable negatives
    X_reliable_negatives = X_unlabeled_scaled[reliable_negatives_idx]

    # Combine the positive samples with reliable negatives
    X_combined = np.vstack((X_train_scaled, X_reliable_negatives))
    y_combined = np.hstack((np.zeros(len(X_train_scaled)), np.ones(len(X_reliable_negatives))))
        
    # Step 2: Train a binary classifier on the selected positives and reliable negatives
    binary_classifier.fit(X_combined, y_combined)

    # After fitting final_clf, predict probabilities
    scores_cal = binary_classifier.predict_proba(X_cal_scaled)[:, 1]
    scores_test = binary_classifier.predict_proba(X_test_scaled)[:, 1]
   
    return scores_cal, scores_test