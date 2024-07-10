import numpy as np
from sklearn.preprocessing import StandardScaler

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