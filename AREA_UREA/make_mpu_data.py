import torch
def make_mpu_train_set(X_train, Y_train, unlabeled_ratio):
    y_train = Y_train.argmax(dim=1) # convert one-hot to integer
    label_set = torch.unique(y_train, sorted=True)
    print(label_set)
    y_train_hat = y_train.clone()
    
    trainset_label_num = []
    
    for label in label_set[:-1]:
        is_equal = (y_train==label) # true or false
        index = torch.arange(0, y_train.shape[0])
        selected_index = index[is_equal]
        num_ins_in_class = selected_index.shape[0]
        unlabeled_num = torch.floor(torch.tensor(num_ins_in_class*unlabeled_ratio)).int()
        y_train_hat[selected_index[:unlabeled_num]]=-1 # pick the example belonging to the class
        trainset_label_num.append(num_ins_in_class)
        
    y_train_hat[y_train==label_set[-1]]=-1 # regard the examples in the last class as unlabeled
    trainset_label_num.append((y_train==label_set[-1]).sum())
    trainset_label_num = torch.tensor(trainset_label_num).float()
    class_prior = trainset_label_num/trainset_label_num.sum()
    class_prior = torch.tensor(class_prior).float()
    unlabeled_check = (y_train_hat == -1)
    X_unlabeled_train = X_train[unlabeled_check]
    Y_unlabeled_train = Y_train[unlabeled_check]
    labeled_check = (y_train_hat != -1)
    X_labeled_train = X_train[labeled_check]
    Y_labeled_train = Y_train[labeled_check]
    
    return class_prior, X_labeled_train, Y_labeled_train, X_unlabeled_train, Y_unlabeled_train
