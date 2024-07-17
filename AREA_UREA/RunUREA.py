from model import LinearClassifier, NNClassifier
import torch
from loss import hinge_loss
from read_mat import get_train_test_data_from_mat
from make_mpu_dataset import make_mpu_train_set
from generate_dataset import generate_dataset
from torch.utils.data import DataLoader
import time

def train(train_dataloader, X_test, Y_test, model, optimizer, class_prior,  loss_name, epoch_num):
    if loss_name == "sigmoid":
        loss_fn = sigmoid_loss
    elif loss_name == "ramp":
        loss_fn = ramp_loss
    elif loss_name == "hinge":
        loss_fn = hinge_loss
    else:
        loss_fn = zero_loss
    
    K = class_prior.shape[0] # number of classes
    y_test = Y_test.argmax(dim=-1)
    
    for epoch in range(epoch_num):
        for batch_data in train_dataloader:
            batchX = batch_data['batchX']
            batchY = batch_data['batchY']
            
            label_sum_for_each_instance = batchY.sum(dim=1) # number of examples in the batch
            unlabeled_index = (label_sum_for_each_instance == 0) # find the index of unlabeled data
            labeled_index = (label_sum_for_each_instance != 0)
            
            labeled_num = labeled_index.sum()
            unlabeled_num = unlabeled_index.sum()
            
            loss_1_average = 0
            loss_2_average = 0
            optimizer.zero_grad()
            
            if labeled_num > 0:
                labeled_batchX = batchX[labeled_index, :]
                labeled_batchY = batchY[labeled_index, :]
                y_labeled_batch = labeled_batchY.argmax(dim=-1)
                class_prior_per_instance = class_prior[y_labeled_batch].float() # class prior for each labeled instance, n 
                class_num_per_instance = labeled_batchY.sum(dim=0)[y_labeled_batch].float() # number of examples in each 
                pred_labeled = model(labeled_batchX)
                labeled_margin = (pred_labeled*labeled_batchY).sum(dim=1)
                loss_1_temp = loss_fn(labeled_margin) - loss_fn(pred_labeled[:,-1]) + 1.0/(K-1)*(loss_fn(-pred_labeled[:,-1])-loss_fn(-labeled_margin))
                loss_1_average = ((loss_1_temp * class_prior_per_instance)/class_num_per_instance).sum()
                
            if unlabeled_num > 0:
                unlabeled_batchX = batchX[unlabeled_index, :]
                pred_unlabeled = model(unlabeled_batchX)
                loss_2_temp = loss_fn(-pred_unlabeled[:,:-1]).mean(dim=1) + loss_fn(pred_unlabeled[:,-1])
                loss_2_average = loss_2_temp.mean()
                
            train_loss = loss_1_average + loss_2_average

        
            train_loss.backward()
            optimizer.step()
        
        if epoch % 1 == 0:
            test_acc= evaluate(model, X_test, y_test)
            print('number of epoch', epoch, ': train_loss', train_loss.data.item(), 'test_accuracy', test_acc)
    return 




def evaluate(model, X_test, y_test):
    outputs = model(X_test)
    pred = outputs.argmax(dim=1)
    accuracy = (pred==y_test).sum().item()/pred.shape[0]

    return accuracy


if __name__ == '__main__':
    dataname = "./dataset/uci_usps.mat"
    test_size = 0.2
    unlabeled_ratio = 0.5
    loss_name = "hinge" # hinge, sigmoid
    epoch_num = 200
    start = time.clock()
    X_train, Y_train, X_test, Y_test = get_train_test_data_from_mat(dataname, test_size)
    class_prior, X_labeled_train, Y_labeled_train, X_unlabeled_train, Y_unlabeled_train = make_mpu_train_set(X_train, Y_train, unlabeled_ratio)
    [dim1, dim2] = Y_unlabeled_train.shape
    pseudo_Y_unlabeled_train = torch.zeros(dim1, dim2) # create the zero label matrix for the unlabeled data
    
    trainX = torch.cat((X_labeled_train, X_unlabeled_train), dim=0)
    trainY = torch.cat((Y_labeled_train, pseudo_Y_unlabeled_train), dim=0)
    trainset = generate_dataset(trainX, trainY)
    train_dataloader = DataLoader(trainset, batch_size=256, shuffle=True) #num_workers=4, 256 batch_size for usps
    
    model = LinearClassifier(X_labeled_train.shape[1], Y_train.shape[1]) #LinearClassifier and NNClassifier
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)  #1e-3,1e-3 for usps
    
    train(train_dataloader, X_test, Y_test, model, optimizer, class_prior, loss_name, epoch_num)
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)

