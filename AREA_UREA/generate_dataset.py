from torch.utils.data import Dataset

class generate_dataset(Dataset):
    def __init__(self, X, Y): # X, Y are both matrix
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index,:]
        y = self.Y[index,:]
        
        return {'batchX': x, 'batchY': y}
