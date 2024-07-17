import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, in_feature, out_feature ):
        super(LinearClassifier, self).__init__()
        self.linear_layer = nn.Linear(in_feature,out_feature,bias=True)
        
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.linear_layer.apply(init_weights)
        
    def forward(self, input):
        out = self.linear_layer(input)
        return out



class NNClassifier(nn.Module):
    def __init__(self, in_feature, out_feature ):
        super(NNClassifier, self).__init__()
        
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.prediction_layer = nn.Sequential(
            nn.Linear(in_feature, 100, bias=True),
            nn.ReLU(),
            nn.Linear(100, out_feature, bias=True)
		)
        
        self.prediction_layer.apply(init_weights)

    def forward(self, input):
        output = self.prediction_layer(input)
        return output