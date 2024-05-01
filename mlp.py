import torch

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, weights):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_output = n_output
        
        self.fcl1 = nn.Linear(n_input, n_hidden1)  
        self.fcl2 = nn.Linear(n_hidden1, n_hidden2)
        self.out = nn.Linear(n_hidden2, n_output)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.set_weights(weights)
        
    def set_weights(self, weights):
        weights = torch.FloatTensor(weights)
        with torch.no_grad():
            p1 = self.n_input * self.n_hidden1
            p2 = p1 + self.n_hidden1
            p3 = p2 + self.n_hidden1 * self.n_hidden2
            p4 = p3 + self.n_hidden2
            p5 = p4 + self.n_hidden2 * self.n_output
            
            self.fcl1.weight.data = weights[0:p1].reshape(self.n_hidden1, self.n_input)
            self.fcl1.bias.data = weights[p1: p2]
            
            self.fcl2.weight.data = weights[p2:p3].reshape(self.n_hidden2, self.n_hidden1)
            self.fcl2.bias.data = weights[p3: p4]
            
            self.out.weight.data = weights[p4:p5].reshape(self.n_output, self.n_hidden2)
            self.out.bias.data = weights[p5:]
            
    def predict(self, input):
        input = torch.FloatTensor(input)
        output = self.forward(input)
        return torch.argmax(output).item()
    
    def forward(self, x):
        y = self.fcl1(x)
        y = self.relu(y)
        y = self.fcl2(y)
        y = self.relu(y)
        y = self.out(y)
        y = self.sigmoid(y)
        
        return y
                