import torch

import torch.nn as nn

class NN(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, weights):
        super(NN, self).__init__()
        self.a = n_input
        self.b = n_hidden1
        self.c = n_hidden2
        self.d = n_output
        
        self.fc1 = nn.Linear(n_input, n_hidden1)  
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.out = nn.Linear(n_hidden2, n_output)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.set_weights(weights)
        
    def set_weights(self, weights):
        weights = torch.FloatTensor(weights)
        with torch.no_grad():
            x = self.a * self.b
            xx = x + self.b
            y = xx + self.b * self.c
            yy = y + self.c
            z = yy + self.c * self.d
            
            self.fc1.weight.data = weights[0:x].reshape(self.b, self.a)
            self.fc1.bias.data = weights[x: xx]
            
            self.fc2.weight.data = weights[xx:y].reshape(self.c, self.b)
            self.fc2.bias.data = weights[y: yy]
            
            self.out.weight.data = weights[yy:z].reshape(self.d, self.c)
            self.out.bias.data = weights[z:]
            
    def predict(self, input):
        input = torch.tensor([input]).float()
        y = self.forward(input)
        return torch.argmax(y, dim=1).tolist()[0]
    
    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.out(y)
        y = self.sigmoid(y)
        
        return y
                