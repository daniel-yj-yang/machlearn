# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

# reference: PyTorch Lightning package

from ..datasets import public_dataset

from ._linear_regression import linear_regression

# some part of the content below was originally based on cognitiveclass.ai under the terms of the MIT License.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import pandas as pd

# class for ploting
class plot_error_surfaces(object):
    # Constructor
    def __init__(self, w_range, b_range, X, Y, n_samples=30, go=True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)
        Z = np.zeros((30, 30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                Z[count1, count2] = np.mean((self.y - w2 * self.x + b2) ** 2)
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go == True:
            plt.figure()
            plt.figure(figsize=(12 , 8))
            plt.axes(projection='3d').plot_surface(self.w, self.b, self.Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()

    # Setter
    def set_para_loss(self, model, loss):
        self.n = self.n + 1
        self.LOSS.append(loss)
        self.W.append(list(model.parameters())[0].item())
        self.B.append(list(model.parameters())[1].item())

    # Plot diagram
    def final_plot(self):
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W, self.B, self.LOSS, c='r', marker='x', s=200, alpha=1)
        plt.figure()
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()

    # Plot diagram
    def plot_ps(self):
        plt.figure(figsize=(12, 8))
        plt.subplot(121)
        plt.ylim()
        plt.plot(self.x, self.y, 'ro', label="training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label="estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.ylim((-10, 15))
        plt.title('Data Space Iteration: ' + str(self.n))
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.title('Loss Surface Contour Iteration' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()

# Import libraries and set random seed
import torch
from torch.utils.data import Dataset, DataLoader

# Create a linear regression model class
from torch import nn, optim

# Create Data Class
class Data(Dataset):
    # Constructor
    def __init__(self, demo_dataset="marketing"):
        if demo_dataset == "marketing":
            data = public_dataset(name="marketing")
            print(f"{data.head()}\n")
            # X and y
            #X = data[['youtube', 'facebook', 'newspaper']]
            #y = data['sales']
        #
        self.x = torch.from_numpy(np.array(data['youtube'].values).reshape(-1, 1).astype(np.float32))
        self.y = torch.from_numpy(np.array(  data['sales'].values).reshape(-1, 1).astype(np.float32))
        #
        self.len = self.x.shape[0]
        
    # Getter
    def __getitem__(self,index):    
        return self.x[index],self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len


class linear_regression_pytorch(nn.Module):
    # Constructor
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    # Prediction
    def forward(self, x):
        yhat = self.linear(x)
        return yhat


def demo_pytorch(demo_dataset="marketing"):
    torch.manual_seed(1)

    # Create dataset object
    dataset = Data(demo_dataset=demo_dataset)

    linear_regression(use_statsmodels=True).run(X=pd.DataFrame(dataset.x.numpy(), columns=['youtube']), y=pd.Series(dataset.y.numpy().reshape(-1,)))
    
    # Plot the data
    plt.plot(dataset.x.numpy(), dataset.y.numpy(), 'rx', label='y')
    slope_, intercept_ = np.polyfit(dataset.x.numpy().flatten(), dataset.y.numpy(), 1)
    print(f"from np.polyfit(): slope: {slope_}, intercept: {intercept_}")
    plt.plot(dataset.x.numpy(), slope_*dataset.x.numpy() + intercept_, label='f')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # Build in cost function
    criterion = nn.MSELoss()

    # Create optimizer
    model = linear_regression_pytorch(input_size=1, output_size=1)
    optimizer = optim.SGD(model.parameters(), lr = 0.000015)

    # Many of the keys correspond to more advanced optimizers.
    print(f"\nrandom initial parameters. list(model.parameters()): {list(model.parameters())}")
    #print(f"\noptimizer.state_dict(): {optimizer.state_dict()}")

    # Create Dataloader object
    # if batch_size=1, the first element is [tensor([[276.1200]]), tensor([[26.5200]])]   (X_sample1), (y_sample1)
    # if batch_size=2, the first element is [tensor([[276.1200], [ 53.4000]]), tensor([[26.5200], [12.4800]])]   (X_sample1, X_sample2), (y_sample1, y_sample2)
    trainloader = DataLoader(dataset=dataset, batch_size=1)

    # show the weight and bias
    model.state_dict()['linear.weight'][0] = 0.05 # 1
    model.state_dict()['linear.bias'][0] = 8 # 10
    print(f"\nreset to zero, list(model.parameters()): {list(model.parameters())}")

    # Create plot surface object
    get_surface = plot_error_surfaces(w_range=1, b_range=10, X=dataset.x, Y=dataset.y, n_samples=30, go=False)

    # Train Model
    def train_model_BGD(iter):
        # 1 epoch means going through the whole training dataset
        for epoch in range(iter):
            print(f"\nEpoch: {epoch}. Before this epoch: {model.state_dict()}")
            # a batch means how many data points are used to update/train the model each time
            for x, y in trainloader:
                #print(f"epoch = {epoch}, len(x) = {len(x)}, len(y) = {len(y)}")
                yhat = model(x)
                loss = criterion(yhat, y)
                #print(loss)
                get_surface.set_para_loss(model, loss.tolist())
                optimizer.zero_grad()
                loss.backward()  # this handles the differentiation
                optimizer.step()
            get_surface.plot_ps()

    train_model_BGD(iter=5)

