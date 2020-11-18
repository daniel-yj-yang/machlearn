# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

class batch_gradient_descent(object):

    def __init__(self, learning_rate=0.01, num_iter=100000, verbose=False, use_simplified_cost=False):
        super().__init__()
        self.X = None
        self.y = None
        self.h = None  # y-hat
        self.cost = None
        self.theta = None
        self.gradient = None # the slope
        self.alpha = learning_rate
        self.num_iter = num_iter
        self.verbose = verbose
        self.use_simplified_cost = use_simplified_cost
        self.n_features = None

    # the following three functions are the core of BGD:
    # θ = _y_pred
    # J(θ) = _cost_function
    # ∂J(θ)/∂θ = _gradient
    def _y_pred(self):
        """
        to get self.h (y-hat)
        """
        return None

    def _cost_function(self):
        """
        loss based on self.h, self.y
        generally, the idea is (self.h - self.y) ** 2
        """
        return None

    def _gradient(self):
        """
        the slope, the partial derivative of the cost function with respect to theta
        """
        return None

    def fit(self):
        # weights initialization
        #self.theta = np.random.rand(self.X.shape[1])
        self.n_features = self.X.shape[1]
        self.theta = np.zeros(shape = self.n_features) # if X has an intercept and two features, then self.theta = array([0., 0., 0.])
        #self.theta = np.ones(shape=self.n_features)
        #self.theta = np.array([0.69254314, -0.49262002,  0.19834042])
        self.theta_history = np.zeros(shape = (self.num_iter, self.n_features))
        self.cost_history = np.zeros(shape = self.num_iter)
        self.gradient_history = np.zeros(shape = (self.num_iter, self.n_features))

        # Note:
        # Batch gradient descent means that we calculate the error for each example in the training dataset, but update the model only after the entire training set has been evaluated.
        # One cycle through the entire training set is called a training epoch.
        for epoch in range(self.num_iter):

            # Step #1: make predictions (self.h = y-hat) on training data.
            self.h = self._y_pred()

            # Step #2: use the error (cost, loss) on the predictions to update the model in a way to minimize the error.
            if self.use_simplified_cost:
                self.cost = np.sum((self.h - self.y) ** 2) # this cost function could be wavy and non-convex (e.g., for logistic regression problem)
            else:
                self.cost = self._cost_function()

            # Step #3: update the model to move it along a gradient (slope) of errors down toward a minimum error value.
            self.gradient = self._gradient() # the slope
            self.theta -= self.alpha * self.gradient

            # self.theta = array([0., 0., 0.]) is not good for plotting, so starting from here
            self.theta_history[epoch,:] = self.theta
            self.cost_history[epoch] = self.cost
            self.gradient_history[epoch,:] = self.gradient

            if(self.verbose == True and epoch % 10000 == 0):
                print(f"#epoch: {epoch}, cost: {self.cost:f}\n")

    def predict_prob(self, X):
        pass

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold

    def plot_decision_boundary(self, epoch=None):
        pass

    def animate_decision_boundary(self):
        pass

    def plot_training_history(self):
        import matplotlib.pyplot as plt
        # construct a figure that plots the loss over time
        plt.figure(figsize=(3, 3))
        plt.plot(np.arange(0, self.num_iter), self.cost_history) #, label='BGD Training Loss')
        #plt.legend(loc=1)
        plt.xlabel("Training Epoch/Iteration #")
        plt.ylabel("Error/Cost/Loss, J(θ)")
        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(nrows = 1, ncols = self.n_features, figsize=(self.n_features*4, 4))
        fig.suptitle('Cost vs. Theta')
        color_dict = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for theta_idx in range(self.n_features):
            axs[theta_idx].plot(self.theta_history[:, theta_idx], self.cost_history, color_dict[theta_idx%4])
            axs[theta_idx].set_title(f"$\\theta_{theta_idx}$")
            axs[theta_idx].set(xlabel=f"$\\theta_{theta_idx}$ values", ylabel='Error/Cost/Loss, J(θ)')
            axs[theta_idx].label_outer()
        plt.tight_layout()
        plt.show()


class logistic_regression_BGD_classifier(batch_gradient_descent):

    def __init__(self, fit_intercept=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_intercept = fit_intercept

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # the following three functions are the core of BGD:
    # θ = _y_pred
    # J(θ) = _cost_function
    # ∂J(θ)/∂θ = _gradient
    def _y_pred(self):
        """
        to get self.h (y-hat)
        """
        return self._sigmoid(self.X.dot(self.theta))

    def _cost_function(self):
        """
        loss based on self.h, self.y
        generally, the idea is (self.h - self.y) ** 2

        one sample's error is called loss
        the sum or average of loss across all the sample (batch) is called cost function
        """
        # cross-entropy, log loss function
        return -1 * (self.y.T.dot(np.log(self.h)) + (1-self.y).T.dot(np.log(1-self.h))) / self.y.size
        #return (-self.y * np.log(self.h) - (1 - self.y) * np.log(1 - self.h)).mean()

    def _gradient(self):
        """
        the slope, the partial derivative of the cost function with respect to theta
        """
        return self.X.T.dot(self.h - self.y) / self.y.size

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.fit_intercept:
            self.X = self._add_intercept(self.X)
        super().fit()

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
        return self._sigmoid(X.dot(self.theta))

################## experimental start ##################

class logistic_regression_torch_model(nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(in_features = n_features, out_features = 1, bias = True)

    def forward(self, x):
        """
        prediction
        """
        y_hat = torch.sigmoid(self.linear(x))
        return y_hat

class logistic_regression_torch_classifier(object):

    def __init__(self, num_iter: int = 100, learning_rate: float = 0.01):
        self.num_iter = num_iter
        self.learning_rate = learning_rate

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        model = logistic_regression_torch_model(X_train.shape[1])
        optimizer = torch.optim.SGD(model.parameters(), lr = self.learning_rate)
        for epoch in range(self.num_iter):
            y_hat = model(X_train)
            loss = self.loss(y_hat, y_train)
            optimizer.zero_grad()
            loss.backward() # gradient
            optimizer.step()

    def loss(self, y_hat, y):
        output = -1 * torch.mean(y * torch.log(y_hat) + (1-y) * torch.log(1-y_hat))
        return output

# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)

################## experimental end ##################

def _demo(dataset, classifier_func, learning_rate=None, num_iter=None):

    from ..datasets import public_dataset

    if dataset == "Gender":
        data = public_dataset(name="Gender")
        print(f"{data.head()}\n")
        print(f"{data['Gender'].value_counts()}\n")
        print(f"{data.describe()}\n")
        # Recode the data: Gender as Male
        mapper_dict = {'Male': 1, 'Female': 0}
        data['Male'] = data['Gender'].map(mapper_dict)
        # pairplot
        import seaborn as sns
        sns.pairplot(data, hue="Male", markers=["o", "s"])
        import matplotlib.pyplot as plt
        plt.show()
        # X and y
        X = data[['Height', 'Weight']]
        y = data['Male'].ravel()
        y_classes = ['Female (y=0)', 'Male (y=1)']

    if dataset == "Social_Network_Ads":
        data = public_dataset(name="Social_Network_Ads")
        print(f"{data.head()}\n")
        print(f"{data['Gender'].value_counts()}\n")
        data = data.drop('User ID', 1)
        print(f"{data.describe()}\n")
        # Recode the data: Gender as Male
        mapper_dict = {'Male': 1, 'Female': 0}
        data['Male'] = data['Gender'].map(mapper_dict)
        # pairplot
        import seaborn as sns
        sns.pairplot(data, hue="Purchased", markers=["o", "s"])
        import matplotlib.pyplot as plt
        plt.show()
        # X and y
        X = data[['Age', 'EstimatedSalary']]
        y = data['Purchased'].ravel()
        y_classes = ['not_purchased (y=0)', 'purchased (y=1)']

    if dataset == "iris_binarized":
        data = public_dataset(name='iris')
        y_classes = ['setosa', 'versicolor', 'virginica']
        X = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
        y = data['target']
        y = (y != 0) * 1

    learning_rate_dict = {'Gender': 0.00025, 'Social_Network_Ads': 1e-9, 'iris_binarized': 0.1}
    if learning_rate is None:
        learning_rate = learning_rate_dict[dataset]

    num_iter_dict = {'Gender': 30000, 'Social_Network_Ads': 300, 'iris_binarized': 300}
    if num_iter is None:
        num_iter = num_iter_dict[dataset]

    if classifier_func == "logistic_regression":
        # comparison
        if dataset != "iris_binarized":
            from ..logistic_regression import logistic_regression_statsmodels
            params_values = logistic_regression_statsmodels().run(X, y)
            #print(params_values)
        # torch version of logistic regression
        #torch_model = logistic_regression_torch_classifier()
        #torch_model.fit(df_to_tensor(X), df_to_tensor(pd.Series(y)))
        # gradient descent classifier
        classifier = logistic_regression_BGD_classifier(learning_rate=learning_rate, num_iter=num_iter)

    classifier.fit(X=X, y=y)
    classifier.plot_training_history()
    print(f"With learning_rate=[{learning_rate}] and num_iter=[{num_iter}], theta estimates from BGD: {classifier.theta} (may not be the same as algebra solution)")
    y_pred = classifier.predict(X)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy of prediction: {accuracy: .3f}")
    return classifier


def demo(dataset="Gender", classifier_func="logistic_regression", learning_rate=None, num_iter=None):
    available_datasets = ["Gender", "Social_Network_Ads", "iris_binarized", ]
    available_classifiers = ["logistic_regression", ]

    if dataset not in available_datasets:
        raise ValueError(f"dataset={dataset} is not yet implemented in this demo.")

    if classifier_func not in available_classifiers:
        raise ValueError(f"classifier_func={classifier_func} is not yet implemented in this demo.")

    return _demo(dataset=dataset, classifier_func=classifier_func, learning_rate=learning_rate, num_iter=num_iter)
        
