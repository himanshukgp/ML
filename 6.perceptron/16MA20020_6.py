'''
ROLL - 16MA20020
NAME - HIMANSHU

ASSIGNMENT NUMBER - ** 6 **  
                    Perceptron Learning

OPTIONAL ARGUMENT:
--train     input file name to take input(DEFAULT = "data4.csv"),
            Assume same size in input.

--test      input file name to take input(DEFAULT = "test4.csv"),
            Assume same size in input.

--lr		input learning rate of the perceptron.

--epochs    Number of epochs to iterate
'''

# Imports
import csv
import argparse
import numpy as np



class perceptron():
    def __init__(self, lr=0.1, n_epochs=20):
        self.lr = lr
        self.n_epochs = n_epochs
        self.w = None
        self.delta_w = None
        self.b = None
        self.delta_b = None

    def __call__(self, X=None, y=None):
        # Initialize the weights and Bias
        self.delta_w = np.zeros(X.shape[1])
        self.delta_b = 0
        np.random.seed(seed=30)
        self.w = np.random.rand(X.shape[1])
        self.b = 0
        self.train(X, y)

    def sigmoid(self, x):
        # Function to calculate sigmoid activation
        return (1 / (1 + np.exp(-x)))

    def forward_prop(self, X=None):
        a_ = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            a_[i] = np.dot(X[i], self.w)
        return self.sigmoid(x=a_+self.b)

    def train(self, X=None, y=None):
        for k in range(self.n_epochs):
            #print("Epoch .... ",k)

            for i in range(self.delta_w.shape[0]):
                self.delta_w[i]=0
            self.delta_b = 0

            a_ = self.forward_prop(X)
            for i in range(X.shape[0]):
                for j in range(self.delta_w.shape[0]):
                    self.delta_w[j] = self.delta_w[j] + (
                        self.lr * (y[i]-a_[i]) * a_[i] * (1-a_[i]) * X[i][j])
                #self.delta_b = 

            for i in range(self.w.shape[0]):
                self.w[i] = self.w[i] + self.delta_w[i]

    def test(self, X=None):
        a_ = self.forward_prop(X)
        for i in range(X.shape[0]):
            if a_[i]<0.5:
                a_[i]=0
            else:
                a_[i]=1
        return a_.astype(int)



def main():

    # Arguments to take file name from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default="data6.csv", type=str)
    parser.add_argument('--test', default="test6.csv", type=str)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    args = parser.parse_args()
    train = args.train
    test = args.test
    lr = args.lr
    epochs = args.epochs

    # Input DATA
    data = []
    test_data = []

    with open(train, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)

    with open(test, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            test_data.append(row)

    # Change data from string to float
    data = np.array(data).astype(float)
    test_data = np.array(test_data).astype(float)

    # Make labels and training data
    X_train = data[:,:-1]
    y_train = data[:,-1]
    X_test = test_data

    # Train the model
    model = perceptron(lr=lr, n_epochs=epochs)
    model(X_train, y_train)

    #print(model.test(X_train))
    o = model.test(X_test)
    
    # Write to output file
    with open('16MA20020_6.out', 'w') as f:
        for i in o:
            print(i, end=' ', file=f)



if __name__=="__main__":
    main()
