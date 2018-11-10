'''
ROLL - 16MA20020
NAME - HIMANSHU

ASSIGNMENT NUMBER - ** 7 **  
                    K-Means Clustering

OPTIONAL ARGUMENT:
--train     input file name to take input(DEFAULT = "data7.csv"),
            Assume same size in input.

--k	        Number of k in k-means clustering

--epochs    Number of epochs to iterate
'''

# Imports
import csv
import argparse
import numpy as np



class k_means():
    def __init__(self, k=2, n_epochs=10):
        self.k = k
        self.n_epochs = n_epochs
        self.labels = None
        self.cluster_centres = None

    def __call__(self, X=None, y=None):
        # Initialize the weights. Bias is first variable
        self.labels = np.zeros((X.shape[0])).astype(int)
        np.random.seed(seed=30)
        #self.cluster_centres = X[np.random.randint(X.shape[0], size=self.k), :]
        X1 = X
        np.random.shuffle(X1)
        self.cluster_centres = X1[:2,:]
        self.update_labels(X)

    def update_labels(self, X=None):
        if X is not None:
            for i in range(X.shape[0]):
                e_d = np.zeros((self.k))
                for j in range(self.cluster_centres.shape[0]):
                    e_d[j] = self.euclidean_dist(self.cluster_centres[j], X[i])
                self.labels[i] = np.argmin(e_d)+1

    def update_cluster_centres(self, X=None):
        for i in range(self.k):
            self.cluster_centres[i] = np.sum(X[self.labels==i+1,:], axis=0) / np.sum(self.labels==1).astype(float)

    def euclidean_dist(self, x1=None, x2=None):
        # Function to calculate euclidean distance of a vector
        sq_sum = 0
        if x1 is not None and x2 is not None and x1.shape==x2.shape:
            for i in range(x1.shape[0]):
                sq_sum = sq_sum + (x1[i]-x2[i])*(x1[i]-x2[i])
        return np.sqrt(sq_sum)
            

    def train(self, X=None, y=None):
        for i in range(self.n_epochs):
            self.update_labels(X = X)
            self.update_cluster_centres(X = X)



def main():

    # Arguments to take file name from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default="data7.csv", type=str)
    parser.add_argument('--k', default=2, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    args = parser.parse_args()
    train = args.train
    k = args.k
    epochs = args.epochs

    # Input DATA
    data = []
    test_data = []

    with open(train, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)

    # Change data from string to float
    data = np.array(data).astype(float)

    # Make labels and training data
    X_train = data

    # Train the model
    model = k_means(k=k, n_epochs=epochs)
    model(X_train)

    #print(model.test(X_train))
    final_label = model.labels
    
    # Write to output file
    with open('16MA20020_7.out', 'w') as f:
        for i in final_label:
            print(i, end=' ', file=f)



if __name__=="__main__":
    main()
