'''
ROLL - 16MA20020
NAME - HIMANSHU

ASSIGNMENT NUMBER - ** 4 **  
					K-NN Classifier

OPTIONAL ARGUMENT:
--train     input file name to take input(DEFAULT = "data4.csv"),
			Assume same size in input.

--test      input file name to take input(DEFAULT = "test4.csv"),
			Assume same size in input.

--K         input value of K in Knn

'''

# Imports
import csv
import argparse
import numpy as np

# Arguments to take file name from command line
parser = argparse.ArgumentParser()
parser.add_argument('--train', default="data4.csv", type=str)
parser.add_argument('--test', default="test4.csv", type=str)
parser.add_argument('--K', default=5, type=int)
args = parser.parse_args()
train = args.train
test = args.test
K = args.K


# Utility function to find whether number of 0 or 1
# is greater in a vector
def find_max_label(vec):
	n_0 = 0
	n_1 = 0
	for i in vec:
		if i==0:
			n_0+=1 
		else:
			n_1+=1
	return 0 if n_0>n_1 else 1


# Find the nearest K neighbours using euclidean distance
# of one example return maximum of class labels in those K values
def predict_knn_one(data, point, K):
	dist = (data[:,:8] - point)**2
	dist = np.sum(dist, axis=1)
	dist = np.sqrt(dist)
	idx = np.argpartition(dist, K)[:K]
	return find_max_label(data[idx,-1])


# Classify using given training dataset with last column as labels
# and matrix of test data and K value. Returns vector of predicted values.
def predict_knn(data, test_data, K):
	res = np.zeros(test_data.shape[0])
	for i in range(res.shape[0]):
		res[i] = predict_knn_one(data, test_data[i], K)
	return res


def main():
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

	data = np.array(data).astype(int)
	test_data = np.array(test_data).astype(int)

	# Test on training data
	#res = predict_knn(data, data[:,:8], K).astype(int)
	#print(data[:,-1])

	# Test on given test_data
	res = predict_knn(data, test_data, K).astype(int)

	# Output in a file named '16MA20020_4.out'
	with open('16MA20020_4.out', 'w') as f:
		for i in res:
			print(i, end=' ', file=f)


if __name__=="__main__":
    main()
