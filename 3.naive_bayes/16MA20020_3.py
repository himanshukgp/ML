'''
ROLL - 16MA20020
NAME - HIMANSHU

ASSIGNMENT NUMBER - ** 3 **  
					Naïve Bayes Classifier

OPTIONAL ARGUMENT:
--train     input file name to take input(DEFAULT = "data3.csv"),
			Assume same size in input.

--test     input file name to take input(DEFAULT = "test3.csv"),
			Assume same size in input.

'''

# Imports
import csv
import argparse
import numpy as np

# Arguments to take file name from command line
parser = argparse.ArgumentParser()
parser.add_argument('--train', default="data3.csv", type=str)
parser.add_argument('--test', default="test3.csv", type=str)
args = parser.parse_args()
train = args.train
test = args.test


# To find the probabilities associated with classes of labels and 
# various feature classes
# returns 4 probabilities at once
def find_probabilities_single(feature, label):
	count_00=0
	count_10=0
	count_01=0
	count_11=0

	for i in range(feature.shape[0]):
		if feature[i]==1 and label[i]==1: count_11+=1
		if feature[i]==1 and label[i]==0: count_10+=1
		if feature[i]==0 and label[i]==1: count_01+=1
		if feature[i]==0 and label[i]==0: count_00+=1

	p1 = (count_00+1)/(count_00+count_10+2) if (count_00+count_10) else 0
	p2 = (count_10+1)/(count_00+count_10+2) if (count_00+count_10) else 0
	p3 = (count_01+1)/(count_01+count_11+2) if (count_01+count_11) else 0
	p4 = (count_11+1)/(count_01+count_11+2) if (count_01+count_11) else 0
	return p1, p2, p3, p4 


# Probability table of entire training data is created
# Input labels as the last column of data
def find_probabilities(data):
	res = np.zeros((data.shape[1]-1,2,2))
	label = data[:,-1]
	data = data[:,:data.shape[1]-1]
	for i in range(data.shape[1]):
		res[i][0][0], res[i][1][0], res[i][0][1], res[i][1][1] = find_probabilities_single(data[:,i],label)
	return res


# To classify just one test case.
# uses probability matrix and Laplacian smoothing of 1.
def bayes_classify_one(row, prob_matrix, n_0, n_1):
	p_0 = 1
	for i in range(row.shape[0]):
		p_0 *= prob_matrix[i][row[i]][0]
	p_1 = 1
	for i in range(row.shape[0]):
		p_1 *= prob_matrix[i][row[i]][1]
	return 0 if p_0*n_0>p_1*n_1 else 1


# Main classifier to classify matrix of test data
# Each row is a test case
def bayes_classify(data, prob_matrix, n_0, n_1):
	res = np.zeros(data.shape[0])
	for i in range(data.shape[0]):
		res[i]=bayes_classify_one(data[i], prob_matrix, n_0, n_1)
	return res


# main driver program
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

	# Calculate probability matrix
	prob_matrix = find_probabilities(data)

	# Calculate the probabilities of classes of labels
	n_1 = np.sum(data[:,-1])/data.shape[0]
	n_0 = 1-n_1

	res = bayes_classify(test_data, prob_matrix, n_0, n_1).astype(int)

	with open('16MA20020_3.out', 'w') as f:
		for i in res:
			print(i, end=' ', file=f)


if __name__=="__main__":
    main()
