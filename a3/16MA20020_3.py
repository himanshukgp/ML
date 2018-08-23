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

	p1 = count_00/(count_00+count_10)
	p2 = count_10/(count_00+count_10)
	p3 = count_01/(count_01+count_11)
	p4 = count_11/(count_01+count_11)
	return p1, p2, p3, p4 


def find_probabilities(data):
	res = np.zeros((data.shape[0],2,2))
	label = data[:,-1]
	data = data[:,:data.shape[0]-1]
	for i in range(data.shape[1]):
		res[i][0][0], res[i][1][0], res[i][0][1], res[i][1][1] = find_probabilities_single(data[:,i],label)
	return res


def bayes_classify_one(row, prob_matrix):
	p_0 = 1
	for i in range(row.shape[0]):
		p_0 *= prob_matrix[i][row[i]][0]
	p_1 = 1
	for i in range(row.shape[0]):
		p_1 *= prob_matrix[i][row[i]][1]
	return 0 if p_0>p_1 else 1


def bayes_classify(data, prob_matrix):
	res = np.zeros(data.shape[0])
	for i in range(data.shape[0]):
		res[i]=bayes_classify_one(data[i], prob_matrix)
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

	prob_matrix = find_probabilities(data)

	res1 = bayes_classify(data[:,:8], prob_matrix).astype(int)
	#print(res1)
	#print(data[:,-1])

	res = bayes_classify(test_data, prob_matrix).astype(int)

	with open('16MA20020_3.out', 'w') as f:
		for i in res:
			print(i, end=' ', file=f)


if __name__=="__main__":
    main()
