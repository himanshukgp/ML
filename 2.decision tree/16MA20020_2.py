'''
ROLL - 16MA20020
NAME - HIMANSHU

ASSIGNMENT NUMBER - ** 2 **  
					Decision Trees

OPTIONAL ARGUMENT:
--train     input file name to take input(DEFAULT = "data2.csv"),
			Assume same size in input.

--test     input file name to take input(DEFAULT = "test2.csv"),
			Assume same size in input.

'''

# Imports
import csv
import argparse
import numpy as np

# Arguments to take file name from command line
parser = argparse.ArgumentParser()
parser.add_argument('--train', default="data2.csv", type=str)
parser.add_argument('--test', default="test2.csv", type=str)
args = parser.parse_args()
train = args.train
test = args.test


# Classes for storing nodes and Leaf of trees respectively.
class Node(object):
    def __init__(self, split_variable = None):
        self.split_variable = split_variable
        self.left_child = None
        self.right_child = None

    def get_name(self):
        return 'Node'

class Leaf(object):
    def __init__(self, value=None):
        self.value = value

    def get_name(self):
        return 'Leaf'


# Calculate entropy with respect to one feature.
def calculate_entropy_feature(feature,label):

	count_true1 = 0
	count_true0 = 0
	count0 = 0
	count1 = 0
	ent_1 = 0
	ent_0 = 0
	p=0

	for i in range(feature.shape[0]):
		if feature[i]==1 and label[i]==1: count_true1+=1
		if feature[i]==0 and label[i]==1: count_true0+=1
		if feature[i]==1: count1+=1
		if feature[i]==0: count0+=1

	if count1!=0: 
		p1=count_true1/count1
		ent_1 = (-1*p1*np.log2(p1) if p1 else 0) - ((1-p1)*np.log2(1-p1) if (1-p1) else 0)
	if count0!=0: 
		p0=count_true0/count0
		ent_0 = (-1*p0*np.log2(p0) if p0 else 0) - ((1-p0)*np.log2(1-p0) if (1-p0) else 0)
	
	if count0+count1!=0: p=count0/(count0+count1)

	return p*ent_0+(1-p)*ent_1


# Calculate entropy of all features and assuming
# end column to be the labels.
def calculate_entropy(data):
	res=np.zeros(data.shape[1]-1)
	for i in range(data.shape[1]-1):
		p = calculate_entropy_feature(data[:,i],data[:,-1])
		if not np.isnan(p):
			res[i]=p
		#print(p)

	return res


# Create the tree assuming all features to be binary
def create_tree(data):
	if np.all(data[:,-1] == data[0,-1], axis = 0):
		return Leaf(data[0][-1])

	column_min_entropy = np.argmin(calculate_entropy(data))
	#print(column_min_entropy)
	node = Node(column_min_entropy)

	data1 = data[data[:, column_min_entropy] == 0]
	data2 = data[data[:, column_min_entropy] == 1]
	data1 = np.delete(data1, column_min_entropy, axis=1)
	data2 = np.delete(data2, column_min_entropy, axis=1)

	node.left_child = create_tree(data1)
	node.right_child = create_tree(data2)

	return node


# To get value of just one example
def test_tree_one(row, node):
	while(node.get_name()!='Leaf'):
		if 0 == row[node.split_variable]:
			row = np.delete(row, node.split_variable)
			node = node.left_child
		else:
			row = np.delete(row, node.split_variable)
			node = node.right_child
	return node.value


# Classify all the examples provided as data on the decision
# tree, assuming all data binary and given as 0 or 1.
def test_tree(data, node):
	res = np.zeros(data.shape[0])
	for i in range(data.shape[0]):
		res[i] = test_tree_one(data[i], node)
	return res


# Print each node and Leaf
# NOTE as we delete feature after it is used the index output
# is according to that. For example if [1,1,0,1,0] is the example
# case then let if 2 is first chosen index of feature and next is
# 2 again then it is 2 in [1,1,1,0] which is actually 3 column
# in the original example data.
def print_node(node):
	if node.get_name() == 'Leaf': 
		return
	print_node(node.left_child)
	print_node(node.right_child)


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

	#Create the tree using training examples
	node = create_tree(data)

	# Test the tree on training set and compare with its labels
	#print(test_tree(data[:,:8], node).astype(int))
	#print(data[:,-1])

	# Classification on test set given
	res = test_tree(test_data, node).astype(int)

	with open('16MA20020_2.out', 'w') as f:
		for i in res:
			print(i, end=' ', file=f)


if __name__=="__main__":
    main()

