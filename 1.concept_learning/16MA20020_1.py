'''
ROLL - 16MA20020
NAME - HIMANSHU

ASSIGNMENT NUMBER - ** 1 **  
					Implement the Find-S Algorithm for Concept Learning

OPTIONAL ARGUMENT:
--fname     input file name to take input(DEFAULT = "data1.csv"),
			Assume same size in input.

'''

import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--fname', default="data1.csv", type=str)
args = parser.parse_args()
fname = args.fname

data = list(csv.reader(open(fname)))
data1 = data
for i in range(20):
	for j in range(9):
		data1[i][j] = int(data[i][j])

for i in range(20):
	if(data[i][8]==1): hypothesis = data[i]

hypothesis[8]=-1

for i in range(20):
	for j in range(8):
		if hypothesis[j]!=-1 and hypothesis[j]!=data[i][j] and data[i][8]==1:
			hypothesis[j]=-1

count =0
for i in hypothesis:
	if i!=-1: count=count+1
print(count, end='')
for j in range(len(hypothesis)):
	if hypothesis[j]==0: 
		print(",", -1*(j+1), end='')
	elif hypothesis[j]==1:
		print(",", j+1, end='')
print("")
