import numpy as np
import pandas as pd


def main():
    data1 = pd.read_csv("phising.data", header=None)
    data = data1.values     # Convert from pandas dataframe to numpy

    X = data[:,:-1]
    y = data[:,-1]

    

if __name__=="__main__":
    main()