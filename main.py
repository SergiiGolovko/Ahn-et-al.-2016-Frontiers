import pandas as pd
import random
from sklearn.preprocessing import scale
from pandas import DataFrame
from split_utils import *
from compute import main_estimation
import matplotlib.pyplot as plt

def main():

    # read data from a file
    data = pd.read_table("input/cocaineData_frontiers.txt")

    # split data into labels and features
    y = data['DIAGNOSIS']
    X = data.drop(['DIAGNOSIS', 'subject'], axis=1)

    # !!! important: scale X for reproduction of exact results !!!
    # male is not scaled as it is a categorical variable
    male = X['Male']
    X = DataFrame(scale(X), index=X.index, columns=X.columns)
    X['Male'] = male

    # set random seed
    random.seed(RANDOM_SEED)

    # run estimation
    main_estimation(X, y)

    # show all figures
    plt.show()

if __name__ == '__main__':
    main()


