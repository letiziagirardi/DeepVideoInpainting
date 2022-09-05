from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import scipy.io
import warnings
warnings.filterwarnings("ignore")
#Get Data

import scipy.stats as st

#data = load_diabetes()
#X, y_ = data.data, data.target

mat1 = scipy.io.loadmat('../median_val/SPAN_OPN_10_epoch_aug_STTN.mat')


data11 = mat1['range']

x, y = sns.distplot(data11, label='OPN-STTN').get_lines()[0].get_data()
mdic = {"x": x, "y": y}


scipy.io.savemat("../DC_median/OS.mat", mdic)
print("fine")
