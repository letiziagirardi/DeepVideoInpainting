from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sn; sn.set()
import pandas as pd
import scipy.io
import numpy as np
import warnings
import scipy.stats as st
warnings.filterwarnings("ignore")

data = scipy.io.loadmat("class_mat_results_pesi_originali.mat") #class_mat_results.mat")
x = data['class']
print(x)
#y = data['y']
#print(y)
