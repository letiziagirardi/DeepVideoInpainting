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

mat1 = scipy.io.loadmat('../Li_finetuned_videoInpainting/median_val/SPAN_GMCNN_75_10_epoch_aug_GMCNN.mat')


data11 = mat1['range']
#Organize Data
SR_y1 = pd.Series(data11[0,:], name="SPAN")
#Plot Data
fig, ax = plt.subplots()
sns.distplot(SR_y1, bins=25, color="r", ax=ax)
x, y = sns.distplot(data11[0,:], label='GMCNN-OPN').get_lines()[0].get_data()
mdic = {"x": x, "y": y}


scipy.io.savemat("DC_median/GO.mat", mdic)
