from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sn; sn.set()
import pandas as pd
import scipy.io
import numpy as np
import warnings
import scipy.stats as st
warnings.filterwarnings("ignore")

#tecnica-pesi_fine_tuned
#array = np.asarray([[0.81,0.065,0.13],[0.16, 0.74, 0.097],[0.23, 0.19, 0.58]])
#TCN-pesi_fine_tuned
#array = np.asarray([[0.81,0.032,0.16],[0.064, 0.9, 0.03],[0.29, 0.097, 0.62]])
#pristine-pesi_fine_tuned t=124.92
#array = np.asarray([[0.55,0.032,0.0645, 0.35],[0.0323, 0.742, 0.0323,0.19],[0.194, 0.0645, 0.484, 0.258],[0.0,0.0,0.0,1.0]])
#pristine-pesi_fine_tuned t=5.20
#array = np.asarray([[0.68,0.032,0.0967, 0.19],[0.0323, 0.87, 0.0323,0.065],[0.258, 0.097, 0.55, 0.097],[0.0645,0.032,0.032,0.87]])


################################


#tecnica-pesi_no_fine_tuned
#array = np.asarray([[1.00,0.0,0.0],[0.0323, 0.97, 0.0],[0.258, 0.0, 0.77]])
#TCN-pesi_no_fine_tuned
#array = np.asarray([[1.00,0.0,0.0],[0.0323, 0.97, 0.0],[0.258, 0.0, 0.77]])
#pristine-pesi_no_fine_tuned
array = np.asarray([[1.0,0.0,0.0, 0.0],[0.0323, 0.97, 0.0,0.0],[0.23, 0.31, 0.77, 0.0],[0.032,0.032,0.0, 0.90]])

#3x3 matrix
#df_cm = pd.DataFrame(array, index = ['GMCNN', 'OPN', 'STTN'], columns = ['GMCNN', 'OPN', 'STTN'])

#4x4 matrix
df_cm = pd.DataFrame(array, index = ['GMCNN', 'OPN', 'STTN','PRISTINE'], columns = ['GMCNN', 'OPN', 'STTN', 'PRISTINE'])

plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, annot_kws={"fontsize":15})
plt.show()
