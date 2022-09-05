import glob
from scipy.io import savemat
import numpy as np
import tensorflow as tf
import csv
import warnings
from hp_fcn_MEDIAN2 import hp_fcn
warnings.filterwarnings("ignore")

training = ['GMCNN', 'OPN', 'STTN']
testing = ['GMCNN', 'OPN', 'STTN']

video_ids = [] # video per testing
with open('../../../../../media/SSD_new/DATASET_AInpaint/dataset.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[1] == 'val':
                    video_ids.append(row[0])
print('video_ids: ', video_ids)

for m1 in range(len(training)):
  for m2 in range(len(testing)):
    scores  = []
    if training[m1] == 'OPN':
      model = 'models/checkpoint_hp_fcn_10_epoch_aug_OPN_double_FULL_tcn'
    else:
      model = 'models/checkpoint_hp_fcn_10_epoch_aug_' + training[m1] + '_double_FULL'
    print("MODELLO : ",model)

    if testing[m2] == 'OPN':
        path =  '../../../../../media/SSD_new/DATASET_AInpaint/'  + testing[m2] + '/432x240_compressed_postprocessed/'
    else:
        path =  '../../../../../media/SSD_new/DATASET_AInpaint/'  + testing[m2] + '/432x240_compressed/'
    print("PATH : ",path)
    for infile in video_ids:
        path_to_file = path + infile + '/'
        s = hp_fcn(path_to_file, model)
        tf.compat.v1.reset_default_graph()
        for i in range(len(s)):
           scores.append(s[i])
    print(scores)
    print("OUTPUT: ",'median_val/' + 'SPAN_' + training[m1] +'_10_epoch_aug_' + testing[m2] + '.mat')
    out = {"range": scores}
    output = 'median_val/' + 'SPAN_' + training[m1] +'_10_epoch_aug_' + testing[m2] + '.mat'
    savemat(output, out)
print("Successfully ended")

