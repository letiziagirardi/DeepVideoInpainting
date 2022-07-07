import tensorflow as tf
from hp_fcn_MEDIAN import hp_fcn
import numpy as np
import scipy.io
import csv

video_ids = []
with open('../DATASET_AInpaint/dataset.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[1] == 'test':
                    video_ids.append(row[0])
print('video_ids: ', video_ids)

class_mat = [[], [], []]
class_inpainters = ['GMCNN', 'OPN', 'STTN']
#lista dei modelli di punto 2
list_models = [['DC_median/GG.mat', 'DC_median/GO.mat', 'DC_median/GS.mat'],
               ['DC_median/OG.mat', 'DC_median/OO.mat', 'DC_median/OS.mat'],
               ['DC_median/SG.mat', 'DC_median/SO.mat', 'DC_median/SS.mat']]

#tf.compat.v1.flags.DEFINE_string('data_dir', './data/full/?/jpg75/TOG/', 'path to dir')
#tf.compat.v1.flags.DEFINE_string('restore', None, 'Explicitly restore checkpoint')
c_id = 0
for c in class_inpainters:
    for v_id in video_ids:
        path_to_file = '../DATASET_V1/' + c + '/output_frames/LD/double_compression/' + v_id + '/' #cambialo
        #double
        decision_GMCNN = [[0],[0],[0]]
        decision_OPN = [[0],[0],[0]]
        decision_STTN = [[0],[0],[0]]
        decision_prist = [[0],[0],[0]]
        #cambia path e adatta codice di hp_fcn -- la funzione main -- perche' prenda in input il path ai frame del video e i pesi
        #la funzione hp_fcn deve ritornare un set di scores non set di mashere

        scores_GMCNN = hp_fcn(path_to_file,
                    '../Li_finetuned_videoInpainting/models_precedent/checkpoint_hp_fcn_5_epoch_aug_GMCNN/trained_with_jpg75')
        tf.compat.v1.reset_default_graph()
        scores_OPN = hp_fcn(path_to_file,
                    '../Li_finetuned_videoInpainting/models_precedent/checkpoint_hp_fcn_10_epoch_aug_OPN/trained_with_jpg75')
        tf.compat.v1.reset_default_graph()
        scores_STTN = hp_fcn(path_to_file,
                    '../Li_finetuned_videoInpainting/models_precedent/checkpoint_hp_fcn_10_epoch_aug_STTN/trained_with_jpg96')
        tf.compat.v1.reset_default_graph()
        scores = [scores_GMCNN, scores_OPN, scores_STTN]
        decision = ([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])
        final_decision = [[],[]]

        for i in range(len(scores)):
            p_values_OPN = scipy.io.loadmat(list_models[i][1])
            p_values_STTN = scipy.io.loadmat(list_models[i][2])
            p_values_GMCNN = scipy.io.loadmat(list_models[i][0])
            for j in range(len(scores[i])):
                ind1o = 0
                ind2o = 0
                for x in range(1,len(p_values_OPN['x'][0])-1):
                    if p_values_OPN['x'][0, x-1] < scores[i][j] < p_values_OPN['x'][0, x+1]: #and score>tau_TCN -- da aggiungere quando mettiamo TCN
                        #cerca indici
                        ind1o = x-1
                        ind2o = x+1

                ind1s = 0
                ind2s = 0
                for x in range(1,len(p_values_STTN['x'][0])-1):
                    if p_values_STTN['x'][0, x-1] < scores[i][j] < p_values_STTN['x'][0, x+1]: #and score<tau_TCN
                        #cerca indici
                        ind1s = x-1
                        ind2s = x+1

                ind1g = 0
                ind2g = 0
                for x in range(1,len(p_values_GMCNN['x'][0])-1):
                    if p_values_GMCNN['x'][0, x-1] < scores[i][j] < p_values_GMCNN['x'][0, x+1]: #and score<tau_TCN
                        #cerca indici
                        ind1g = x-1
                        ind2g = x+1

                if ind1g != 0:
                    decision[0][i] += (p_values_GMCNN['y'][0,ind1g]/np.sum(p_values_GMCNN['y'][0]) +
                                   p_values_GMCNN['y'][0, ind2g]/np.sum(p_values_GMCNN['y'][0]))/2
                else:
                    decision[0][i] += 0
                if ind1o!=0:
                    decision[1][i] += (p_values_OPN['y'][0,ind1o]/np.sum(p_values_OPN['y'][0]) +
                                   p_values_OPN['y'][0,ind2o]/np.sum(p_values_OPN['y'][0]))/2
                else:
                    decision[1][i] += 0
                if ind1s != 0:
                    decision[2][i] += (p_values_STTN['y'][0,ind1s]/np.sum(p_values_STTN['y'][0])
                                   + p_values_STTN['y'][0, ind2s]/np.sum(p_values_STTN['y'][0]))/2
                else:
                    decision[2][i] +=0
                decision[3][i] += 0


            decision1 = np.asarray(decision)
        #print(decision1)
        decision1 = decision1

        ##CLASSIFICATORE PER MAJORITY
        max_by_column = [(np.sum(decision1[0,:])), np.sum(decision1[1,:]), np.sum(decision1[2,:])]
        if np.max(max_by_column) == max_by_column[0]:
                print('GMCNN')
                print('score: ',max_by_column[0])
                class_mat[c_id].append('GMCNN')
        elif np.max(max_by_column) == max_by_column[1]:
                print('OPN')
                print('score: ',max_by_column[1])
                class_mat[c_id].append('OPN')
        elif np.max(max_by_column) == max_by_column[2]:
                print('STTN')
                print('score: ',max_by_column[2])
                class_mat[c_id].append('STTN')

        mdic = {"class": class_mat}
        scipy.io.savemat("class_mat_v10.mat", mdic)
    c_id += 1
