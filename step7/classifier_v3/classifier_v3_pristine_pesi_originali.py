import tensorflow as tf
from hp_fcn_PRISTINE_MEDIAN import calculatePixels
from hp_fcn_PRISTINE_MEDIAN import hp_fcn
import numpy as np
import scipy.io
import csv
import glob
import PIL
import tensorflow.compat.v1 as tfc
from skimage.io import imread, imshow
from skimage.transform import resize
tfc.disable_v2_behavior()


def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return np.expand_dims(mask, axis=2)

def TCN_scores(full_path):
    full_path = full_path + '*'
    files = sorted(list(glob.glob(full_path)))
    scores_TCN = []
    mask = create_circular_mask(240, 432, center=None, radius=50)
    mask = np.repeat(mask, 3, axis=2)
    for f in files:
        data_TCN = np.asarray(PIL.Image.open(f))
        data_TCN_c = tf.cast(data_TCN, dtype=tf.complex64)
        data_TCN_ct = tf.transpose(data_TCN_c)
        data_TCN_ct_fft = tf.signal.fft2d(data_TCN_ct)
        data_TCN_c_fft = tf.transpose(data_TCN_ct_fft)
        data_TCN_filtered = tf.multiply(data_TCN_c_fft, mask)
        data_TCN_filtered_t = tf.transpose(data_TCN_filtered)
        res_TCN = tf.math.real(tf.transpose(tf.signal.ifft2d(data_TCN_filtered_t)))
        scores_TCN.append((tf.math.reduce_min(tf.math.reduce_variance(
                            res_TCN, axis=[0, 1]))/tf.reduce_max(
                            tf.math.reduce_variance(res_TCN, axis=[0, 1]))).eval(session=tf.compat.v1.Session()))#.numpy())
    return scores_TCN


video_ids = [] # video per testing
with open('../../../../../media/SSD_new/DATASET_AInpaint/dataset.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[1] == 'test':
                    video_ids.append(row[0])
print('video_ids: ', video_ids)

class_mat = [[], [], [], []]
#class_inpainters = ['GMCNN', 'OPN', 'STTN']
#class_inpainters = ['GMCNN', 'OPN', 'STTN', 'PRISTINE']
#class_inpainters = ['PRISTINE']
class_inpainters = ['GMCNN']
#lista dei modelli di punto 2
list_models = [['DC_median/GG.mat', 'DC_median/GO.mat', 'DC_median/GS.mat'],
#               ['DC_median/OG_tcn.mat', 'DC_median/OO_tcn.mat', 'DC_median/OS_tcn.mat'],
                ['DC_median/OG.mat', 'DC_median/OO.mat', 'DC_median/OS.mat'],
                ['DC_median/SG.mat', 'DC_median/SO.mat', 'DC_median/SS.mat']]
#list_models = [['DC_median_montibeller/GG.mat', 'DC_median_montibeller/GO.mat', 'DC_median_montibeller/GS.mat'],
#              ['DC_median_montibeller/OG.mat', 'DC_median_montibeller/OO.mat', 'DC_median_montibeller/OS.mat'],
#              ['DC_median_montibeller/SG.mat', 'DC_median_montibeller/SO.mat', 'DC_median_montibeller/SS.mat']]

p_value_TCN = scipy.io.loadmat('score_TCN/TCN.mat')
p_value_noTCN = scipy.io.loadmat('score_TCN/noTCN.mat')
#tf.compat.v1.flags.DEFINE_string('data_dir', './data/full/?/jpg75/TOG/', 'path to dir')
#tf.compat.v1.flags.DEFINE_string('restore', None, 'Explicitly restore checkpoint')
c_id = 0
for c in class_inpainters:
    for v_id in video_ids:
        # if c == 'OPN':
        #     path_to_file =  '../../../../../media/SSD_new/DATASET_AInpaint/input_frames/432x240_compressed_postprocessed/' + v_id + '/'
        # else:
        #     path_to_file =  '../../../../../media/SSD_new/DATASET_AInpaint/input_frames/432x240_compressed/' + v_id + '/'
#        path_to_file =  '../../../../../media/SSD_new/DATASET_AInpaint/OPN/432x240_compressed_postprocessed/' + v_id + '/'
        if c == 'OPN':
            path_to_file =  '../../../../../media/SSD_new/DATASET_AInpaint/'  + c + '/432x240_compressed_postprocessed/' + v_id + '/'
        elif c ==  'PRISTINE':
            path_to_file =  '../../../../../media/SSD_new/DATASET_AInpaint/input_frames/432x240_compressed/' + v_id + '/'
        else:
            path_to_file =  '../../../../../media/SSD_new/DATASET_AInpaint/'  + c + '/432x240_compressed/' + v_id + '/'
    #    path_to_file =  '../../../../media/SSD_new/DATASET_AInpaint/'  + c + '/432x240_compressed_postprocessed/' + v_id + '/'
        #double
        decision_GMCNN = [[0],[0],[0]]
        decision_OPN = [[0],[0],[0]]
        decision_STTN = [[0],[0],[0]]
        decision_prist = [[0],[0],[0]]

        decision = ([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
        decision_GMCNN = [[0],[0],[0]]
        decision_OPN = [[0],[0],[0]]
        decision_STTN = [[0],[0],[0]]
        decision_prist = [[0],[0],[0]]
        print(path_to_file)
        #compute numero medio di pixls_video > 0.5
#        pixels_video_GMCNN = calculatePixels(path_to_file, 'model_montibeller/checkpoint_hp_fcn_5_epoch_aug_GMCNN/trained_with_jpg75')
        pixels_video_GMCNN = calculatePixels(path_to_file, 'models/checkpoint_hp_fcn_10_epoch_aug_GMCNN_double_FULL')
        tf.compat.v1.reset_default_graph()
        pixels_video_STTN = calculatePixels(path_to_file,'models/checkpoint_hp_fcn_10_epoch_aug_STTN_double_FULL')
#        pixels_video_STTN = calculatePixels(path_to_file, 'model_montibeller/checkpoint_hp_fcn_10_epoch_aug_OPN/trained_with_jpg75')
        tf.compat.v1.reset_default_graph()
        pixels_video_OPN = calculatePixels(path_to_file,'models/checkpoint_hp_fcn_10_epoch_aug_OPN_double_FULL')
#        pixels_video_OPN = calculatePixels(path_to_file, 'model_montibeller/checkpoint_hp_fcn_10_epoch_aug_STTN/trained_with_jpg96')
        tf.compat.v1.reset_default_graph()
        print('\033[91mNumero pixels_video > 0.5: \033[0m', pixels_video_GMCNN, pixels_video_STTN,  pixels_video_OPN)

        if pixels_video_GMCNN < 10.59 or pixels_video_STTN < 10.59 or pixels_video_OPN < 10.59:
            print("\033[91m video non modificato \033[0m")
#            print('Numero pixels_video > 0.5: ', pixels_video_GMCNN, pixels_video_STTN,  pixels_video_OPN)
            class_mat[c_id].append('PRISTINE')

            mdic = {"class": class_mat}
            scipy.io.savemat("class_mat_results_TCN_pristine_pesi_originali_prova.mat", mdic)
        else:
            scores_GMCNN = hp_fcn(path_to_file, 'models/checkpoint_hp_fcn_10_epoch_aug_GMCNN_double_FULL')
    #        scores_GMCNN = hp_fcn(path_to_file, 'model_montibeller/checkpoint_hp_fcn_5_epoch_aug_GMCNN/trained_with_jpg75')
            tf.compat.v1.reset_default_graph()
    #        scores_OPN = hp_fcn(path_to_file, 'model_montibeller/checkpoint_hp_fcn_10_epoch_aug_OPN/trained_with_jpg75')
            scores_OPN = hp_fcn(path_to_file, 'models/checkpoint_hp_fcn_10_epoch_aug_OPN_double_FULL_tcn')
            tf.compat.v1.reset_default_graph()
    #        scores_STTN = hp_fcn(path_to_file, 'model_montibeller/checkpoint_hp_fcn_10_epoch_aug_STTN/trained_with_jpg96')
            scores_STTN = hp_fcn(path_to_file, 'models/checkpoint_hp_fcn_10_epoch_aug_STTN_double_FULL')

            scores_TCN = TCN_scores(path_to_file)
            tf.compat.v1.reset_default_graph()
            scores = [scores_GMCNN, scores_OPN, scores_STTN]
            decision = ([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
            final_decision = [[], []]
            print('decision pre tcn: ', decision)

            for score_TCN in scores_TCN:
                ind1TCN = 0
                ind2TCN = 0
                for x in range(1, len(p_value_TCN['x'][0])-1):
                    if p_value_TCN['x'][0, x-1] < score_TCN < p_value_TCN['x'][0, x+1]:
                        ind1TCN = x-1
                        ind2TCN = x+1
                if ind1TCN != 0:
                    #decision[1][0:3] += (p_value_TCN['y'][0, ind1TCN]/np.sum(p_value_TCN['y'][0]) +
                    #                p_value_TCN['y'][0, ind2TCN]/np.sum(p_value_TCN['y'][0]))/2
                    decision[1][3] += (p_value_TCN['y'][0, ind1TCN]/np.sum(p_value_TCN['y'][0]) +
                                    p_value_TCN['y'][0, ind2TCN]/np.sum(p_value_TCN['y'][0]))/2
                else:
                    #decision[1][0:3] += 0
                    decision[1][3] += 0

            for score_TCN in scores_TCN:
                ind1noTCN = 0
                ind2noTCN = 0
                for x in range(1, len(p_value_noTCN['x'][0])-1):
                    if p_value_noTCN['x'][0, x-1] < score_TCN < p_value_noTCN['x'][0, x+1]:
                        ind1noTCN = x-1
                        ind2noTCN = x+1
                if ind1noTCN != 0:
                    #decision[0][0:3] += (p_value_noTCN['y'][0, ind1noTCN]/np.sum(p_value_noTCN['y'][0]) +
                    #                p_value_noTCN['y'][0, ind2noTCN]/np.sum(p_value_noTCN['y'][0]))/2
                    #decision[2][0:3] += (p_value_noTCN['y'][0, ind1noTCN]/np.sum(p_value_noTCN['y'][0]) +
                    #                p_value_noTCN['y'][0, ind2noTCN]/np.sum(p_value_noTCN['y'][0]))/2
                    decision[0][3] += (p_value_noTCN['y'][0, ind1noTCN]/np.sum(p_value_noTCN['y'][0]) +
                                    p_value_noTCN['y'][0, ind2noTCN]/np.sum(p_value_noTCN['y'][0]))/2
                    decision[2][3] += (p_value_noTCN['y'][0, ind1noTCN]/np.sum(p_value_noTCN['y'][0]) +
                                    p_value_noTCN['y'][0, ind2noTCN]/np.sum(p_value_noTCN['y'][0]))/2
                else:
                    #decision[0][0:3] += 0
                    #decision[2][0:3] += 0
                    decision[0][3] += 0
                    decision[2][3] += 0
    ###########################
            print('decision post tcn: ', decision)
            for i in range(len(scores)):
                p_values_OPN = scipy.io.loadmat(list_models[i][1])
                p_values_STTN = scipy.io.loadmat(list_models[i][2])
                p_values_GMCNN = scipy.io.loadmat(list_models[i][0])
                for j in range(len(scores[i])):
                    ind1o = 0
                    ind2o = 0
                    for x in range(1, len(p_values_OPN['x'][0])-1):
                        if p_values_OPN['x'][0, x-1] < scores[i][j] < p_values_OPN['x'][0, x+1]:
                            ind1o = x-1
                            ind2o = x+1

                    ind1s = 0
                    ind2s = 0
                    for x in range(1, len(p_values_STTN['x'][0])-1):
                        if p_values_STTN['x'][0, x-1] < scores[i][j] < p_values_STTN['x'][0, x+1]: #and score<tau_TCN
                            ind1s = x-1
                            ind2s = x+1
                    ind1g = 0
                    ind2g = 0
                    for x in range(1, len(p_values_GMCNN['x'][0])-1):
                        if p_values_GMCNN['x'][0, x-1] < scores[i][j] < p_values_GMCNN['x'][0, x+1]: #and score<tau_TCN
                            ind1g = x-1
                            ind2g = x+1
                    if ind1g != 0:
                        decision[0][i] += (p_values_GMCNN['y'][0,ind1g]/np.sum(p_values_GMCNN['y'][0]) +
                                            p_values_GMCNN['y'][0, ind2g]/np.sum(p_values_GMCNN['y'][0]))/2
                    else:
                        decision[0][i] += 0
                    if ind1o != 0:
                        decision[1][i] += (p_values_OPN['y'][0,ind1o]/np.sum(p_values_OPN['y'][0]) +
                                            p_values_OPN['y'][0,ind2o]/np.sum(p_values_OPN['y'][0]))/2#1
                    else:
                        decision[1][i] += 0
                    if ind1s != 0:
                        decision[2][i] += (p_values_STTN['y'][0,ind1s]/np.sum(p_values_STTN['y'][0])
                                            + p_values_STTN['y'][0, ind2s]/np.sum(p_values_STTN['y'][0]))/2#1
                    else:
                        decision[2][i] += 0
                    decision[3][i] += 0
                decision1 = np.asarray(decision)
            print(decision)
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
            scipy.io.savemat("class_mat_results_TCN_pristine_pesi_originali_prova.mat", mdic)
    c_id += 1

