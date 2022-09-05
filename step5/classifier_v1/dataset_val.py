from os import listdir
from os.path import join
import random
from numpy import random as random_np
from PIL import Image
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import csv
from pprint import pprint
import numpy as np

# from torchvision.datasets.vision import VisionDataset
_root_dir = '/media/SSD_new/DATASET_AInpaint/'
_pristine_dir = 'input_frames/'#'original_clips'
_mask_dir = 'input_masks/resized_432x240'#'masks'
_resolutions = ('432x240', '1280x720', '864x480')
_manipulations = ('GMCNN', 'OPN', 'STTN')
#_compression = 'output_frames/LD'
#_specres = ('double_compression', 'single_compression', 'double_compression_postprocessed', 'single_compression_postprocessed')
_specres = ('432x240_compressed', '432x240_compressed_postprocessed')

def zeros_like(x):
    """
    Crea una maschera per l'immagine pristine
    :param x: PIL.Image or numpy array
    :return: all zero PIL.Image or numpy array
    """
    if isinstance(x, Image.Image):
        return Image.new(mode='L', size=x.size, color=0)
    elif isinstance(x, str):
        return join(_root_dir, _mask_dir, 'all-zeros.png')  # debug
    else:
        return deepcopy(x) * 0


def collate_fn(batch):
    if len(batch[0]) == 3:  # return_masks is True
        coll_images, coll_labels, coll_masks = [], [], []
        for images, labels, masks in batch:
            coll_images += images
            coll_labels += labels
            coll_masks += masks
        return coll_images, coll_labels, coll_masks

    else:  # return_masks is False
        coll_images, coll_labels = [], []
        for images, labels in batch:
            coll_images += images
            coll_labels += labels
        return coll_images, coll_labels




class VideoDataset(Dataset):
    def __init__(self, video_ids, manipulations, specres, resolutions, shuffle, read_fn, max_frames, return_masks,
                 img_transform=None, label_transform=None, mask_transform=None, seed=0):
        # random seed
        random.seed(seed)
        random_np.seed(seed)

        # videos
        self.video_ids = video_ids
        self.video_num = len(self.video_ids)
        print('Videos:', self.video_num)

        # manipulations and resolutions
        self.manipulations = manipulations
        print('Manipulations:', self.manipulations)
        self.resolutions = resolutions
        print('Resolutions:', self.resolutions)
        self.specres = specres
        print('specres:', self.specres)

        # number of frames per video
        self.frame_num = [0 for _ in range(self.video_num)]
        self.init_frame_num()

        # frame indexes
        self.frame_indexes = [[] for _ in range(self.video_num)]
        self.shuffle = shuffle
        if max_frames is None or max_frames == 'max':
            max_frames = max(self.frame_num)
        elif max_frames == 'min':
            max_frames = min(self.frame_num)
        elif max_frames == 'mean' or max_frames == 'avg' or max_frames == 'average':
            max_frames = sum(self.frame_num) // len(self.frame_num)
        elif not isinstance(max_frames, int):
            raise ValueError('max_frames must be integer or one of [None, "max", "min", "mean", "avg", "average"]')
        self.max_frames = max_frames
        print('Max frames:', self.max_frames)
        for video_index in range(self.video_num):
            self.reset_frames(video_index)

        # dataset length
        self.total_len = self.video_num#sum(self.frame_num)
        print('Total len:', self.total_len)

        # avoid reading masks for detection problems
        self.return_masks = return_masks

        # input: image path, output: image
        self.read_fn = read_fn

    def init_frame_num(self):
        """
        Legge il numero di frame per ogni video utilizzando la directory delle maschere. Viene eseguita solo una volta.
        """
        self.frame_num = []
        for v in self.video_ids:
            files = listdir(join(_root_dir, self.manipulations, self.specres, v))#_root_dir, _mask_dir, v))##_root_dir, _mask_dir, 'original', v))
            n = len([file for file in files if file.endswith('.png')])
            self.frame_num.append(n)
        self.frame_num = tuple(self.frame_num)


    def reset_frames(self, video_index):
        """
        Resetta le liste degli indici dei frame da cui pescare una volta esaurite.
        E' necessario perché non tutti i video esauriscono la lista delle frame insieme al termine dell'epoca.
        :param video_index: video index
        """
        self.frame_indexes[video_index] = list(range(self.frame_num[video_index]))
        if self.shuffle:
            random.shuffle(self.frame_indexes[video_index])
        if self.max_frames:
            self.frame_indexes[video_index] = self.frame_indexes[video_index][:self.max_frames]

    def get_images_labels_masks_mod(self, video_id, frames):
        """
        Calcola i path dei file e legge le immagini, le maschere (se richiesto) e genera le label di detection.
        I dati sono paired: in ogni batch avremo la frame pristine e tutte le versioni manipolate.
        :param video_id: name of the video (str)
        :param frame_num:  frame index (int)
        :return: paired images, labels, masks tuple
        """

        # TODO: transformations
        # TODO: consentire lettura unpaired

        images, labels, masks = [], [], []
        for frame in range(0, frames):
            frame_num = '%04d.png' % frame
            images.append(self.read_fn(join(_root_dir, _pristine_dir, self.specres, video_id, frame_num)))#'frame_resized', 'frame',  frame_num)))# 'resized_' + resolution, video_id, frame_num)))
            labels.append(0)
            if self.return_masks:
                masks.append(zeros_like(images[-1]))

                mask = self.read_fn(join(_root_dir, _mask_dir, video_id, frame_num))# 'masks_resized', 'mask',  frame_num))#, 'resized_' + resolution, video_id, frame_num))  # read just once for all the manipulations
            #for manipulation in self.manipulations:
            images.append(self.read_fn(join(_root_dir, self.manipulations,  self.specres, video_id, frame_num)))
            labels.append(1)
            if self.return_masks:
                masks.append(deepcopy(mask))
        if self.return_masks:
            return images, labels, masks
        else:
            return images, labels

    def __getitem__(self, index):
        """
        In base all'indice randomico, seleziona un video.
        In un batch possono capitare più frame dello stesso video, ma durante l'epoca tutti i video dovrebbero essere
        selezionati lo stesso numero di volte.
        Una volta selezionato il video, sceglie un frame dagli indici disponibili e infine
        carica l'immagine pristine e le versioni manipolate.
        :param index: indice fornito dal dataloader (int)
        :return: vedi get_images_labels_masks
        """
        '''
        print('index: ', index)
        i = index % self.video_num
        #for i in range(0, len(self.frame_num)):
        selected_video = self.video_ids[i]
        for j in range(0, self.frame_num[i]):
                if len(self.frame_indexes[i]) == 0:
                    self.reset_frames(i)  # reset video frame list if empty
                #selected_frame = self.frame_indexes[j].pop()
                #print('selected_frame: ', selected_frame)
                return self.get_images_labels_masks(selected_video, j)
                #return self.get_images_labels_masks(selected_video, selected_frame)
                ###############
        # video selection
        print('index: ', index)
        selected_video_ind = index % self.video_num
        selected_video = self.video_ids[selected_video_ind]
        print('selected_video_ind: ', selected_video)

        # frame selection
        if len(self.frame_indexes[selected_video_ind]) == 0:
            self.reset_frames(selected_video_ind)  # reset video frame list if empty
        selected_frame = self.frame_indexes[selected_video_ind].pop()

        # read data from disk
        return self.get_images_labels_masks_mod(selected_video, selected_frame)
        '''
        # video selection
        selected_video_ind = index #% self.video_num
        selected_video = self.video_ids[selected_video_ind]

        # frame selection
        if len(self.frame_indexes[selected_video_ind]) == 0:
            self.reset_frames(selected_video_ind)  # reset video frame list if empty
        selected_frame = self.frame_num[selected_video_ind]#self.frame_indexes[selected_video_ind].pop()

        # read data from disk
        return self.get_images_labels_masks_mod(selected_video, selected_frame)

    def __len__(self):
        return self.total_len


#if __name__ == "__main__":
def read_frame_masks_fine(id_manu, id_specres):

    print("VALIDATION DATASET")
    video_ids = []
    with open('/media/SSD_new/DATASET_AInpaint/dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[1] == 'val':
                video_ids.append(row[0])
    print(video_ids)
    idx = random.randint(0, 2)
    while id_manu == idx:
        idx = random.randint(0, 2)
    manipulations = _manipulations[idx]
    resolutions = _resolutions[:1]
    specres = _specres[id_specres]
    shuffle = False
    return_masks = True

    max_frames = 'min' #'max' or int

    read_fn = lambda x: x  # debug
    dataset = VideoDataset(video_ids, manipulations, specres, resolutions, shuffle, read_fn, max_frames, return_masks)

    paired_batch_size = 19000#6000

    num_workers = 0
    dataloader = DataLoader(dataset, batch_size=paired_batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)
    for i, (images, labels, masks) in enumerate(dataloader):
        print('\n\nbatch:', i)
    instance_num = len(images)
    dataset = tf.compat.v1.data.Dataset.from_tensor_slices((images, masks))
    return dataset, instance_num
