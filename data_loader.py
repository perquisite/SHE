import scipy.io as sio
import h5py
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch
from pycocotools.coco import COCO
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
import random
import os
def get_noisylabels(labels, noise_ratio, noise_mode, seed):
    from to_seed import to_seed
    to_seed(seed)
    noise_label = []
    classes = np.unique(labels)
    class_num = classes.shape[0]

    inx = np.arange(class_num)
    np.random.shuffle(inx)
    transition = {i: i for i in range(class_num)}
    half_num = int(class_num // 2)
    for i in range(half_num):
        transition[inx[i]] = int(inx[half_num + i])
    noise_label = []
    data_num = labels.shape[0]
    idx = list(range(data_num))
    random.shuffle(idx)
    num_noise = int(noise_ratio * data_num)
    noise_idx = idx[:num_noise]
    for i in range(data_num):
        if i in noise_idx:
            if noise_mode == 'sym':
                noiselabel = int(random.randint(min(classes), max(classes)))
                noise_label.append(noiselabel)
            elif noise_mode == 'asym':
                noiselabel = transition[labels[i]]
                noise_label.append(noiselabel)
        else:
            noise_label.append(int(labels[i]))
    return np.array(noise_label).reshape((-1,1))


def load_deep_features(data_name):
    valid_data = True
    np.random.seed(1)
    if data_name == 'xmedia':
        path = '/home/qinyang/windows/sda1/ProjectsOfQy/NoisyLabel/PRT_projects/MARS_DRSL/datasets/XMedia/XMediaFeatures.mat'
        all_data = sio.loadmat(path)
        I_te_CNN = all_data['I_te_CNN'].astype('float32')   # Features of test set for image data, CNN feature
        I_tr_CNN = all_data['I_tr_CNN'].astype('float32')   # Features of training set for image data, CNN feature
        T_te_BOW = all_data['T_te_BOW'].astype('float32')   # Features of test set for text data, BOW feature
        T_tr_BOW = all_data['T_tr_BOW'].astype('float32')   # Features of training set for text data, BOW feature
        V_te_CNN = all_data['V_te_CNN'].astype('float32')   # Features of test set for video(frame) data, CNN feature
        V_tr_CNN = all_data['V_tr_CNN'].astype('float32')   # Features of training set for video(frame) data, CNN feature
        A_te = all_data['A_te'].astype('float32')           # Features of test set for audio data, MFCC feature
        A_tr = all_data['A_tr'].astype('float32')           # Features of training set for audio data, MFCC feature
        d3_te = all_data['d3_te'].astype('float32')         # Features of test set for 3D data, LightField feature
        d3_tr = all_data['d3_tr'].astype('float32')         # Features of training set for 3D data, LightField feature

        teImgCat = all_data['teImgCat'].reshape([-1,1]).astype('int64') # category label of test set for image data
        trImgCat = all_data['trImgCat'].reshape([-1,1]).astype('int64') # category label of training set for image data
        teVidCat = all_data['teVidCat'].reshape([-1,1]).astype('int64') # category label of test set for video(frame) data
        trVidCat = all_data['trVidCat'].reshape([-1,1]).astype('int64') # category label of training set for video(frame) data
        teTxtCat = all_data['teTxtCat'].reshape([-1,1]).astype('int64') # category label of test set for text data
        trTxtCat = all_data['trTxtCat'].reshape([-1,1]).astype('int64') # category label of training set for text data
        te3dCat = all_data['te3dCat'].reshape([-1,1]).astype('int64')   # category label of test set for 3D data
        tr3dCat = all_data['tr3dCat'].reshape([-1,1]).astype('int64')   # category label of training set for 3D data
        teAudCat = all_data['teAudCat'].reshape([-1,1]).astype('int64') # category label of test set for audio data
        trAudCat = all_data['trAudCat'].reshape([-1,1]).astype('int64') # category label of training set for audio data


        train_data = [I_tr_CNN, T_tr_BOW, A_tr, d3_tr, V_tr_CNN]
        test_data = [I_te_CNN[0: 500], T_te_BOW[0: 500], A_te[0: 100], d3_te[0: 50], V_te_CNN[0: 87]]
        valid_data = [I_te_CNN[500::], T_te_BOW[500::], A_te[100::], d3_te[50::], V_te_CNN[87::]]
        train_labels = [trImgCat, trTxtCat, trAudCat, tr3dCat, trVidCat]
        test_labels = [teImgCat[0: 500], teTxtCat[0: 500], teAudCat[0: 100], te3dCat[0: 50], teVidCat[0: 87]]
        valid_labels = [teImgCat[500::], teTxtCat[500::], teAudCat[100::], te3dCat[50::], teVidCat[87::]]

        retrieval_data, retrieval_labels = [], []
        for i in range(len(train_data)):
            train_data_tmp, valid_data_tmp, test_data_tmp = train_data[i], valid_data[i], test_data[i]
            train_label_tmp, valid_label_tmp, test_label_tmp = train_labels[i], valid_labels[i], test_labels[i]
            test_data_new, test_label_new = np.vstack((test_data_tmp, valid_data_tmp)), np.vstack((test_label_tmp, valid_label_tmp))
            test_data[i], test_labels[i] = test_data_new, test_label_new
            retrieval_data_new, retrieval_label_new = np.vstack((train_data_tmp, test_data_new)), np.vstack((train_label_tmp, test_label_new))
            retrieval_data.append(retrieval_data_new)
            retrieval_labels.append(retrieval_label_new)

    elif data_name == 'xmedianet': # label 1*n
        valid_len = 4000
        path = '/home/qinyang/windows/sda1/ProjectsOfQy/NoisyLabel/CrossNL/DSCMR/newData/xmedianet_deep_doc2vec_data.h5py'
        with h5py.File(path, 'r') as file:
            groups = list(file.keys())
        path = '/home/qinyang/windows/sda1/ProjectsOfQy/NoisyLabel/CrossNL/DSCMR/newData/XMediaNet5View_Doc2Vec.mat'
        all_data = sio.loadmat(path)
        all_train_data = all_data['train'][0]
        all_train_labels = all_data['train_labels'][0]
        all_valid_data = all_data['valid'][0]
        all_valid_labels = all_data['valid_labels'][0]
        all_test_data = all_data['test'][0]
        all_test_labels = all_data['test_labels'][0]

        train_data, valid_data, test_data, train_labels, valid_labels, test_labels = [],[],[],[],[],[]
        for i in range(5):
            train_data.append(all_train_data[i].astype('float32'))
            train_labels.append(all_train_labels[i].reshape([-1,1]).astype('int32'))
            valid_data.append(all_valid_data[i].astype('float32'))
            valid_labels.append(all_valid_labels[i].reshape([-1,1]).astype('int32'))
            test_data.append(all_test_data[i].astype('float32'))
            test_labels.append(all_test_labels[i].reshape([-1,1]).astype('int32'))
        retrieval_data, retrieval_labels = [], []
        for i in range(len(train_data)):
            train_data_tmp, valid_data_tmp, test_data_tmp = train_data[i], valid_data[i], test_data[i]
            train_label_tmp, valid_label_tmp, test_label_tmp = train_labels[i], valid_labels[i], test_labels[i]
            test_data_new, test_label_new = np.vstack((test_data_tmp, valid_data_tmp)), np.vstack((test_label_tmp, valid_label_tmp))
            test_data[i], test_labels[i] = test_data_new, test_label_new
            retrieval_data_new, retrieval_label_new = np.vstack((train_data_tmp, test_data_new)), np.vstack((train_label_tmp, test_label_new))
            retrieval_data.append(retrieval_data_new)
            retrieval_labels.append(retrieval_label_new)
 
    elif data_name == 'wiki_old':
        valid_len = 231
        path = '/home/qinyang/windows/sda1/ProjectsOfQy/NoisyLabel/PRT_projects/MARS_DRSL/datasets/Wiki/wiki.mat'
        all_data = sio.loadmat(path)
        img_train = all_data['train_imgs_deep']
        text_train = all_data['train_texts_doc']
        label_train = all_data['train_imgs_labels'].reshape([-1,1])

        img_test = all_data['test_imgs_deep']
        text_test = all_data['test_texts_doc']
        label_test = all_data['test_imgs_labels'].reshape([-1,1])

        img_val = img_test[0:valid_len]
        text_val = text_test[0:valid_len]
        label_val = label_test[0:valid_len]

        img_test = img_test[valid_len:]
        text_test = text_test[valid_len:]
        label_test = label_test[valid_len:]

        train_data = [img_train, text_train]
        test_data = [img_test, text_test]
        valid_data = [img_val, text_val]
        train_labels = [label_train, label_train]
        test_labels = [label_test, label_test]
        valid_labels =  [label_val, label_val]

        retrieval_data, retrieval_labels = [], []
        for i in range(len(train_data)):
            train_data_tmp, valid_data_tmp, test_data_tmp = train_data[i], valid_data[i], test_data[i]
            train_label_tmp, valid_label_tmp, test_label_tmp = train_labels[i], valid_labels[i], test_labels[i]
            test_data_new, test_label_new = np.vstack((test_data_tmp, valid_data_tmp)), np.vstack((test_label_tmp, valid_label_tmp))
            test_data[i], test_labels[i] = test_data_new, test_label_new
            retrieval_data_new, retrieval_label_new = np.vstack((train_data_tmp, test_data_new)), np.vstack((train_label_tmp, test_label_new))
            retrieval_data.append(retrieval_data_new)
            retrieval_labels.append(retrieval_label_new)
    elif data_name == 'nus':
        valid_len = 0
        path = '/home/qinyang/windows/sda1/ProjectsOfQy/NoisyLabel/CrossNL/DSCMR/newData/nus_wide_deep_doc2vec-corr-ae.h5py'
        with h5py.File(path, 'r') as file:
            groups = list(file.keys())
            img_train = file['train_imgs_deep'][:]
            text_train = file['train_texts'][:]
            label_train = file['train_imgs_labels'][:]

            img_val = file['valid_imgs_deep'][:]
            text_val = file['valid_texts'][:]
            label_val = file['valid_imgs_labels'][:]

            img_test = file['test_imgs_deep'][:]
            text_test = file['test_texts'][:]
            label_test = file['test_imgs_labels'][:]

            train_data = [img_train, text_train]
            test_data = [img_test, text_test]
            valid_data = [img_val, text_val]
            train_labels = [label_train.reshape([-1,1]), label_train.reshape([-1,1])]
            test_labels = [label_test.reshape([-1,1]), label_test.reshape([-1,1])]
            valid_labels =  [label_val.reshape([-1,1]), label_val.reshape([-1,1])]
        retrieval_data, retrieval_labels = [], []
        for i in range(len(train_data)):
            train_data_tmp, valid_data_tmp, test_data_tmp = train_data[i], valid_data[i], test_data[i]
            train_label_tmp, valid_label_tmp, test_label_tmp = train_labels[i], valid_labels[i], test_labels[i]
            test_data_new, test_label_new = np.vstack((test_data_tmp, valid_data_tmp)), np.vstack((test_label_tmp, valid_label_tmp))
            test_data[i], test_labels[i] = test_data_new, test_label_new
            retrieval_data_new, retrieval_label_new = np.vstack((train_data_tmp, test_data_new)), np.vstack((train_label_tmp, test_label_new))
            retrieval_data.append(retrieval_data_new)
            retrieval_labels.append(retrieval_label_new)
    elif data_name == 'mirflickr':
        path = '/home/qinyang/windows/sda1/ProjectsOfQy/NoisyLabel/PRT_projects/datasets/MIRFlickr.h5'
        with h5py.File(path, 'r') as file:
            groups = list(file.keys())
            img_train, text_train = file['ImgTrain'][:], file['TagTrain'][:]
            img_train_labels, text_train_labels = file['LabTrain'][:], file['LabTrain'][:]

            img_test, text_test = file['ImgQuery'][:], file['TagQuery'][:]
            img_test_labels, text_test_labels = file['LabQuery'][:], file['LabQuery'][:]

            img_retrieval, text_retrieval = file['ImgDataBase'][:], file['TagDataBase'][:]
            img_retrieval_labels, text_retrieval_labels = file['LabDataBase'][:], file['LabDataBase'][:]

            img_train, text_train = img_train.astype('float32'), text_train.astype('float32') 
            img_test, text_test = img_test.astype('float32'), text_test.astype('float32')
            img_retrieval, text_retrieval = img_retrieval.astype('float32'), text_retrieval.astype('float32')
        train_data = [img_train, text_train]
        train_labels = [img_train_labels, text_train_labels]
        test_data = [img_test, text_test]
        test_labels = [img_test_labels, text_test_labels]
        retrieval_data = [img_retrieval, text_retrieval]
        retrieval_labels = [img_retrieval_labels, text_retrieval_labels]
    elif data_name == 'mscoco':
        path = '/home/qinyang/windows/sda1/ProjectsOfQy/NoisyLabel/PRT_projects/datasets/MS-COCO.h5'
        with h5py.File(path, 'r') as file:
            groups = list(file.keys())
            img_train, text_train = file['ImgTrain'][:], file['TagTrain'][:]
            img_train_labels, text_train_labels = file['LabTrain'][:], file['LabTrain'][:]

            img_test, text_test = file['ImgQuery'][:], file['TagQuery'][:]
            img_test_labels, text_test_labels = file['LabQuery'][:], file['LabQuery'][:]

            img_retrieval, text_retrieval = file['ImgDataBase'][:], file['TagDataBase'][:]
            img_retrieval_labels, text_retrieval_labels = file['LabDataBase'][:], file['LabDataBase'][:]

            img_train, text_train = img_train.astype('float32'), text_train.astype('float32') 
            img_test, text_test = img_test.astype('float32'), text_test.astype('float32')
            img_retrieval, text_retrieval = img_retrieval.astype('float32'), text_retrieval.astype('float32')
        train_data = [img_train, text_train]
        train_labels = [img_train_labels, text_train_labels]
        test_data = [img_test, text_test]
        test_labels = [img_test_labels, text_test_labels]
        retrieval_data = [img_retrieval, text_retrieval]
        retrieval_labels = [img_retrieval_labels, text_retrieval_labels]
    elif data_name == 'iapr':
        path = '/home/qinyang/windows/sda1/ProjectsOfQy/NoisyLabel/PRT_projects/datasets/IAPR.h5'
        with h5py.File(path, 'r') as file:
            groups = list(file.keys())
            img_train, text_train = file['ImgTrain'][:], file['TagTrain'][:]
            img_train_labels, text_train_labels = file['LabTrain'][:], file['LabTrain'][:]

            img_test, text_test = file['ImgQuery'][:], file['TagQuery'][:]
            img_test_labels, text_test_labels = file['LabQuery'][:], file['LabQuery'][:]

            img_retrieval, text_retrieval = file['ImgDataBase'][:], file['TagDataBase'][:]
            img_retrieval_labels, text_retrieval_labels = file['LabDataBase'][:], file['LabDataBase'][:]

            img_train, text_train = img_train.astype('float32'), text_train.astype('float32') 
            img_test, text_test = img_test.astype('float32'), text_test.astype('float32')
            img_retrieval, text_retrieval = img_retrieval.astype('float32'), text_retrieval.astype('float32')
        train_data = [img_train, text_train]
        train_labels = [img_train_labels, text_train_labels]
        test_data = [img_test, text_test]
        test_labels = [img_test_labels, text_test_labels]
        retrieval_data = [img_retrieval, text_retrieval]
        retrieval_labels = [img_retrieval_labels, text_retrieval_labels]
    

    
    # for i in range(len(train_labels)):
    #     dir = '/home/qinyang/windows/sda1/ProjectsOfQy/NoisyLabel/PRT_projects/datasets/noisy_labels_streaming_media/'
    #     noise_path = dir + data_name + '_noisy_' + str(noise_ratio) + '_' + noise_mode + '_' + str(i) + '.h5'
    #     if os.path.exists(noise_path):
    #         with h5py.File(noise_path, 'r') as f:
    #             noisy_labels = f['noisy_labels'][:]
    #     else:
    #         noisy_labels = get_noisylabels(train_labels[i], noise_ratio=noise_ratio, noise_mode=noise_mode, seed=i)
    #         with h5py.File(noise_path, 'w') as f:
    #             f.create_dataset('noisy_labels', data=noisy_labels)
    #     train_labels[i] = noisy_labels
    return train_data, train_labels, test_data, test_labels, retrieval_data, retrieval_labels
