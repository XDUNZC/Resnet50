import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import time
import random
import torchvision.datasets as dset
from PIL import Image
# from sklearn.cross_validation import train_test_split
import torchvision
from torchvision import transforms
from mmdet.apis import init_detector, inference_detector
import numpy as np
from mmcv import imshow_bboxes
import os
from PIL import Image
from torchvision import transforms

class OmniglotTrain(Dataset):

    def __init__(self, dataPath, transform=None):
        super(OmniglotTrain, self).__init__()
        np.random.seed(0)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath)
        
    '''
    old version loadToMem(self, dataPath)
    '''
    # def loadToMem(self, dataPath):
    #     print("begin loading training dataset to memory")
    #     datas = {}
    #     idx = 1
    #     alphaPath = os.listdir(dataPath)[0]
    #     betaPath = os.listdir(dataPath)[1]
    #     for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
    #         datas[idx] = []
    #         imagefilePath = os.path.join(dataPath, alphaPath, charPath)
    #         datas[idx].append((Image.open(imagefilePath)).resize((105, 105)))
    #         vidoefilePath = os.path.join(dataPath, betaPath, charPath)
    #         datas[idx].append((Image.open(vidoefilePath)).resize((105, 105)))
    #         idx += 1
    #     print("finish loading training dataset to memory")
    #     return datas, idx

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        idx_load = 0
        alphaPath = os.listdir(dataPath)[0]
        betaPath = os.listdir(dataPath)[1]
        for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
            datas[idx_load] = []
            imagefilePath = os.path.join(dataPath, alphaPath, charPath)
            '''
            Create all the roi_feats.
            '''
            idx = 0  # 0 for faster rcnn, 1 for retinanet
            config_file = ['configs/tbtc_fater_rcnn_voc.py',
                           'tbtc_retinanet_voc.py', 'tbtc_feature_exteactor_faster_rcnn.py',
                           'tbtc_feature_exteactor_faster_rcnn.py'
                           ][idx]
            checkpoint_file = ['checkpoints/faster_rcnn_x101_64x4d_fpn_1x20200324-ba5926a5.pth',
                               'retinanet_x101_64x4d_fpn_1x20200322-53c08bb4.pth'][idx]
            device = 'cuda:0'  # GPU 卡号
            
            model = self.get_model(config_file = config_file, checkpoint_file = checkpoint_file, device = device)
            result_over_thr, labels_over_thr, image_roi_feats = self.get_result_and_feats(model, imagefilePath)
            image_roi_feats = image_roi_feats.reshape(-1, 7, 7).cpu().numpy()
            datas[idx_load].append(image_roi_feats)
            # import pdb;pdb.set_trace()
            vidoefilePath = os.path.join(dataPath, betaPath, charPath)
            '''
            Create all the roi_feats.
            '''
            # img = 'demo/tbtc_test.jpg'
            result_over_thr, labels_over_thr, videoimage_roi_feats = self.get_result_and_feats(model, vidoefilePath)
            videoimage_roi_feats = videoimage_roi_feats.reshape(-1, 7, 7).cpu().numpy()
            datas[idx_load].append(videoimage_roi_feats)
            idx_load += 1
        print("finish loading training dataset to memory")
        return datas, idx_load

    def get_model(self, config_file='configs/my.py',
              checkpoint_file='work_dirs/retinanet_x101_64x4d_fpn_1x/latest.pth',
              device='cuda:0'):
        model = init_detector(config_file, checkpoint_file, device=device)
        return model

    def get_result_box(self, result, score_thr=0.7):
        '''

        :param score_thr: 后处理，只输出概率值大于thr的框
        :return: bboxes_over_thr 是array，输出格式是(N,5),N个满足条件的框
                 每个框与5个值，前4个是位置信息，最后一个是概率值 0-1
        '''
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        bboxes_over_thr = bboxes[inds]
        labels_over_thr = labels[inds]
        return bboxes_over_thr, labels_over_thr

    def get_result_and_feats(self, model,img, score_thr=0.7):

        result, roi_feats = inference_detector(model, img)
        # print(result)
        bboxes_over_thr,labels_over_thr = self.get_result_box(result, score_thr=score_thr)
        return bboxes_over_thr, labels_over_thr, roi_feats

    def __len__(self):
        return  21000000
        # return self.num_classes * 2 / 16

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])

        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return torch.from_numpy(image1), torch.from_numpy(image2), torch.from_numpy(np.array([label], dtype=np.float32))

class OmniglotTest(Dataset):

    def __init__(self, dataPath, transform=None, times=200, way=200):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)
        
    # def loadToMem(self, dataPath):
    #     print("begin loading test dataset to memory")
    #     datas = {}
    #     idx = 0
    #     for alphaPath in os.listdir(dataPath):
    #         for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
    #             datas[idx] = []
    #             for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
    #                 filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
    #                 datas[idx].append(Image.open(filePath).convert('L'))
    #             idx += 1
    #     print("finish loading test dataset to memory")
    #     return datas, idx

    def loadToMem(self, dataPath):
        print("begin loading testing dataset to memory")
        datas = {}
        idx_load = 0
        alphaPath = os.listdir(dataPath)[0]
        betaPath = os.listdir(dataPath)[1]
        for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
            datas[idx_load] = []
            imagefilePath = os.path.join(dataPath, alphaPath, charPath)

            '''
            Create all the roi_feats.
            '''
            idx = 0  # 0 for faster rcnn, 1 for retinanet
            config_file = ['configs/tbtc_fater_rcnn_voc.py',
                           'tbtc_retinanet_voc.py', 'tbtc_feature_exteactor_faster_rcnn.py',
                           'tbtc_feature_exteactor_faster_rcnn.py'
                           ][idx]
            checkpoint_file = ['checkpoints/faster_rcnn_x101_64x4d_fpn_1x20200324-ba5926a5.pth',
                               'retinanet_x101_64x4d_fpn_1x20200322-53c08bb4.pth'][idx]
            device = 'cuda:0'  # GPU 卡号
            
            model = self.get_model(config_file = config_file, checkpoint_file = checkpoint_file, device = device)
            # img = 'demo/tbtc_test.jpg'
            result_over_thr, labels_over_thr, image_roi_feats = self.get_result_and_feats(model, imagefilePath)
            image_roi_feats = image_roi_feats.reshape(-1, 7, 7).cpu().numpy()
            datas[idx_load].append(image_roi_feats)

            vidoefilePath = os.path.join(dataPath, betaPath, charPath)
            '''
            Create all the roi_feats.
            '''
            # img = 'demo/tbtc_test.jpg'
            result_over_thr, labels_over_thr, videoimage_roi_feats = self.get_result_and_feats(model, vidoefilePath)
            videoimage_roi_feats = videoimage_roi_feats.reshape(-1, 7, 7).cpu().numpy()
            datas[idx_load].append(videoimage_roi_feats)
            idx_load += 1
        print("finish loading testing dataset to memory")
        return datas, idx_load


    def get_model(self, config_file='configs/my.py',
              checkpoint_file='work_dirs/retinanet_x101_64x4d_fpn_1x/latest.pth',
              device='cuda:0'):
        model = init_detector(config_file, checkpoint_file, device=device)
        return model

    def get_result_box(self, result, score_thr=0.7):
        '''
        :param score_thr: 后处理，只输出概率值大于thr的框
        :return: bboxes_over_thr 是array，输出格式是(N,5),N个满足条件的框
                 每个框与5个值，前4个是位置信息，最后一个是概率值 0-1
        '''
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        bboxes_over_thr = bboxes[inds]
        labels_over_thr = labels[inds]
        return bboxes_over_thr, labels_over_thr

    def get_result_and_feats(self, model,img, score_thr=0.7):

        result, roi_feats = inference_detector(model, img)
        # print(result)
        bboxes_over_thr,labels_over_thr = self.get_result_box(result, score_thr=score_thr)
        return bboxes_over_thr, labels_over_thr, roi_feats

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])

        if self.transform:
            img1 = self.transform(self.img1).permute(1, 2, 0)
            img2 = self.transform(img2).permute(1, 2, 0)
        return img1, img2

# test
if __name__=='__main__':
    omniglotTrain = OmniglotTrain('.images_background', 30000*8)
    print(omniglotTrain)

