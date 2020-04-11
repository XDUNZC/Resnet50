from mmdet.apis import init_detector, inference_detector
import numpy as np
from mmcv import imshow_bboxes
import os
from PIL import Image
import torch
from torchvision import transforms

def get_model(config_file='configs/my.py',
              checkpoint_file='work_dirs/retinanet_x101_64x4d_fpn_1x/latest.pth',
              device='cuda:0'):
    model = init_detector(config_file, checkpoint_file, device=device)
    return model


def get_result_box(result, score_thr=0.7):
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
    return bboxes_over_thr,labels_over_thr


def get_result_and_feats(model,img,score_thr=0.7):

    result, roi_feats = inference_detector(model, img)
    # print(result)
    bboxes_over_thr,labels_over_thr = get_result_box(result, score_thr=score_thr)
    return bboxes_over_thr, labels_over_thr, roi_feats

def save_image(tensor, filepath):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    # image = image.squeeze(0)  # remove the fake batch dimension
    image = transforms.ToPILImage()(image)
    image.save(filepath)


# todo 封装时注意一个问题：不要把模型model读进来初始化，检测完丢掉，再读进来，再检测完再丢掉。注意节省时间
if __name__ == '__main__':
    # # demo 检测图片 注意替换文件路径
    # config_file = 'configs/my.py'  # 替换为指定的configs/xx.py
    # checkpoint_file = 'checkpoints/retinanet_x101_64x4d_fpn_1x20200322-53c08bb4.pth'  # 替换为预训练权重
    # device = 'cuda:0'  # GPU 卡号
    # model = get_model(config_file=config_file, checkpoint_file=checkpoint_file, device=device)
    # img_path = 'demo/demo.jpg'
    # bboxes = get_result_box(img_path=img_path, model=model, score_thr=0.7) # array，输出格式是(N,5),N个满足条件的框 每个框与5个值，前4个是位置信息，最后一个是概率值 0-1
    # print(bboxes)
    #
    # # demo 生成feature_extractor
    # idx = 0  # 0 for faster rcnn, 1 for retinanet
    # config_file = ['configs/tbtc_fater_rcnn_voc.py',
    #                'tbtc_retinanet_voc.py', 'tbtc_feature_exteactor_faster_rcnn.py',
    #                'tbtc_feature_exteactor_faster_rcnn.py'
    #                ][idx]
    # checkpoint_file = ['checkpoints/faster_rcnn_x101_64x4d_fpn_1x20200324-ba5926a5.pth',
    #                    'retinanet_x101_64x4d_fpn_1x20200322-53c08bb4.pth'][idx]



    idx = 0  # 0 for faster rcnn, 1 for retinanet
    config_file = ['configs/tbtc_fater_rcnn_voc.py',
                   'tbtc_retinanet_voc.py', 'tbtc_feature_exteactor_faster_rcnn.py',
                   'tbtc_feature_exteactor_faster_rcnn.py'
                   ][idx]
    checkpoint_file = ['checkpoints/faster_rcnn_x101_64x4d_fpn_1x20200324-ba5926a5.pth',
                       'retinanet_x101_64x4d_fpn_1x20200322-53c08bb4.pth'][idx]
    device = 'cuda:0'  # GPU 卡号
    # import pdb;pdb.set_trace()
    model = get_model(config_file=config_file, checkpoint_file=checkpoint_file, device=device)
    img = 'tbtc_test.jpg'
    result_over_thr, labels_over_thr, roi_feats = get_result_and_feats(model, img)
    roi_feats = roi_feats.reshape(-1, 7, 7)
    print(result_over_thr)
    print(labels_over_thr)
    print(roi_feats.shape)

    dataPath = '../trainset_demo/sub_match_data/train'
    datas = {}
    idx = 0
    alphaPath = os.listdir(dataPath)[0]
    betaPath = os.listdir(dataPath)[1]
    for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
        datas[idx] = []
        imagefilePath = os.path.join(dataPath, alphaPath, charPath)
        image_train = Image.open(imagefilePath).resize((105, 105))
        # datas[idx].append((Image.open(imagefilePath)).resize((105, 105)))
        image_numpy = np.array(image_train)
        import pdb;pdb.set_trace()	
        vidoefilePath = os.path.join(dataPath, betaPath, charPath)
        datas[idx].append((Image.open(vidoefilePath)).resize((105, 105)))
        idx += 1

        print(datas[0].size)
        break

    print("finish loading training dataset to memory")

    '''
    # Create all the roi_feats and save them.
    '''
    # idx = 0  # 0 for faster rcnn, 1 for retinanet
    # config_file = ['configs/tbtc_fater_rcnn_voc.py',
    #                'tbtc_retinanet_voc.py', 'tbtc_feature_exteactor_faster_rcnn.py',
    #                'tbtc_feature_exteactor_faster_rcnn.py'
    #                ][idx]
    # checkpoint_file = ['checkpoints/faster_rcnn_x101_64x4d_fpn_1x20200324-ba5926a5.pth',
    #                    'retinanet_x101_64x4d_fpn_1x20200322-53c08bb4.pth'][idx]
    # device = 'cuda:0'  # GPU 卡号
    # model = get_model(config_file=config_file, checkpoint_file=checkpoint_file, device=device)

    # TrainPath = '../TBtianchi_code/data/train'
    # Train_image_path = os.listdir(TrainPath)[0]
    # Train_videoimage_path = os.listdir(TrainPath)[1]
    # Train_image_file = '../TBtianchi_code/data/train_rois/train_rois_images'
    # Train_videoimage_file = '../TBtianchi_code/data/train_rois/train_rois_video_image'
    # for charPath in os.listdir(os.path.join(TrainPath, Train_image_path)):
    #     imagefilePath = os.path.join(TrainPath, Train_image_path, charPath)
    #     # img = 'demo/tbtc_test.jpg'
    #     result_over_thr, labels_over_thr, roi_feats = get_result_and_feats(model, imagefilePath)
    #     roi_feats = roi_feats.reshape(-1, 7, 7)
    #     save_image(roi_feats, Train_image_file + charPath)
    #     # roi_feats.ospath.save(os.path.join(Train_image_file))
    #     import pdb;pdb.set_trace()
    # TestPath = '../TBtianchi_code/data/test'
    # TestAlphaPath = os.listdir(dataPath)[0]
    # TestBetaPath = os.listdir(dataPath)[1]
    # print(result_over_thr)
    # print(labels_over_thr)
    # print(roi_feats.shape)
