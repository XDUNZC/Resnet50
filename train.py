import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from mydataset import OmniglotTrain, OmniglotTest
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import Siamese
import time
import numpy as np
import gflags
import sys
from collections import deque
import os
import torch.nn as nn
from mmdet.apis import init_detector, inference_detector
import numpy as np
from mmcv import imshow_bboxes

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


if __name__ == '__main__':

    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", True, "use cuda")
    gflags.DEFINE_string("train_path", "/home/niexing/projects/Tianchi/siamese-pytorch/trainset_demo/train", "training folder")
    gflags.DEFINE_string("test_path", "/home/niexing/projects/Tianchi/siamese-pytorch/trainset_demo/test", 'path of testing folder')
    gflags.DEFINE_integer("way", 200, "how much way one-shot learning")
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 32, "number of batch size")
    gflags.DEFINE_float("lr", 0.00006, "learning rate")
    gflags.DEFINE_integer("show_every", 10, "show result after each show_every iter.")
    gflags.DEFINE_integer("save_every", 10000, "save model after each save_every iter.")
    gflags.DEFINE_integer("test_every", 10000, "test model after each test_every iter.")
    gflags.DEFINE_integer("max_iter", 500000, "number of iterations before stopping")
    gflags.DEFINE_string("model_path", "/home/data/pin/model/siamese", "path to store model")
    gflags.DEFINE_string("gpu_ids", "0,1,2,3", "gpu ids used to train")

    Flags(sys.argv)

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    # train_dataset = dset.ImageFolder(root=Flags.train_path)
    # test_dataset = dset.ImageFolder(root=Flags.test_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
    print("use gpu:", Flags.gpu_ids, "to train.")
    
    # trainSet = OmniglotTrain(Flags.train_path, transform=data_transforms)
    trainSet = OmniglotTrain(Flags.train_path)
    trainLoader = DataLoader(trainSet, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)

    testSet = OmniglotTest(Flags.test_path, transform=transforms.ToTensor(), times = Flags.times, way = Flags.way)
    testLoader = DataLoader(testSet, batch_size=Flags.way, shuffle=False, num_workers=Flags.workers)
    
    # import pdb;pdb.set_trace()
    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)

    # net = Siamese(ResidualBlock)
    # resnet50 = torchvision.models.resnet50(pretrained=True)
    # # for param in resnet18.parameters():
    # #     param.requires_grad = False
    # num_ftrs = resnet50.fc.in_features
    # resnet50.fc = nn.Linear(num_ftrs, 1)#将全连接层做出改变类别改为一类

    # net = Siamese_ResNet([3, 4, 6, 3])
    # #读取参数
    # pretrained_dict = resnet50.state_dict()
    # model_dict = net.state_dict()
    # # 将pretrained_dict里不属于model_dict的键剔除掉
    # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 更新现有的model_dict
    # model_dict.update(pretrained_dict)
    # # 加载真正需要的state_dict
    # net.load_state_dict(model_dict)
    
    net = Siamese()
    
    # multi gpu
    if len(Flags.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)

    if Flags.cuda:
        net.cuda()

    net.train()
    
    optimizer = torch.optim.Adam(net.parameters(),lr = Flags.lr)
    optimizer.zero_grad()
    
    train_loss = []
    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)
    
    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
    # for batch_id, data_0 in enumerate(trainLoader, 1):
        # (img1, img2, label) = data_0  
        if batch_id > Flags.max_iter:
            break
        if Flags.cuda:
            img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
        else:
            img1, img2, label = Variable(img1), Variable(img2), Variable(label)
        optimizer.zero_grad()
        # with torch.no_grad():
        #     output = net.forward(img1, img2)
        
        output = net.forward(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()
        if batch_id % Flags.show_every == 0 :
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s'%(batch_id, loss_val/Flags.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if batch_id % Flags.save_every == 0:
            torch.save(net.state_dict(), Flags.model_path + '/model-inter-' + str(batch_id+1) + ".pt")
        if batch_id % Flags.test_every == 0:
            right, error = 0, 0
            with torch.no_grad():
                for _, (test1, test2) in enumerate(testLoader, 1):
                    if Flags.cuda:
                        test1, test2 = test1.cuda(), test2.cuda()
                    test1, test2 = Variable(test1), Variable(test2)
                    
                    output = net.forward(test1, test2).data.cpu().numpy()
                    pred = np.argmax(output)
                    if pred == 0:
                        right += 1
                    else: error += 1
                print('*'*70)
                print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
                print('*'*70)
                queue.append(right*1.0/(right+error))
        train_loss.append(loss_val)
    #  learning_rate = learning_rate * 0.95

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    acc = 0.0
    for d in queue:
        acc += d
    print("#"*70)
    print("final accuracy: ", acc/20)
