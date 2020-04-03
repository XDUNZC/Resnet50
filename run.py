import torch
from torch.autograd import Variable
import gflags
import numpy as np

from .model import Siamese_ResNet, Bottleneck


class Worker:
    def __init__(self,
                 model_path="./model/Resnet50/models/"):  # TODO 改变model路径

        """
        初始化模型
        :param model_path:
        :param gpu_list:gpu列表，逗号字符串包含GPU号，逗号分割
        """
        # self.Flags = gflags.FLAGS
        # gflags.DEFINE_bool("cuda", True, "use cuda")
        # gflags.DEFINE_string("gpu_ids", "0", "gpu ids used to train")
        class item:
            def __init__(self):
                self.cuda=True
                self.gpu_ids="0"
        self.Flags=item()

        # 创建模型
        self.net = Siamese_ResNet([3, 4, 6, 3])
        # 加载模型参数
        model = torch.load(model_path)
        # model_dict = model.state_dict()
        # self.net.load_state_dict(model_dict)
        self.net.load_state_dict(model)

        # 加载整个模型
        # self.net = torch.load(model_path)

        # multi gpu
        if len(self.Flags.gpu_ids.split(",")) > 1:
            self.net = torch.nn.DataParallel(self.net)

        if self.Flags.cuda:
            self.net.cuda()
        self.net.eval()

    def get_match_value(self, img_a, img_b):
        """
        获得匹配值
        :param img_a:第一个图像 numpy(H*W*C)
        :param img_b:第二个图像 numpy(H*W*C
        :return:
        """

        with torch.no_grad():
            if self.Flags.cuda:
                img_a, img_b = img_a.cuda(), img_b.cuda()
            test1, test2 = Variable(img_a), Variable(img_b)
            output = self.net.forward(test1, test2).data.cpu().numpy()
        return output
