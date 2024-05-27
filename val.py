import torch
import torchvision.transforms as transforms
import cv2
import torch.utils.data as Data

from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image
from model.metrics import IoUMetric, nIoUMetric, PD_FA, ROCMetric2
from utils.data import SirstDataset, IRSTD1K_Dataset
from model.ilnet import ILNet_S, ILNet_M, ILNet_L

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = ArgumentParser(description='Implement of MY net')

    parser.add_argument('--img_size', type=int, default=512, help='image size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size for testing')
    parser.add_argument('--dataset', type=str, default='sirst', help='datasets: sirst or IRSTD-1k')
    parser.add_argument('--mode', type=str, default='L', help='mode: L, M, S')
    parser.add_argument('--checkpoint', type=str, default=r'.pth', help='checkpoint: .pth')

    args = parser.parse_args()
    return args


class Val:

    def __init__(self, args, load_path: str):
        self.args = args

        # datasets
        if args.dataset == 'sirst':
            self.val_set = SirstDataset(args, mode='val')
        elif args.dataset == 'IRSTD-1k':
            self.val_set = IRSTD1K_Dataset(args, mode='val')
        else:
            NameError
        self.val_data_loader = Data.DataLoader(self.val_set, batch_size=args.batch_size)

        assert args.mode in ['L', 'M', 'S']
        if args.mode == 'L':
            self.net = ILNet_L()
        elif args.mode == 'M':
            self.net = ILNet_M()
        elif args.mode == 'S':
            self.net = ILNet_S()
        else:
            NameError
        checkpoint = torch.load(load_path)
        # self.net.load_state_dict(checkpoint)
        self.net.load_state_dict(checkpoint['net'])

        self.net.to(device)

        self.iou_metric = IoUMetric()
        self.nIoU_metric = nIoUMetric(1, score_thresh=0.5)
        self.PD_FA = PD_FA(args.img_size)
        self.ROC = ROCMetric2(1, bins=10)

    def test_model(self):
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()

        self.net.eval()
        print(next(self.net.parameters()).device)
        tbar = tqdm(self.val_data_loader)
        for i, (data, labels) in enumerate(tbar):
            with torch.no_grad():
                output = self.net(data.cuda())
                output = output.cpu()

            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.PD_FA.update(output, labels)
            output2 = output.squeeze(0).permute(1, 2, 0)
            labels2 = labels.squeeze(0)
            self.ROC.update(output2, labels2)

            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            Fa, Pd = self.PD_FA.get(len(self.val_set))
            ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()

            # 进度条描述信息
            tbar.set_description('IoU:%f, nIoU:%f, Fa:%.10f, Pd:%.10f'
                                 % (IoU, nIoU, Fa, Pd))
        return IoU, nIoU, Fa, Pd, ture_positive_rate, false_positive_rate


if __name__ == "__main__":
    args = parse_args()

    value = Val(args, load_path=args.checkpoint)
    IoU, nIoU, Fa, Pd, ture_positive_rate, false_positive_rate = value.test_model()
    print('IoU:{},\n nIoU:{},\n Fa:{},\n Pd:{},\n TPR:{},\n FPR:{}'.format(IoU, nIoU, Fa, Pd, ture_positive_rate, false_positive_rate))
