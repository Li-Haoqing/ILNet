import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

from PIL import Image
from model.metrics import IoUMetric, nIoUMetric, ROCMetric
from model.metrics import cal_tp_pos_fp_neg
from model.ilnet import ILNet_S, ILNet_M, ILNet_L

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = ArgumentParser(description='Implement of MY net')

    parser.add_argument('--img_path', type=int, default='', help='image path: .png')
    parser.add_argument('--mask_path', type=int, default=1, help='mask path: .png')
    parser.add_argument('--mode', type=str, default='L', help='mode: L, M, S')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint: .pth')

    args = parser.parse_args()
    return args


class Demo:

    def __init__(self, img_path: str, mask_path: str, img_size=512):
        self.img_path = img_path
        self.mask_path = mask_path

        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.35619214, 0.35575104, 0.35673013], [0.2614548, 0.26135704, 0.26168558]),
        ])
        self.score_thresh = 0

    def test(self, model_path: str):
        img = Image.open(self.img_path).convert('RGB')
        mask = Image.open(self.mask_path).convert('RGB')
        img, mask = self._testval_sync_transform(img, mask)
        imgs = np.hstack([img, mask])
        np_mask = np.array(mask.convert('L'))
        np_mask = np_mask[:, :, np.newaxis]  # np[512,512,1]

        img = self.transform(img).unsqueeze(0)

        assert args.mode in ['L', 'M', 'S']
        if args.mode == 'L':
            self.net = ILNet_L()
        elif args.mode == 'M':
            self.net = ILNet_M()
        elif args.mode == 'S':
            self.net = ILNet_S()
        else:
            NameError

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(img.cuda()).squeeze(0).permute(1, 2, 0).cpu()
            predict = (output.detach().numpy() > self.score_thresh).astype('float32')

        num_pre, num_label, IoU, TPR, FPR, _fpr, Pd, Fa = self.evaluate(output, np_mask)

        print('T=={}'.format(num_label))
        print('TP=={}'.format(num_pre))
        print('TPR=={}'.format(max(TPR)))
        print('TPR=={}'.format(num_pre / num_label))  # score_thresh=self.score_thresh
        print('FPR=={}'.format(_fpr))  # score_thresh=self.score_thresh
        print('IoU=={}'.format(IoU))
        print('Pd=={}'.format(Pd))
        print('Fa=={}'.format(Fa))

        cv2.imshow('img', imgs)
        cv2.imshow('predict', predict)
        cv2.waitKey(0)

        return predict

    def _testval_sync_transform(self, img, mask):
        img_size = self.img_size
        img = img.resize((img_size, img_size), Image.BILINEAR)
        mask = mask.resize((img_size, img_size), Image.NEAREST)

        return img, mask

    def evaluate(self, output, label):
        pre_mask = (output.detach().numpy() > self.score_thresh).astype('float32')
        pre_mask, label = transforms.ToTensor()(pre_mask), transforms.ToTensor()(label)  # [1,512,512]

        iou_metric = IoUMetric()
        # nIoU_metric = nIoUMetric(1, score_thresh=0.5)
        iou_metric.reset()
        # nIoU_metric.reset()
        roc = ROCMetric(40)
        roc.update(output, label)  # output:[512,512,1]  label:[1,512,512]

        TPR, FPR = roc.get()
        _tp, _pos, _fp, _neg, _tn, _fn = cal_tp_pos_fp_neg(output, label, score_thresh=self.score_thresh)
        num_pre, num_label = iou_metric.batch_pix_accuracy(pre_mask, label)
        area_inter, area_union = iou_metric.batch_intersection_union(pre_mask, label)
        IoU = 1.0 * area_inter / (np.spacing(1) + area_union)

        Pd = (_tp + _tn) / (_pos + _neg)  # pd
        Fa = (_fn + _fp) / (_pos + _neg)  # fa
        return num_pre, num_label, np.around(IoU, 4).item(), TPR, FPR, _fp / _neg, Pd, Fa


if __name__ == "__main__":
    args = parse_args()

    demodata = Demo(img_path=args.img_path, mask_path=args.mask_path)
    pre = demodata.test(model_path=args.checkpoint)
