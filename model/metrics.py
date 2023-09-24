import torch.nn.functional as F
import numpy as np
import torch

from skimage import measure
from torchvision import transforms


class IoUMetric:

    def __init__(self):
        self.reset()

    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)  # TP, T
        inter, union = self.batch_intersection_union(pred, labels)
        #
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        output = output.detach().numpy()
        target = target.detach().numpy()

        predict = (output > 0).astype('int64')  # P
        pixel_labeled = np.sum(target > 0)  # T
        pixel_correct = np.sum((predict == target) * (target > 0))  # TP
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass
        predict = (output.detach().numpy() > 0).astype('int64')  # P
        target = target.numpy().astype('int64')  # T
        intersection = predict * (predict == target)  # TP

        # areas of intersection and union
        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all()
        return area_inter, area_union


class nIoUMetric:

    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()

    def update(self, preds, labels):
        inter_arr, union_arr = self.batch_intersection_union(preds, labels, self.nclass, self.score_thresh)
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return IoU, mIoU

    def reset(self):
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])

    def batch_intersection_union(self, output, target, nclass, score_thresh):
        # inputs are tensor
        # the category 0 is ignored class, typically for background / boundary
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass

        predict = (F.sigmoid(output).detach().numpy() > score_thresh).astype('int64')  # P
        target = target.detach().numpy().astype('int64')  # T
        intersection = predict * (predict == target)  # TP

        num_sample = intersection.shape[0]
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_lab_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)

        for b in range(num_sample):
            # areas of intersection and union
            area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            area_inter_arr[b] = area_inter

            area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
            area_pred_arr[b] = area_pred

            area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
            area_lab_arr[b] = area_lab

            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union

            assert (area_inter <= area_union).all()

        return area_inter_arr, area_union_arr


class ROCMetric:

    def __init__(self, bins, nclass=1):
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)

    def update(self, outputs, labels):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) * 42 / self.bins
            # score_thresh = iBin
            i_tp, i_pos, i_fp, i_neg, _, _ = cal_tp_pos_fp_neg(outputs, labels, score_thresh)

            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + np.spacing(1))
        fp_rates = self.fp_arr / (self.neg_arr + np.spacing(1))

        return tp_rates, fp_rates

    def reset(self):
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)


class ROCMetric2:

    def __init__(self, nclass, bins):
        super(ROCMetric2, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos, _ = cal_tp_pos_fp_neg(preds, labels, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + np.spacing(1))
        fp_rates = self.fp_arr / (self.neg_arr + np.spacing(1))

        recall = self.tp_arr / (self.pos_arr + np.spacing(1))
        precision = self.tp_arr / (self.class_pos + np.spacing(1))

        return tp_rates, fp_rates, recall, precision

    def reset(self):
        self.tp_arr = np.zeros([self.bins + 1])
        self.pos_arr = np.zeros([self.bins + 1])
        self.fp_arr = np.zeros([self.bins + 1])
        self.neg_arr = np.zeros([self.bins + 1])
        self.class_pos = np.zeros([self.bins + 1])


def cal_tp_pos_fp_neg(output, target, score_thresh):

    predict = (output.detach().numpy() > score_thresh).astype('int64')  # P # np[512,512,1]
    predict = transforms.ToTensor()(predict).numpy()  # tensor[1,512,512] -> np[1,512,512]
    target = target.detach().numpy()  # np[1,512,512]

    tp = np.sum((predict == target) * (target > 0))  # TP
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    fn = ((predict != target) * (1 - predict)).sum()  # FN
    pos = tp + fn
    neg = fp + tn
    return tp, pos, fp, neg, tn, fn


class PD_FA:
    def __init__(self, img_size):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.img_size = img_size
        self.FA = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels):

        preds, labels = preds.squeeze(1), labels.squeeze(1)
        predits = np.array((preds > 0).cpu()).astype('int64')
        labelss = np.array(labels.cpu()).astype('int64')  # P

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):
             centroid_label = np.array(list(coord_label[i].centroid))
             for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.FA += np.sum(self.dismatch)
        self.PD += len(self.distance_match)

    def get(self, img_num):

        Final_FA = self.FA / ((self.img_size * self.img_size) * img_num)
        Final_PD = self.PD / self.target

        return Final_FA, Final_PD

    def reset(self):
        self.image_area_total = []
        self.image_area_match = []
        self.FA = 0
        self.PD = 0
        self.target = 0
