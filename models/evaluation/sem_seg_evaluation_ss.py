import itertools
import json
import logging
import numpy as np
import os
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image
from itertools import zip_longest
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False


def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
        array = array // 255  # * AVSS4 dataset, change object 255 to 1.
    return array


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0]
        else:
            v_list = [self.__data[k][0] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v


def _batch_miou_fscore(output, target, nclass, T, beta2=0.3):
    """batch mIoU and Fscore"""
    # output: [BF, C, H, W],
    # target: [BF, H, W]
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1
    predict = predict.float() * (target > 0).float()  # [BF, H, W]
    intersection = predict * (predict == target).float()  # [BF, H, W]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    batch_size = target.shape[0] // T
    cls_count = torch.zeros(nclass).float()
    ious = torch.zeros(nclass).float()
    fscores = torch.zeros(nclass).float()

    vid_miou_list = []
    for i in range(target.shape[0]):
        area_inter = torch.histc(intersection[i].cpu(), bins=nbins, min=mini, max=maxi)  # TP
        area_pred = torch.histc(predict[i].cpu(), bins=nbins, min=mini, max=maxi)  # TP + FP
        area_lab = torch.histc(target[i].cpu(), bins=nbins, min=mini, max=maxi)  # TP + FN
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        iou = 1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
        # iou[torch.isnan(iou)] = 1.
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1

        precision = area_inter / area_pred
        recall = area_inter / area_lab
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[torch.isnan(fscore)] = 0.0
        fscores += fscore

        vid_miou_list.append(torch.sum(iou) / (torch.sum(iou != 0).float()))

    return ious, fscores, cls_count, vid_miou_list


def calc_color_miou_fscore(pred, target, T=10):
    r"""
    J measure
        param:
            pred: size [BF x C x H x W], C is category number including background
            target: size [BF x H x W]
    """
    nclass = pred.shape[1]
    pred = torch.softmax(pred, dim=1)  # [BF, C, H, W]
    # miou, fscore, cls_count = _batch_miou_fscore(pred, target, nclass, T)
    miou, fscore, cls_count, vid_miou_list = _batch_miou_fscore(pred, target, nclass, T)
    return miou, fscore, cls_count, vid_miou_list


class SemSegEvaluator_SS(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn("SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata.")
        if ignore_label is not None:
            self._logger.warn("SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata.")
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {}
        self.input_file_to_gt_file = {
            tuple(dataset_record["file_names"]): dataset_record["sem_seg_file_names"] for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )

        self.miou_pc = AverageMeter("miou_pc")
        self.f_score_pc = AverageMeter("f_score_pc")
        self.cls_pc = AverageMeter("cls_pc")

    def reset(self):
        self.miou_pc = AverageMeter("miou_pc")
        self.f_score_pc = AverageMeter("f_score_pc")
        self.cls_pc = AverageMeter("cls_pc")
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        num_video = -1
        for num_img, output in enumerate(outputs):
            output = output["sem_seg"]
            if num_img % 10 == 0:  # v1s and v1m is less than 10. But len(outputs)==1.
                num_video += 1
                if num_img == 0:
                    gts = inputs[num_video]["sem_segs"].squeeze(dim=1).cuda()
                else:
                    gts = torch.cat((gts, inputs[num_video]["sem_segs"].squeeze(dim=1).cuda()), dim=0)
            if num_img == 0:
                preds = output.unsqueeze(dim=0)
            else:
                preds = torch.cat((preds, output.unsqueeze(dim=0)), dim=0)
        _miou_pc, _fscore_pc, _cls_pc, _ = calc_color_miou_fscore(preds, gts)

        self.miou_pc.add({"miou_pc": _miou_pc})
        self.f_score_pc.add({"f_score_pc": _fscore_pc})
        self.cls_pc.add({"cls_pc": _cls_pc})

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            # conf_matrix_list = all_gather(self._conf_matrix)
            # b_conf_matrix_list = all_gather(self._b_conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            miou_pc_list = all_gather(self.miou_pc.pop("miou_pc"))
            miou_pc = sum(miou_pc_list) / len(miou_pc_list)
            f_score_pc_list = all_gather(self.f_score_pc.pop("f_score_pc"))
            f_score_pc = sum(f_score_pc_list) / len(f_score_pc_list)
            cls_pc_list = all_gather(self.cls_pc.pop("cls_pc"))
            cls_pc = sum(cls_pc_list) / len(cls_pc_list)
            if not is_main_process():
                return

        miou_pc = miou_pc / cls_pc
        self._logger.info(f"[test miou] {torch.sum(torch.isnan(miou_pc)).item()} classes are not predicted in this batch")
        miou_pc[torch.isnan(miou_pc)] = 0
        miou = torch.mean(miou_pc).item()
        miou_noBg = torch.mean(miou_pc[:-1]).item()
        f_score_pc = f_score_pc / cls_pc
        self._logger.info(f"[test fscore] {torch.sum(torch.isnan(f_score_pc)).item()} classes are not predicted in this batch")
        f_score_pc[torch.isnan(f_score_pc)] = 0
        f_score = torch.mean(f_score_pc).item()
        f_score_noBg = torch.mean(f_score_pc[:-1]).item()

        self._logger.info(
            "test | cls {}, miou: {:.4f}, miou_noBg: {:.4f}, F_score: {:.4f}, F_score_noBg: {:.4f}".format(
                torch.sum(cls_pc != 0).item(), miou, miou_noBg, f_score, f_score_noBg
            )
        )
        res = {}
        res["mIoU"] = round(miou, 4)  
        res["f_score"] = round(f_score, 4)
        os.makedirs(self._output_dir, exist_ok=True)
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)

        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results

        Args:
            sem_seg: predicted semantic segmentation of shape (H, W).
            input_file_name: the file name of the image.
        Returns:
            list[dict]: encoded prediction strings in COCO format.
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert label in self._contiguous_id_to_dataset_id, "Label {} is not in the metadata info for {}".format(
                    label, self._dataset_name
                )
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append({"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle})
        return json_list

    def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
        assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        kernel = np.ones((3, 3), dtype=np.uint8)

        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        eroded_mask_with_padding = cv2.erode(padded_mask, kernel, iterations=dilation)
        eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
        boundary = mask - eroded_mask
        return boundary
