import numpy as np

COLOR_MAP = [
    [0, 0, 0],  # 0
    [255, 255, 255],  # 1
    [0, 0, 255],  # 2
]


def get_v2_pallete(num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while lab > 0:
                pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
                pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
                pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
                i = i + 1
                lab >>= 3
        return pallete  # list, lenth is n_classes*3

    v2_pallete = _getpallete(num_cls)  # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    return v2_pallete


COLOR_MAP_SS = get_v2_pallete()


def mean_iou(input, target, classes=2):
    """compute the value of mean iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        miou: float, the value of miou
    """
    miou = 0
    for i in range(classes):
        intersection = np.logical_and(target == i, input == i)
        # print(intersection.any())
        union = np.logical_or(target == i, input == i)
        temp = np.sum(intersection) / np.sum(union)
        miou += temp
    return miou / classes
