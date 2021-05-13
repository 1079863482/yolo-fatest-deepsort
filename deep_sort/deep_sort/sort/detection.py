# vim: expandtab:ts=4:sw=4
"""
Detection类用于保存通过目标检测器得到的一个检测框，包含top left坐标+框的宽和高，以及该bbox的置信度还有通过reid获取得到的对应的embedding。除此以外提供了不同bbox位置格式的转换方法：

tlwh: 代表左上角坐标+宽高
tlbr: 代表左上角坐标+右下角坐标
xyah: 代表中心坐标+宽高比+高

"""

"""
This class represents a bounding box detection in a single image.

Parameters
----------
tlwh : array_like
    Bounding box in format `(x, y, w, h)`.
confidence : float
    Detector confidence score.
feature : array_like
    A feature vector that describes the object contained in this image.

Attributes
----------
tlwh : ndarray
    Bounding box in format `(top left x, top left y, width, height)`.
confidence : ndarray
    Detector confidence score.
feature : ndarray | NoneType
    A feature vector that describes the object contained in this image.

"""

import numpy as np

class Detection(object):

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
