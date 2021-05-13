import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker

__all__ = ['DeepSort']

"""
DeepSORT的update流程：

根据传入的参数（bbox_xywh, conf, img）使用ReID模型提取对应bbox的表观特征。
构建detections的列表，列表中的内容就是Detection类,在此处限制了bbox的最小置信度。
使用非极大抑制算法，由于默认nms_thres=1，实际上并没有用。
Tracker类进行一次预测，然后将detections传入，进行更新。
最后将Tracker中保存的轨迹中状态属于确认态的轨迹返回。
"""

class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence                      # 检测的置信度阈值
        self.nms_max_overlap = nms_max_overlap                     # 非极大值抑制，设置为1代表不进行抑制

        self.extractor = Extractor(model_path, use_cuda=use_cuda)         # reid模型初始化，用于提取图片的embedding，返回的是一个batch图片对应的特征

        max_cosine_distance = max_dist                             # 用在级联匹配的地方，如果大于改阈值，直接忽略
        nn_budget = 100                              # 缓存，每个类别最多的样本个数，如果超过直接忽略
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)            # 可选的距离参数，cosine或者euclidean
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)   # 提取每个bbox图片的特征得到embedding
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)         # [cx,cy,w,h] - > [x1,y1,w,h]
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]      # 置信度过滤,小于0.3的全部过滤
        # Detection是一个存储图中一个bbox结果，需要：1.bbox（tlwh形式）2.对应的置信度 3. 对应embedding

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)        # 使用非极大值抑制，去除检测中重叠的bonding box，实际上没有使用
        detections = [detections[i] for i in indices]   # NMS后，更新检测bbox（其实没变）

        # update tracker  更新追踪
        # tracker给出一个预测结果，然后将detection传入，进行卡尔曼滤波操作
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities  存储结果可视化
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


