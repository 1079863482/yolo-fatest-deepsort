# vim: expandtab:ts=4:sw=4

"""
Tracker类是最核心的类，Tracker中保存了所有的轨迹信息，
负责初始化第一帧的轨迹、卡尔曼滤波的预测和更新、负责级联匹配、IOU匹配等等核心工作。
"""

from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric       # metric是一个类，用于计算距离（余弦距离和马氏距离）
        self.max_iou_distance = max_iou_distance         # 最大iou，iou匹配的时候使用
        self.max_age = max_age    # 错过的最大次数
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()  # 卡尔曼滤波器
        self.tracks = []         # 用于保存一系列的轨迹
        self._next_id = 1        # 下一个分配轨迹的id

    def predict(self):
        """
        遍历每个track都进行一次预测
        主要是对轨迹列表中所有的轨迹使用卡尔曼滤波算法进行状态的预测
        """
        for track in self.tracks:         # 依次对tracks里的子track遍历并且预测
            track.predict(self.kf)

    def update(self, detections):
        """
        进行测量的更新和轨迹管理
        """
        # Run matching cascade.              # 级联匹配，匹配成功、未匹配的轨迹、未匹配的检测目标
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:                 # 1. 匹配上的结果，track更新对应的detection
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:                       # 2. 未匹配上的tracker，调用mark_missed标记
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:               # 3. 未匹配上的detection，detection失配，进行初始化
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]       # 得到新的tracker列表，保存的是标记未confirmed和Tentative的track

        # Update distance metric.     更新度量距离
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]            # 获取所有的confirmed状态的track id
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features              # 将tracks列表拼接到features列表
            targets += [track.track_id for _ in track.features]             # 获取每个feature对应的track id
            track.features = []
        self.metric.partial_fit(                                           # 距离度量中的 ，特征集更新
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        """
        主要功能是进行匹配，找到匹配的、未匹配的部分
        """
        def gated_metric(tracks, dets, track_indices, detection_indices):
            """
            功能：用于计算track和detection之间的距离，代价函数
            需要使用在KM算法前
            """
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)                 # 1. 通过最近邻计算代价矩阵
            cost_matrix = linear_assignment.gate_cost_matrix(                     # 2. 计算马氏距离，得到新的状态矩阵
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # 划分不同的轨迹状态
        confirmed_tracks = [                # 确认的轨迹
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [              # 未确认的轨迹
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        """
        进行级联匹配，得到匹配的track、不匹配的track、不匹配的detection
        仅仅对确定态的轨迹进行级联匹配
        """
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        # 将所有未确定态的轨迹和刚刚没有匹配上的轨迹组合为iou_track_candidates，进行iou的匹配
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]         # 刚刚没有匹配上
        unmatched_tracks_a = [                             # 未匹配
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]         # 已经很久没有匹配上

        """
        IOU匹配，对级联匹配中还没有匹配成功的目标再进行IOU匹配
        虽然和级联匹配中使用的都是min_cost_matching作为核心，这里使用的metric是iou cost和以上不同
        """

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b                 # 组合两部分的match得到结果
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):               # 初始化分配track
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
