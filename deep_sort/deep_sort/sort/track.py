# vim: expandtab:ts=4:sw=4

"""
Track类主要存储的是轨迹信息，mean和covariance是保存的框的位置和速度信息，track_id代表分配给这个轨迹的ID。state代表框的状态，有三种：

Tentative: 不确定态，这种状态会在初始化一个Track的时候分配，并且只有在连续匹配上n_init帧才会转变为确定态。如果在处于不确定态的情况下没有匹配上任何detection，那将转变为删除态。
Confirmed: 确定态，代表该Track确实处于匹配状态。如果当前Track属于确定态，但是失配连续达到max age次数的时候，就会被转变为删除态。
Deleted: 删除态，说明该Track已经失效。
"""


class TrackState:

    Tentative = 1         # 不确定态
    Confirmed = 2         # 确定态
    Deleted = 3           # 删除态

class Track:

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean            # 保存框位置
        self.covariance = covariance    # 速度信息
        self.track_id = track_id         # 轨迹id
        self.hits = 1                    # 确认次数大于n_init，不确定转确定
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative        # 刚开始都是不确定态
        self.features = []             # 存储该轨迹不同帧对应位置通过reid提取到的特征
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init             # 连续确认帧数
        self._max_age = max_age           # track存活期限，超过没匹配上会删除

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
