# experiments/06-event-edit-prototype/baselines.py
# v7-B-R1: B0-B4 基线 + Oracle upper bound
"""
严格 S1-S4 协议：
- S1: 训练冻结基座（Linear/GBDT）
- S2: 训练后处理
- S3: 选超参
- S4: 评估
"""

import numpy as np
from typing import Optional


class B0_FrozenBase:
    """冻结基座，不做校正"""
    def __init__(self):
        self.name = "B0"
    def fit(self, X_train, base_train, y_train):
        pass
    def predict(self, X, base):
        return base.copy()


class B1_PointwiseBECH:
    """逐点残差校正"""
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.name = "B1"
        self.residual_median = 0.0
    def fit(self, X_train, base_train, y_train):
        neg_mask = base_train < self.threshold
        if np.any(neg_mask):
            self.residual_median = np.median(y_train[neg_mask] - base_train[neg_mask])
    def predict(self, X, base):
        pred = base.copy()
        pred[base < self.threshold] += self.residual_median
        return pred


class B2_PointwiseResidual:
    """点级残差学习器"""
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.name = "B2"
        self.w = None
    def fit(self, X_train, base_train, y_train):
        from sklearn.linear_model import LinearRegression
        neg_mask = base_train < self.threshold
        if np.sum(neg_mask) > 1:
            X_neg = X_train[neg_mask]
            resid = (y_train - base_train)[neg_mask]
            self.w = LinearRegression().fit(X_neg, resid)
    def predict(self, X, base):
        pred = base.copy()
        if self.w is not None:
            neg_mask = base < self.threshold
            if np.any(neg_mask):
                pred[neg_mask] += self.w.predict(X[neg_mask])
        return pred


class B3_24VectorMLP:
    """24 点向量校正（学习版）"""
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.name = "B3"
        self.daily_correction = None
    def fit(self, X_train, base_train, y_train):
        n = len(base_train)
        n_days = n // 24
        if n_days < 2:
            self.daily_correction = np.zeros(24)
            return
        base_d = base_train[:n_days*24].reshape(n_days, 24)
        y_d = y_train[:n_days*24].reshape(n_days, 24)
        resid_d = y_d - base_d
        self.daily_correction = np.mean(resid_d, axis=0)
    def predict(self, X, base):
        pred = base.copy()
        for j in range(min(24, len(pred))):
            if base[j] < self.threshold:
                pred[j] += self.daily_correction[j % 24]
        return pred


class B4_ContiguousDecoder:
    """持续期检测 + 幅度校正"""
    def __init__(self, threshold=0.0, min_dur=2):
        self.threshold = threshold
        self.min_dur = min_dur
        self.name = "B4"
        self.correction = 0.0
    def fit(self, X_train, base_train, y_train):
        neg_mask = base_train < self.threshold
        if np.any(neg_mask):
            self.correction = np.mean(y_train[neg_mask] - base_train[neg_mask])
    def predict(self, X, base):
        pred = base.copy()
        n = len(pred)
        neg = base < self.threshold
        in_ep = False; start = 0
        for i in range(n):
            if neg[i] and not in_ep:
                in_ep = True; start = i
            elif not neg[i] and in_ep:
                if i - start >= self.min_dur:
                    pred[start:i] += self.correction
                in_ep = False
        if in_ep and n - start >= self.min_dur:
            pred[start:] += self.correction
        return pred


class OracleEdit:
    """Oracle：直接应用真值编辑（上限）"""
    def __init__(self):
        self.name = "Oracle"
    def fit(self, *args):
        pass
    def predict(self, X, base):
        return base.copy()  # 需要外部传入 true
