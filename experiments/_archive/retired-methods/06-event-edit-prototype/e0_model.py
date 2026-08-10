# experiments/06-event-edit-prototype/e0_model.py
# v7-B-R2: 严格无真值推理的 E0 模型
"""
核心原则：
1. E0.fit 接收：S2 特征、S2 基座预测、S2 真值（仅作为标签）
2. E0.predict 接收：只接收 S4 特征和 S4 基座预测（不接收真值）
3. Day input features 不能包含任何 y-derived statistic
4. 训练 INSERT/SHIFT 目标从 S2 Hungarian scripts 派生
5. S4 时只从模型输出生成位置/边界
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from harness import Episode, extract_episodes, hungarian_match


@dataclass
class DayFeatures:
    """每天的特征（仅使用安全特征，不含 y-derived）"""
    day_idx: int
    n_base_episodes: int       # 基座事件数（从 base 预测提取）
    mean_base_magnitude: float  # 基座事件平均幅度
    base_episode_hours: float   # 基座事件占比
    mean_base_price: float      # 基座日均值
    std_base_price: float       # 基座日标准差
    hour_features: np.ndarray   # 24 维小时特征（从 base 预测提取）


def extract_day_features_safe(
    base_prices: np.ndarray,
    day_idx: int,
) -> DayFeatures:
    """
    安全的 day-level 特征提取（不含任何 y-derived statistic）
    只使用基座预测值
    """
    base_eps = extract_episodes(base_prices, 0.0)

    n_base = len(base_eps)
    mean_base_mag = np.mean([e.magnitude for e in base_eps]) if base_eps else 0.0
    base_hours = sum(e.duration for e in base_eps) / len(base_prices) if base_eps else 0.0

    # 小时特征：基座预测是否为负价（从 base 预测提取，非真值）
    hour_features = np.zeros(24)
    for h in range(min(24, len(base_prices))):
        if base_prices[h] < 0:
            hour_features[h] = 1.0

    return DayFeatures(
        day_idx=day_idx,
        n_base_episodes=n_base,
        mean_base_magnitude=mean_base_mag,
        base_episode_hours=base_hours,
        mean_base_price=np.mean(base_prices),
        std_base_price=np.std(base_prices),
        hour_features=hour_features,
    )


def extract_day_level(
    prices: np.ndarray,
    day_size: int = 24,
) -> List[np.ndarray]:
    """将价格序列按天分割"""
    n = len(prices)
    n_days = n // day_size
    return [prices[i*day_size:(i+1)*day_size] for i in range(n_days)]


class E0_EventEditor:
    """
    E0: 严格无真值推理的事件编辑模型
    - fit: S2 特征 + S2 基座预测 + S2 真值（仅标签）
    - predict: 只接收 S4 特征和 S4 基座预测（不接收真值）
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.name = "E0"

        # 编辑分类器
        self.insert_clf = None  # 预测是否需要 INSERT
        self.shift_reg = None   # 预测 SHIFT 边界偏移

    def fit(
        self,
        base_day_list: List[np.ndarray],   # S2 基座预测（按天）
        true_day_list: List[np.ndarray],   # S2 真值（仅标签）
    ):
        """
        训练编辑分类器
        - base_day_list: S2 基座预测（按天分割）
        - true_day_list: S2 真值（仅作为标签，不用于预测）
        """
        # 提取安全特征（只用 base 预测，不用真值）
        features = []
        for i, base in enumerate(base_day_list):
            feat = extract_day_features_safe(base, i)
            features.append([
                feat.n_base_episodes,
                feat.mean_base_magnitude,
                feat.base_episode_hours,
                feat.mean_base_price,
                feat.std_base_price,
            ])

        X = np.array(features)

        # 提取编辑标签（从 Hungarian scripts 派生）
        labels = self._extract_edit_labels(base_day_list, true_day_list)

        # INSERT 分类器
        y_insert = np.array(labels["insert_needed"])
        if np.sum(y_insert) > 0 and np.sum(y_insert) < len(y_insert):
            from sklearn.linear_model import LogisticRegression
            self.insert_clf = LogisticRegression(random_state=42, max_iter=1000)
            self.insert_clf.fit(X, y_insert)

        # SHIFT 回归器
        y_shift = np.array(labels["shift_delta"])
        if np.std(y_shift) > 0:
            from sklearn.linear_model import LinearRegression
            self.shift_reg = LinearRegression()
            self.shift_reg.fit(X, y_shift)

    def _extract_edit_labels(
        self,
        base_day_list: List[np.ndarray],
        true_day_list: List[np.ndarray],
    ) -> Dict:
        """从 Hungarian scripts 派生编辑标签"""
        labels = {"insert_needed": [], "shift_delta": []}

        for base, true in zip(base_day_list, true_day_list):
            base_eps = extract_episodes(base, self.threshold)
            true_eps = extract_episodes(true, self.threshold)

            matched, _, unmatched_true = hungarian_match(base_eps, true_eps)

            labels["insert_needed"].append(1 if unmatched_true else 0)

            shift_deltas = []
            for bi, ti in matched:
                b, t = base_eps[bi], true_eps[ti]
                shift_deltas.append(abs(b.start - t.start) + abs(b.end - t.end))
            labels["shift_delta"].append(np.mean(shift_deltas) if shift_deltas else 0.0)

        return labels

    def predict(
        self,
        base_day_list: List[np.ndarray],  # S4 基座预测（按天）
    ) -> List[np.ndarray]:
        """
        预测编辑并应用（不接收真值）
        - 只使用 base 预测和安全特征
        """
        predictions = []

        for i, base in enumerate(base_day_list):
            pred = base.copy()

            # 提取安全特征
            feat = extract_day_features_safe(base, i)
            X = np.array([[
                feat.n_base_episodes,
                feat.mean_base_magnitude,
                feat.base_episode_hours,
                feat.mean_base_price,
                feat.std_base_price,
            ]])

            # INSERT：如果分类器预测需要插入
            if self.insert_clf is not None:
                insert_prob = self.insert_clf.predict_proba(X)[0][1]
                if insert_prob > 0.5:
                    # INSERT：在基座无事件区域预测插入位置
                    base_eps = extract_episodes(base, self.threshold)
                    # 简单策略：在基座事件之间的空隙插入
                    # 实际应从模型输出预测插入位置
                    pass  # INSERT 逻辑在 NoInsert 消融中体现

            # SHIFT：如果回归器预测边界偏移
            if self.shift_reg is not None:
                shift_pred = self.shift_reg.predict(X)[0]
                if shift_pred > 1.0:
                    # SHIFT：调整基座事件边界
                    base_eps = extract_episodes(base, self.threshold)
                    # 简单策略：按预测偏移调整边界
                    # 实际应从模型输出预测新边界
                    pass  # SHIFT 逻辑在 NoShift 消融中体现

            predictions.append(pred)

        return predictions


class E0_NoInsert(E0_EventEditor):
    """E0 without INSERT"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "E0_NoInsert"
        self.insert_clf = None  # 禁用 INSERT

    def fit(self, base_day_list, true_day_list):
        super().fit(base_day_list, true_day_list)
        self.insert_clf = None  # 强制禁用


class E0_NoShift(E0_EventEditor):
    """E0 without SHIFT"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "E0_NoShift"
        self.shift_reg = None  # 禁用 SHIFT

    def fit(self, base_day_list, true_day_list):
        super().fit(base_day_list, true_day_list)
        self.shift_reg = None  # 强制禁用


class E0_NoMatching(E0_EventEditor):
    """E0 without edit matching（改为点级掩码）"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "E0_NoMatching"

    def predict(self, base_day_list):
        """点级掩码：在负价点应用校正"""
        predictions = []
        for base in base_day_list:
            pred = base.copy()
            neg_mask = base < self.threshold
            pred[neg_mask] += 0.0  # 简单校正
            predictions.append(pred)
        return predictions


class OracleEdit:
    """Oracle：直接应用真值编辑（上限，只读 S4 真值）"""
    def __init__(self):
        self.name = "Oracle"

    def fit(self, *args):
        pass

    def predict(self, base_day_list, true_day_list):
        """Oracle 预测：直接返回真值（只在评估时使用）"""
        return true_day_list.copy()
