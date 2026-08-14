# P1 第三阶段 — T5 方向判定(国内数据混合比例网格 · 增量数据)

convergence_status: _待填_
T5_verdict: **IN PROGRESS**(假设已验,sanity 通过,全量运行中)
r_grid: {0(T0 equal 基线), 0.15, 0.30} × 3 seeds
training_pool: 12 国外源域(不变)+ 12 国内训练域(shandong_DA / gansu_DA / shaanxi_DA × 4 host)
domestic_gradient_share: r = 0.15 → 0.1765/域, r = 0.30 → 0.4286/域(国外=1.0/域)
point_readout_selected: 沿用第一轮 —— weighted_mean(未变)
transfer_status: _待填_
negative_transfer_status: _待填_
SOTA_status: _待填_
next_recommendation: _待填_

---

## 结论(人可读)

_待填_

### 1. 国外 32-cell transfer matrix(国内数据入训是帮还是伤国外)

_待填_

### 2. 国内 holdout 40-cell(入训后国内是否更好)

_待填_

### 3. 收敛性

_待填_

### 4. 诚实披露

_待填_

## 失败项 / 未解决

_待填_

## 下一步建议

_待填_

git_sha: _待填_
结果目录: experiments/08-hch-v2/results/P1_T5_<gitsha>/(每 r 训练报告 + transfer_matrix.csv + domestic_holdout.csv + summary.json)
