# scoreline —— 比分预测增强包

仅依赖 `numpy` / `scipy`，可直接放入 `models/quantbet/scoreline/`，与现有
`BayesianDixonColesModel` 协同或替换其比分矩阵层。

## 模块

| 文件 | 作用 |
|------|------|
| `count_dists.py` | 边际计数分布：Poisson、负二项（过离散）、Weibull-count（McShane 2008）|
| `bivariate.py` | 全表相依：双变量 Poisson（Karlis–Ntzoufras）、Frank copula、对角膨胀 |
| `dynamic_strength.py` | 动态强度状态空间滤波（Koopman–Lit 思想的在线 EKF 近似）|
| `calibration.py` | 验证集上温度 + 对角膨胀联合校准 |
| `score_model.py` | `FlexibleScoreModel`，与 `BayesianDixonColesModel.predict` 同构输出 |

## 最小用法

```python
from models.quantbet.scoreline import FlexibleScoreModel

m = FlexibleScoreModel(margin="nb", dependence="frank",
                       diagonal_inflation=True, max_goals=10)
m.fit(df)                      # df: date, home_team, away_team, home_goals, away_goals
pred = m.predict("Arsenal", "Chelsea", "home")
# pred 的键与 BayesianDixonColesModel.predict 完全一致 → 可直接喂给下游
```

## 与现有贝叶斯模型对接（推荐）

```python
# 1) 用现有 MAP 后验的 attack/defense 注入，跳过重训
m.set_strengths(attack=att_map, defense=def_map,
                home_adv=ha, intercept=mu)

# 2) 或启用动态强度（捕捉伤病/转会/状态拐点）
from models.quantbet.scoreline import DynamicStrengthFilter
psd = DynamicStrengthFilter.tune_process_sd(train_df, val_df)   # 含 0.0 候选
filt = DynamicStrengthFilter(process_sd=psd).fit(train_df)
lh, la = filt.expected_goals("Arsenal", "Chelsea", as_of=match_date)
M = m.score_matrix(lh, la)
```

校准器在验证集上拟合后 `m.calibrator = cal`，`score_matrix` 会自动应用。

## 调参提示

- `margin`：先在验证集比较 `poisson` / `nb` / `weibull`，过离散明显选 `nb`。
- `dependence`：`frank` 支持正负相依；联赛实测常为弱负相依。
- `process_sd`：务必在真实验证集重调；候选含 `0.0`，无漂移时自动退回静态。
