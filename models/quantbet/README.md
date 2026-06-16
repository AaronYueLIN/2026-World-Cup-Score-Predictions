# QuantBet-EV v6.1 增强包

在 v6.0（Bayesian DC + HistGBM）之上补齐五块短板，取材自金融 / 经济学 / 凸优化文献。
纯 NumPy / SciPy / sklearn，无新增重依赖（不强制 CVXPY）。

```
quantbet/
├── devig.py            Shin (1992/1993) de-vig + 比例/幂法
├── dc_utils.py         Dixon-Coles 比分矩阵 (参考实现)
├── markets.py          盘口谓词 + 精确联合概率 (same-game parlay)
├── staking.py          Kelly / 分数 Kelly / 后验下分位 Kelly
├── portfolio.py        Busseti-Ryu-Boyd 风险约束 Kelly 组合 (回撤上界)
├── posterior.py        Laplace 后验 + 后验预测概率
├── pooling.py          线性 vs 对数池化 + RPS 最优权
├── evaluation.py       RPS/logloss/Brier + bootstrap CI + CLV + 校准
├── value_engine_v2.py  价值层编排 (de-vig → EV → 注码)
└── demo.py             端到端演示 (合成数据可独立运行)
tests/test_quantbet.py  21 项正确性测试
run_tests.py            无 pytest 时的轻量运行器
```

运行：
```bash
python -m quantbet.demo          # 端到端演示
python run_tests.py              # 21/21 通过 (或 python -m pytest tests/ -v)
```

## 五项改进与对应文献

| 模块 | 改进 | 替换/升级 | 文献 |
|------|------|-----------|------|
| devig | Shin 法修正 favorite-longshot 偏差 | 替换 v1 比例 de-vig | Shin 1992/1993；Štrumbelj 2014 |
| posterior | MAP → Laplace 后验预测 | 升级点估计 | Gelman BDA3 Ch.4 |
| portfolio | 风险约束 Kelly（回撤上界） | 替换固定 1/4 + 启发式枚举 | Busseti-Ryu-Boyd 2016 |
| markets | 同场串关用矩阵精确联合概率 | 替换独立相乘 | （已有 11×11 矩阵） |
| pooling | 对数池化 + RPS 最优权 | 替换线性软投票 | Genest-Zidek 1986；Ranjan-Gneiting 2010 |
| evaluation | bootstrap CI + CLV | 修正 n=5 准确率误导 | Gneiting-Raftery 2007；Levitt 2004 |

## 接入现有 pipeline

**1. 替换 `value_engine.py` 的 de-vig**
```python
from quantbet.devig import devig_shin
p_fair, z = devig_shin([o_home, o_draw, o_away], return_z=True)
```

**2. 给 Bayesian DC 加后验（几乎零改动）**
你的 `BayesianDixonColesModel` 已有负对数后验目标 `obj(theta)` 与 MAP 解 `theta_hat`：
```python
from quantbet.posterior import numerical_hessian, laplace_covariance, posterior_predictive

H = numerical_hessian(model._neg_log_posterior, model.theta_)   # 或用模型已算的 Hessian
cov, _ = laplace_covariance(H)

def predict_fn(theta):
    return model.predict_from_theta(theta, 'Brazil', 'Morocco', venue='neutral')

# attack 块去均值, 严格满足 Σatk=0
proj = lambda th: model.project_sum_to_zero(th)
p_pp, samples = posterior_predictive(model.theta_, cov, predict_fn, constraint_proj=proj)
```
`p_pp` 即后验预测概率（比 MAP 插值更不极端），`samples` 喂给 `lower_confidence_kelly`。
> n 大时数值 Hessian 偏慢；production 建议改用 NumPyro ADVI，接口不变。

**3. 同场串关用精确联合概率**
你的模型已输出 11×11 矩阵 `M`：
```python
from quantbet.markets import joint_prob, home_win, over
p_exact = joint_prob(M, home_win(), over(2.5))   # 而非 P(home)*P(over)
```

**4. 组合下注换成风险约束 Kelly**
```python
from quantbet.portfolio import Bet, Leg, MatchModel, risk_constrained_kelly
res = risk_constrained_kelly(bets, [MatchModel('BRA_MAR', M)], lam=2.0)
# res.stakes: 各注最优比例; res.drawdown_bound(0.3): P(回撤≤30%) 上界
```
设定 λ 即设定回撤容忍度：保证 `P(inf_t W_t ≤ α·W_0) ≤ α^λ`。

**5. 集成换对数池化，评估加 CI**
```python
from quantbet.pooling import optimize_weight, log_pool
from quantbet.evaluation import compare_models_ci
w = optimize_weight(P_dc, P_gbm, y_val, method='log')      # A/B 你现有的 0.70/0.30
P = log_pool([P_dc, P_gbm], [w, 1-w])
diff, lo, hi = compare_models_ci(rps_dc_per, rps_gbm_per)  # CI 含 0 = 差异不显著
```

## 落地顺序建议

1. **devig（Shin）+ evaluation（CI/CLV）** — 成本极低，立刻修正 EV 偏差与评估误导。
2. **posterior（Laplace）+ staking（下分位 Kelly）** — 解锁不确定性、防过度自信。
3. **portfolio（风险约束 Kelly）+ markets（精确串关）** — 把组合从启发式升级为有回撤保证的最优解。
4. **pooling（对数池化）** — 让 ensemble 真正产生增益（你当前 0.226 与单 DC 持平 = 软投票未生效）。

## 关于「不确定性收缩」的诚实说明

单注对数效用 Kelly 对概率是线性的，**后验均值是充分统计量**，单注层面不会因方差自动收缩。
真正使注码收缩的两个正确机制：(a) 用后验预测均值（已被先验收缩）替代 MAP 插值；
(b) 回撤/风险约束（`portfolio.risk_constrained_kelly`）。`lower_confidence_kelly` 是工程上常用、
Baker-McHale (2013) 精神的启发式保守化，已在代码中标注。
