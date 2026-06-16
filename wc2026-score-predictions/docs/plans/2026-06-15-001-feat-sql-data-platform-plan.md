---
date: 2026-06-15
type: feat
status: active
origin: docs/brainstorms/2026-06-15-sql-data-platform-requirements.md
---
# feat: QuantBet-EV SQL 数据平台

## Summary

为 QuantBet-EV 搭建本地 PostgreSQL 数据平台，包含数据库 schema、ETL pipeline、FastAPI REST API、Docker 部署和集成测试。模型预测代码从 CSV 文件迁移到数据库查询。数据用 Kaggle CSV 全量导入引导，日常更新靠 football-data.org API 手动触发。

## Problem Frame

当前项目数据管理靠 CSV 文件，有三个致命问题：(1) Kaggle snapshot 下载就是过期的，(2) 手动清洗造成静默数据丢失（v7.0 清洗版砍了 84.8%），(3) `data/` 目录堆了 475 个文件没有统一数据源。最直接的后果：瑞典 2026 年 6 月的比赛漏了，导致 4-1 胜突尼斯这场模型完全没预测到。

本计划把数据层从"CSV 文件散落"升级到"PostgreSQL 单一数据源"，同时不改变 Bayesian DC 预测引擎本身。

## Requirements

需求来源：`docs/brainstorms/2026-06-15-sql-data-platform-requirements.md`

### Database and Schema

- R1. 本地 PostgreSQL 实例存储所有比赛数据。Schema 最少涵盖 teams、tournaments、matches（比分、场地、日期）和 odds。
- R2. Alembic 管理所有 schema 迁移，每次变更可版本化、可回滚。

### ETL Pipeline

- R3. ETL 脚本从 football-data.org API 拉取新比赛结果（手动触发），执行 upsert。初始用 Kaggle CSV dump 全量导入引导数据库。
- R4. ETL 处理队名标准化（Soviet Union → Russia, Zaïre → DR Congo 等），使历史数据可按当前队名统一查询。
- R5. ETL 每次运行记录日志：时间戳、插入数、更新数、错误信息。失败不破坏已有数据。

### API

- R6. FastAPI 提供 REST 端点查询比赛、球队、赛事。
- R7. API 可被现有 Bayesian DC 预测代码调用，模型从数据库读取数据。

### Operations

- R8. 整套栈（PostgreSQL、API server、ETL）通过 `docker-compose up` 在本地单机运行。
- R9. 结构化 JSON 日志，含时间戳和 trace ID。

### Testing

- R10. 集成测试跑在真实 PostgreSQL 实例上，验证 ETL 写入正确、API 返回预期数据、migration 可正常应用。

## Key Technical Decisions

- **`db/` 和 `api/` 作为顶层新包，与 `models/` 平级。** 数据层代码和预测层代码不混在一起。`db/` 放 SQLAlchemy 模型和 Alembic 迁移，`api/` 放 FastAPI 路由和 server。
- **队名标准化用静态文件而非数据库表。** `former_names.csv`（37 条映射，来自 Kaggle 数据集）打包进 `db/`，ETL 启动时加载为内存字典。v1 不需要动态维护队名映射。
- **初始导入用 `archive (3)/results.csv`，增量用 football-data.org。** 初始全量导入从你已经验证过的 49,413 场 Kaggle dump 走；日常增量用 football-data.org 免费 API 补最新的比赛。
- **FastAPI 不加认证。** 本地单用户部署，API 绑定 localhost，无需 auth 层。后续如果开放对外访问再加。
- **结构化日志用 Python `logging` + `python-json-logger`。** 不需要 ELK 或其他日志收集系统——日志输出到 stdout，Docker 收集即可。

## High-Level Technical Design

```mermaid
flowchart TB
    subgraph Data Sources
        K[Kaggle CSV\n49,413 matches]
        FD[football-data.org API\nincremental]
    end

    subgraph ETL[ETL Pipeline - scripts/etl_pipeline.py]
        LOAD[CSV Loader]
        FETCH[API Fetcher]
        NORM[Team Name Normalizer\nformer_names.csv]
        UPSERT[Upsert Engine]
    end

    subgraph DB[PostgreSQL - port 5432]
        T_teams[teams]
        T_tourneys[tournaments]
        T_matches[matches]
        T_odds[odds]
    end

    subgraph API[FastAPI - port 8000]
        ROUTES[/matches /teams /tournaments]
        LOG[Structured Logger]
    end

    subgraph MODEL[Prediction Engine]
        DC[Bayesian DC]
        GBM[HistGBM Ensemble]
    end

    K --> LOAD
    FD --> FETCH
    LOAD --> NORM
    FETCH --> NORM
    NORM --> UPSERT
    UPSERT --> T_matches
    UPSERT --> T_teams
    UPSERT --> T_tourneys
    T_matches --> ROUTES
    T_teams --> ROUTES
    T_tourneys --> ROUTES
    ROUTES --> LOG
    ROUTES --> DC
    ROUTES --> GBM
```

## Implementation Units

### U1. 项目基础设施搭建

**Goal:** 创建 `db/` 和 `api/` 包、更新依赖、创建 Alembic 配置和 Docker 骨架。

**Requirements:** R2

**Files:**
- `requirements.txt` — 新增 sqlalchemy, alembic, psycopg2-binary, fastapi, uvicorn, python-json-logger, httpx
- `pyproject.toml` — 新增 db 和 api 包到 setuptools include
- `db/__init__.py` — 空包声明
- `api/__init__.py` — 空包声明
- `db/models.py` — SQLAlchemy Base 和模型文件占位
- `alembic.ini` — Alembic 配置
- `db/migrations/env.py` — Alembic 环境配置
- `Dockerfile` — API server 镜像
- `docker-compose.yml` — PostgreSQL + API 两个服务

**Approach:** 按照 Python 项目标准骨架搭建。Alembic 用 `alembic init` 初始化后修改 `env.py` 指向 `db/models.py` 的 Base metadata。Docker 用 `python:3.13-slim` 基础镜像，PostgreSQL 用 `postgres:16-alpine`。

**Test scenarios:**
- `docker-compose up` 启动后 PostgreSQL 和 API 两个容器均健康运行
- `alembic upgrade head` 可执行（即使尚无业务表）

**Verification:** `docker-compose up` 成功，API 返回 200 on `/health`，PostgreSQL 可被 SQLAlchemy 连接。

---

### U2. 数据库 Schema 和初始迁移

**Goal:** 定义 SQLAlchemy 模型并生成初始 Alembic 迁移。

**Requirements:** R1, R2

**Dependencies:** U1

**Files:**
- `db/models.py` — Team, Tournament, Match, OddsOffer 四个模型
- `db/migrations/versions/001_initial_schema.py` — Alembic 自动生成

**Approach:**

```python
# 模型关系示意 (directional, not implementation)
Team: id, name, normalized_name, fifa_code
Tournament: id, name, tier (friendly/qualifier/continental/world_cup)
Match: id, home_team_id(FK), away_team_id(FK), tournament_id(FK),
       home_score, away_score, date, city, country, neutral, venue
OddsOffer: id, match_id(FK), provider, home_odds, draw_odds, away_odds, recorded_at
```

`Match` 表上建复合唯一索引 `(home_team_id, away_team_id, date)` 以支持 upsert。`Tournament.tier` 用枚举：`friendly`, `qualifier`, `continental`, `world_cup`, `other`。

**Test scenarios:**
- Alembic migration 在空数据库上成功执行 `upgrade` → `downgrade` → `upgrade` 完整循环
- 插入一场比赛后外键约束有效：引用不存在的 team_id 会报错

**Verification:** `alembic upgrade head` 创建四张表，`\dt` 可见。

---

### U3. ETL Pipeline

**Goal:** 实现从 Kaggle CSV 全量导入 + football-data.org API 增量拉取的 ETL 脚本，含队名标准化和 upsert 逻辑。

**Requirements:** R3, R4, R5

**Dependencies:** U2

**Files:**
- `scripts/etl_pipeline.py` — 主入口
- `db/etl.py` — load_csv, fetch_api, normalize_teams, upsert_matches 核心函数
- `db/former_names.csv` — 37 条队名映射（从 archive (3) 复制）
- `tests/test_etl.py` — ETL 集成测试

**Approach:**

ETL 分四个阶段，每阶段独立函数：

1. **CSV Loader** — `pandas.read_csv` 读取初始 CSV，转为 dict 列表
2. **API Fetcher** — `httpx` 调 football-data.org `/v4/matches`，按日期范围增量拉取
3. **Team Normalizer** — 加载 `former_names.csv` 为 `{former: current}` 字典，按日期段映射
4. **Upsert Engine** — 用 SQLAlchemy `insert().on_conflict_do_update()` 对 `(home_team_id, away_team_id, date)` 唯一约束做 upsert

队名标准化逻辑：`former_names.csv` 含 `start_date` / `end_date`，按比赛的日期判断应该映射到哪个当前队名。不在此 CSV 中的队名保持原样。

**Execution note:** test-first — 先写 sample CSV 和 mock API response 的集成测试。

**Patterns to follow:** 和现有 `scripts/` 下的脚本风格一致：单文件可独立运行，argparse 支持 `--source csv|api` 参数。

**Test scenarios:**
- CSV 导入 10 场 sample 数据，验证 matches 表行数 = 10，teams 表正确去重
- API fetcher 用 mock server 返回 3 场比赛，验证只写入了新比赛
- 队名 "Soviet Union" 在 1988 年的比赛被映射为 "Russia"
- 重复导入同一场 CSV 不产生重复行（upsert 行为正确）
- API 拉取失败时抛异常但不破坏已有数据（事务回滚）

**Verification:** `python scripts/etl_pipeline.py --source csv --file path/to/results.csv` 成功写入 49,413 行；`python scripts/etl_pipeline.py --source api` 增量拉取新比赛。

---

### U4. FastAPI REST API

**Goal:** 实现 REST API，暴露 matches、teams、tournaments 查询端点，支持结构化日志。

**Requirements:** R6, R7, R9

**Dependencies:** U2

**Files:**
- `api/server.py` — FastAPI app 创建、CORS、启动
- `api/routes.py` — 端点定义
- `api/dependencies.py` — SQLAlchemy session 依赖注入
- `api/schemas.py` — Pydantic 响应模型
- `tests/test_api.py` — API 集成测试

**Approach:**

核心端点：

| 方法 | 路径 | 参数 | 返回 |
|---|---|---|---|
| GET | `/matches` | team, from, to, tournament, limit | Match 列表 |
| GET | `/teams/{name}` | — | Team 详情 |
| GET | `/teams/{name}/history` | from, to | 该队历史比赛 |
| GET | `/tournaments` | — | Tournament 列表 |
| GET | `/health` | — | `{"status": "ok"}` |

所有端点返回 JSON。FastAPI 自动生成 OpenAPI docs at `/docs`。

结构化日志用 `python-json-logger`，每个请求记录 `{timestamp, trace_id, method, path, status, duration_ms}`。`trace_id` 用 `uuid4` 在 middleware 中注入。

Session 管理：FastAPI dependency `get_db()` 用 yield 模式，请求结束自动关闭 session。

**Test scenarios:**
- `GET /matches?team=Sweden&from=2026-01-01` 返回正确过滤的比赛列表
- `GET /teams/Sweden/history` 返回所有瑞典历史比赛
- 无效参数（`from=not-a-date`）返回 422
- `GET /health` 返回 200
- 请求日志输出为有效 JSON

**Verification:** `curl localhost:8000/matches?team=Sweden&from=2026-01-01` 返回 JSON 数组。

---

### U5. Docker 部署

**Goal:** 用 `docker-compose.yml` 编排 PostgreSQL + API server，数据卷持久化，一键启动。

**Requirements:** R8

**Dependencies:** U2, U4

**Files:**
- `docker-compose.yml` — 服务编排
- `Dockerfile` — API 镜像构建
- `.env.example` — 环境变量模板

**Approach:**

```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: quantbet
      POSTGRES_USER: quantbet
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  api:
    build: .
    depends_on:
      - db
    environment:
      DATABASE_URL: postgresql://quantbet:${DB_PASSWORD}@db:5432/quantbet
    ports:
      - "8000:8000"
    volumes:
      - ./db/migrations:/app/db/migrations

volumes:
  pgdata:
```

`Dockerfile` 用 `python:3.13-slim`，安装依赖后 `CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]`。

`.env.example` 只含 `DB_PASSWORD=changeme`。用户复制为 `.env` 后修改。

**Test scenarios:**
- `docker-compose up -d` 后 `curl localhost:8000/health` 返回 200
- 重启容器后数据不丢失（volume 持久化验证）
- 缺少 `.env` 时 docker-compose 报清晰错误

**Verification:** `docker-compose up` 后 API 可访问，`docker-compose down && docker-compose up` 后数据仍存在。

---

### U6. 集成测试

**Goal:** 对 ETL、API、migration 写集成测试，跑在真实 PostgreSQL 上。

**Requirements:** R10

**Dependencies:** U3, U4

**Files:**
- `tests/conftest.py` — PostgreSQL 测试 fixture
- `tests/test_etl.py` — ETL 集成测试
- `tests/test_api.py` — API 集成测试（使用 FastAPI TestClient）

**Approach:**

`conftest.py` 提供 session-scoped fixture：

```python
# directional — 用 SQLAlchemy create_engine 连测试 DB
# 每个 session 创建一次表，测试结束 drop
# 用 testcontainers-python 或 docker 启动临时 PostgreSQL
```

简化版：测试前要求手动 `docker-compose up -d db`，测试连 `localhost:5432/quantbet_test`。不引入 testcontainers 依赖（太重）。

FastAPI 测试用 `TestClient`，依赖注入 override 指向测试数据库 session。

**Test scenarios:**
- (ETL) 导入 sample CSV → API 查询返回对应比赛
- (API) `GET /matches` 带多个过滤条件同时应用
- (API) `GET /teams/X/history` 返回结果按日期降序
- (Migration) 在已有数据的数据库上运行 `alembic upgrade head`，数据不丢失

**Verification:** `python -m pytest tests/test_etl.py tests/test_api.py -v` 全部通过。

---

### U7. 接入预测代码

**Goal:** 让 Bayesian DC 和 GBM ensemble 的训练 / 预测代码能从数据库读取比赛数据，替代原来的 CSV 读取。

**Requirements:** R7

**Dependencies:** U4

**Files:**
- `db/reader.py` — `get_matches(team, from_date, to_date)` 等查询函数
- `models/bayesian_dixon_coles.py` — `fit()` 新增 `from_db` 参数
- `models/ml_predictor.py` — `FeatureEngineer` 新增数据库读取路径

**Approach:**

在 `db/reader.py` 中封装常用查询，返回 pandas DataFrame（和现有代码接口兼容）：

```python
# directional
def get_team_history(team: str, before_date: str | None = None) -> pd.DataFrame
def get_all_matches(from_date: str | None = None) -> pd.DataFrame
def get_match_count() -> int
```

现有模型代码改动最小化——`fit(df)` 保持不变，新增 `fit_from_db()` 方法调 `db/reader.py` 获取数据后转调 `fit(df)`。

**Patterns to follow:** 现有的 `models/` 导入风格：`from db.reader import get_all_matches`。

**Test scenarios:**
- `get_team_history('Sweden')` 返回的 DataFrame 列名与现有 `FeatureEngineer` 期望一致
- `get_all_matches()` 返回行数等于数据库中 matches 表行数

**Verification:** 在数据库中导入 49,413 场比赛后，`python -c "from db.reader import get_all_matches; print(len(get_all_matches()))"` 输出 49413。

---

## Scope Boundaries

### Deferred for later

- 云部署（AWS / GCP / Azure）和 Kubernetes
- 前端 dashboard 或 UI
- 球员级 xG 数据接入
- ETL 定时调度（cron / Airflow）

### Deferred to Follow-Up Work

- Bayesian DC 用完整 49k 数据重训——本计划只搭数据平台，重训是后续步骤
- odds 数据的实时更新和 Shin de-vig 接入——表已建好，ETL 逻辑后续补

## Risks & Dependencies

| 风险 | 缓解 |
|---|---|
| football-data.org 免费 API 有限流（每分钟 10 次） | 手动触发 ETL 不频繁调用；初始全量走 CSV |
| Docker Desktop 在 Windows 上的性能 | PostgreSQL 用 alpine 镜像；数据卷只存 DB 文件 |
| 49k 场比赛 upsert 可能慢 | 批量 insert（每 1000 行 commit 一次） |
| Team 名标准化不可能 100% 覆盖 | `former_names.csv` 只覆盖已知改名；新的未知队名直接保留原名，不阻塞 ETL |

## Verification

全栈验收：

```bash
# 1. 启动
docker-compose up -d

# 2. 迁移
docker-compose exec api alembic upgrade head

# 3. 全量导入
docker-compose exec api python scripts/etl_pipeline.py --source csv --file data/results_raw.csv

# 4. API 验证
curl localhost:8000/matches?team=Sweden&from=2026-01-01

# 5. 模型读 DB
docker-compose exec api python -c "
from db.reader import get_all_matches
df = get_all_matches()
print(f'Loaded {len(df)} matches')
"

# 6. 测试
docker-compose exec api python -m pytest tests/ -v
```
