---
date: 2026-06-15
topic: sql-data-platform
---
# QuantBet-EV SQL 数据平台

## Summary

Replace the current CSV-file-based data management with a local PostgreSQL database, an automated ETL pipeline, a FastAPI REST API, and production-grade tooling (Alembic migrations, Docker deployment, pytest integration tests, structured logging). The platform runs on a single local machine and gives the Bayesian DC prediction engine reliable, queryable, never-stale match data.

## Problem Frame

QuantBet-EV's prediction accuracy is compromised by CSV files that are stale, incomplete, or incorrectly filtered. The most recent incident — Sweden's June 2026 matches missing from the dataset — caused the model to underestimate Sweden's form ahead of their 4-1 win over Tunisia. Diagnosing this required manually comparing three versions of a Kaggle dump across two archive directories.

The current workflow is: download a CSV snapshot from Kaggle → manually clean and filter it → read it with pandas on every prediction run. This has three failure modes: (1) the snapshot is out of date the moment it's downloaded, (2) manual cleaning introduces silent data loss (the v7.0 clean script dropped 84.8% of matches), and (3) there is no single source of truth — the `data/` directory contained 475 files across CSV, HTML, TXT, XLSX, and JSON formats before being purged.

## Requirements

### Database and Schema

- R1. A local PostgreSQL instance stores all match data. The schema supports at minimum: teams, tournaments, matches (with home/away scores, venue, date), and odds.
- R2. Alembic manages all schema migrations. Every schema change is versioned and reversible.

### ETL Pipeline

- R3. An ETL script pulls match results from football-data.org API on manual trigger and performs upserts — new matches are inserted, existing matches updated if scores changed. An initial full import from the Kaggle CSV dump bootstraps the database.
- R4. The ETL pipeline handles team name normalization. Historical team names (e.g., Soviet Union → Russia, Zaïre → DR Congo) map to their current successors so that a query for "Russia" returns matches from the Soviet era.
- R5. The ETL pipeline logs every run: timestamp, records inserted, records updated, errors encountered. A failed run does not corrupt existing data.

### API

- R6. A FastAPI server exposes REST endpoints for querying matches, teams, and tournaments. Core endpoints include `GET /matches?team=X&from=Y&to=Z` and `GET /teams/{name}/history`.
- R7. The API can be called by the existing Bayesian DC prediction code — the model reads from the database instead of from CSV files.

### Operations

- R8. The entire stack (PostgreSQL, API server, ETL scheduler) runs via `docker-compose up` on a single local machine.
- R9. Structured application logs (JSON format) record every API request, ETL run, and database error, with timestamps and trace IDs for debugging.

### Testing

- R10. Integration tests run against a real PostgreSQL instance (not mocks). Tests verify the ETL pipeline ingests sample data correctly, the API returns expected responses, and schema migrations apply cleanly.

## Scope Boundaries

### Deferred for later

- Cloud deployment (AWS/GCP/Azure) and Kubernetes orchestration.
- A web dashboard or frontend UI.
- Player-level xG data ingestion.

### Outside this product's identity

- Replacing the Bayesian Dixon-Coles model itself. The SQL platform is a data layer; the prediction engine remains unchanged.
- Real-time odds streaming or in-play betting support.

## Key Decisions

- **Python native stack (FastAPI + SQLAlchemy) over PostgREST.** The API server and the Bayesian DC model share a Python process, eliminating cross-language overhead and keeping the deployment surface small.
- **Single PostgreSQL node, no read replicas.** At ~50,000 matches and single-user query volume, a single node is sufficient. Read replicas add operational complexity with no benefit at this scale.
- **Local deployment over cloud.** The user runs everything on their own machine. Docker Compose provides reproducible setup without external infrastructure dependencies.

## Dependencies / Assumptions

- **Data source:** Kaggle CSV dump for initial full import (49,413 matches as of 2026-06-14); football-data.org free API for daily incremental updates.
- **ETL trigger:** Manual — the user runs the ETL script on demand. No automated scheduler in v1.
- PostgreSQL is available via Docker; the user does not need to install it natively.
- The existing Bayesian DC pkl model will be retrained against the full database after the platform is operational — this is a separate step, not part of the data platform build.
