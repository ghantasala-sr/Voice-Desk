#!/usr/bin/env bash
# Apply migrations to an already-running database, in filename order.
#
# docker-entrypoint-initdb.d only runs on the FIRST start of an empty volume,
# so this is how you apply a new migration without `make clean`.
#
# Migrations are written to be idempotent (CREATE TABLE IF NOT EXISTS etc),
# so re-running is safe.
set -euo pipefail

cd "$(dirname "$0")/.."

for f in migrations/*.sql; do
  echo "applying ${f}"
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U voicedesk -d voicedesk < "$f"
done

echo "migrations: done"
