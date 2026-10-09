#!/usr/bin/env bash
# Block until Postgres accepts connections, or fail loudly after ~60s.
set -euo pipefail

CONTAINER="${PG_CONTAINER:-voicedesk-postgres}"
RETRIES=30

for i in $(seq 1 "$RETRIES"); do
  if docker compose exec -T postgres pg_isready -U voicedesk -d voicedesk >/dev/null 2>&1; then
    echo "postgres: ready (after ${i} attempt(s))"
    exit 0
  fi
  printf '.'
  sleep 2
done

echo
echo "postgres: NOT ready after $((RETRIES * 2))s" >&2
echo "Check logs with: docker compose logs postgres" >&2
exit 1
