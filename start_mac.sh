#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
command -v node >/dev/null || { echo 'Instala Node.js 22 LTS o superior.'; exit 1; }
command -v npm >/dev/null || { echo 'npm no disponible.'; exit 1; }
node -e 'if(Number(process.versions.node.split(".")[0])<22) process.exit(1)' || { echo 'Se necesita Node.js 22+.'; exit 1; }
PY="${FORECAST_PYTHON:-}"
if [ -z "$PY" ]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12)' 2>/dev/null; then PY="$candidate"; break; fi
  done
fi
[ -n "$PY" ] || { echo 'Instala Python 3.10, 3.11 o 3.12 (recomendado 3.12).'; exit 1; }
"$PY" -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12)' || { echo 'Utiliza Python 3.10–3.12.'; exit 1; }
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12)' || { echo 'La .venv existente requiere Python 3.10–3.12. Recrea el entorno compatible.'; exit 1; }
.venv/bin/python -m pip install --upgrade pip
REQ=python/requirements.txt
[ "${FORECAST_ADVANCED:-1}" = 1 ] && REQ=python/requirements-advanced.txt
.venv/bin/python -m pip install -r "$REQ"
.venv/bin/python python/check_environment.py "$REQ"
if [ -f package-lock.json ]; then npm ci; else npm install; fi
export FORECAST_PYTHON="$PWD/.venv/bin/python"
npm run dev
