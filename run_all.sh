#!/usr/bin/env bash
# Полный прогон. Перед запуском поднимите GROBID (см. README) и задайте NCBI_EMAIL.
# Шаг ручной сверки эталона пропустить нельзя: скрипт остановится на нём.
set -euo pipefail
cd "$(dirname "$0")"

python 01_select_and_download.py --n "${N_ARTICLES:-30}"
python 02_pymupdf_diagnostics.py
python 03_run_grobid.py --label "${GROBID_LABEL:-grobid}"
python 04_parse_tei.py
python 05_make_gold_drafts.py --n "${N_GOLD:-12}"

echo
echo "=== СТОП: проверьте gold/*.json вручную по PDF и поставьте \"verified\": true ==="
echo "Затем:  python 06_evaluate.py && python 07_timing_report.py"
python 07_timing_report.py