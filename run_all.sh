#!/usr/bin/env bash
# Reproduce every result from the raw data. Total runtime on a 2-core machine: ~3 hours
# (the experiment queue is ~60 of them; it skips configurations already saved in data/results/).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs data/results

echo "== tests";            python3 tests/test_simulator.py && python3 tests/test_pipeline.py
echo "== data";             python3 src/data_prep.py
echo "== step 1";           python3 src/step1_demo.py
echo "== step 2";           python3 src/step2_demo.py
echo "== step 3";           python3 src/step3_demo.py
echo "== step 4";           python3 src/step4_demo.py
echo "== step 5";           python3 src/step5_demo.py
echo "== experiments";      python3 src/experiment.py
echo "== dead-product rule"; python3 src/dead_rule.py
echo "== shock tests";      python3 src/shock_test.py
echo "== report";           python3 src/results_report.py
