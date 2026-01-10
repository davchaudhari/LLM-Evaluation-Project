#!/bin/bash
# Run all experiments sequentially

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Running All Experiments"
echo "=========================================="
echo ""

# HF Baseline
echo "=========================================="
echo "1. HuggingFace Baseline Suite"
echo "=========================================="
modal run modal_app.py::run_hf_suite
echo ""

# vLLM Baseline
echo "=========================================="
echo "2. vLLM Baseline Suite"
echo "=========================================="
modal run modal_app.py::run_vllm_suite
echo ""

# Intervention
echo "=========================================="
echo "3. Intervention Suite"
echo "=========================================="
modal run modal_app.py::run_intervention_suite
echo ""

# Profiling
echo "=========================================="
echo "4. Profiling"
echo "=========================================="
modal run modal_app.py::profile_vllm
echo ""

# Generate Report
echo "=========================================="
echo "5. Generating Report"
echo "=========================================="
modal run modal_app.py::generate_report
echo ""

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to Modal volume /results/"
echo "Report: /results/report.md"
