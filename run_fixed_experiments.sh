#!/bin/bash
# Run fixed experiments with proper statistical methodology
# This script runs the intervention suite with increased sample size and statistical testing

set -e

echo "=========================================="
echo "Running Fixed Intervention Experiment"
echo "=========================================="
echo ""
echo "Fixes applied:"
echo "  ✓ Percentile calculation corrected (proper quantile method)"
echo "  ✓ Sample size increased (32 -> 1000 requests)"
echo "  ✓ Statistical testing added (Mann-Whitney U, effect size)"
echo "  ✓ Bootstrap confidence intervals for P99"
echo "  ✓ Estimated TTFT clearly labeled"
echo ""

# Run intervention suite
echo "Running intervention suite..."
modal run modal_app.py::run_intervention_suite

echo ""
echo "=========================================="
echo "Experiment Complete"
echo "=========================================="
echo ""
echo "Results saved to Modal volume /results/:"
echo "  - /results/intervention_results.json (summary with statistics)"
echo "  - /results/runs/intervention_fifo.jsonl (baseline raw data)"
echo "  - /results/runs/intervention_length_aware.jsonl (intervention raw data)"
echo ""
echo "To view results:"
echo "  modal run modal_app.py::show_results_summary"
echo ""
echo "To download results locally:"
echo "  modal volume download modal-results /results/ ./local_results/"
echo ""
