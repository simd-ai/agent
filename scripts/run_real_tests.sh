#!/bin/bash
# Run real integration tests for simd_agent
#
# Prerequisites:
# 1. Set GEMINI_API_KEY environment variable
# 2. Simulation server should be running at SIMULATION_SERVER_URL
#
# Usage:
#   ./scripts/run_real_tests.sh codegen    # Test AI code generation only
#   ./scripts/run_real_tests.sh full       # Full end-to-end test
#   ./scripts/run_real_tests.sh all        # Run all tests

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}SIMD Agent Real Tests${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check environment
if [ -z "$GEMINI_API_KEY" ]; then
    echo -e "${RED}Error: GEMINI_API_KEY not set${NC}"
    echo "  export GEMINI_API_KEY=your-key-here"
    exit 1
fi

echo -e "Gemini API Key: ${YELLOW}...${GEMINI_API_KEY: -8}${NC}"
echo ""

# Change to project root
cd "$(dirname "$0")/.."

case "${1:-all}" in
    codegen)
        echo -e "${GREEN}Running code generation tests...${NC}"
        pytest tests/test_codegen_real.py -v -s --tb=short
        ;;
    full)
        echo -e "${GREEN}Running full integration test...${NC}"
        pytest tests/test_integration_real.py::TestFullIntegration -v -s --tb=short --timeout=600
        ;;
    linting)
        echo -e "${GREEN}Running linting tests...${NC}"
        pytest tests/test_codegen_real.py::TestRealLinting -v -s --tb=short
        ;;
    all)
        echo -e "${GREEN}Running all real tests...${NC}"
        pytest tests/test_codegen_real.py tests/test_integration_real.py -v -s --tb=short --timeout=600
        ;;
    *)
        echo "Usage: $0 {codegen|full|linting|all}"
        echo ""
        echo "  codegen  - Test AI code generation"
        echo "  full     - Full end-to-end integration test"
        echo "  linting  - Test CFD linting logic"
        echo "  all      - Run all real tests"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Tests complete!${NC}"
echo -e "${GREEN}================================${NC}"
