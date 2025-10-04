#!/usr/bin/env bash
set -euo pipefail

# Format all *.py / *.pyi outside ./SVD-LLM
find . -path './SVD-LLM' -prune -o \
  -type f \( -name '*.py' -o -name '*.pyi' \) -print0 \
  | xargs -0 -n 100 black --