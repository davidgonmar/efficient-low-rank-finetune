#!/usr/bin/env bash
set -euo pipefail

# Format all *.py / *.pyi outside ./balf
find . -path './balf' -prune -o \
  -type f \( -name '*.py' -o -name '*.pyi' \) -print0 \
  | xargs -0 -n 100 black --