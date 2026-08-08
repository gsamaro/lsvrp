#!/usr/bin/env zsh

set -euo pipefail

source "$HOME/.zshrc"

PYTHON_BIN="${PYTHON_BIN:-/Users/gabamaro/Library/Caches/pypoetry/virtualenvs/multi-product-production-routing-problem-1qEUYht3-py3.13/bin/python}"

exec "$PYTHON_BIN" main.py
