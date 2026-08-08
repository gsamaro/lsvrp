#!/usr/bin/env zsh

set -euo pipefail

source "$HOME/.zshrc"

exec poetry run python main.py
