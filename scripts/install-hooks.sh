#!/usr/bin/env bash
# Install rustorch git hooks for the current clone.
# Idempotent — safe to re-run after pulling new hooks.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="${REPO_ROOT}/.githooks"

if [ ! -d "${HOOKS_DIR}" ]; then
    echo "Error: ${HOOKS_DIR} not found. Are you inside the rustorch repository?"
    exit 1
fi

# Ensure all hook scripts are executable
chmod +x "${HOOKS_DIR}"/*

# Point git to the in-repo hooks directory
git config core.hooksPath "${HOOKS_DIR}"

echo "rustorch hooks installed."
echo "  pre-commit  → cargo fmt --check + cargo clippy"
echo "  commit-msg  → Conventional Commits validation"
echo
echo "Bypass any hook with: git commit --no-verify"
echo "Uninstall with:        git config --unset core.hooksPath"
