#!/usr/bin/env bash
# rustorch RFC linter — validates front-matter and required sections.
#
# Usage:
#   scripts/lint-rfc.sh                       # check every RFC under docs/rfcs/
#   scripts/lint-rfc.sh docs/rfcs/0001-x.md   # check a single file

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
RFC_DIR="${REPO_ROOT}/docs/rfcs"

# --------------------------------------------------------------------------
# Required sections — each RFC must contain these as level-2 headings.
# --------------------------------------------------------------------------
REQUIRED_SECTIONS=(
    "Summary"
    "Motivation"
    "Constraints"
    "Alternatives Considered"
    "Decision"
    "Rationale"
    "How does PyTorch do it?"
    "How does burn / candle do it?"
    "Migration plan"
    "Open questions"
    "References"
    "Decision matrix"
)

# Required YAML front-matter keys
REQUIRED_FRONTMATTER=(id title status date authors)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
ERR=0
err()  { echo "ERROR: $*" >&2; ERR=1; }
info() { echo "  $*"; }

# Extract front-matter block (between two `---` lines) into a temp file.
extract_frontmatter() {
    local file="$1"
    awk '/^---$/{ c++; if (c==2) exit; next } c==1 { print }' "$file"
}

# Check a single RFC file
check_rfc() {
    local file="$1"
    local name
    name="$(basename "$file")"
    echo "Linting $name..."

    # Skip the template + README + this lint config + .gitkeep
    case "$name" in
        template.md|README.md|.markdownlint.json|.gitkeep) return 0 ;;
    esac

    # File-name pattern: NNNN-slug.md
    if ! [[ "$name" =~ ^[0-9]{4}-[a-z0-9-]+\.md$ ]]; then
        err "$name: filename must match NNNN-slug.md"
    fi

    # Front-matter present
    local fm
    fm="$(extract_frontmatter "$file")"
    if [ -z "$fm" ]; then
        err "$name: missing YAML front-matter (--- block)"
        return
    fi

    # Required keys
    for key in "${REQUIRED_FRONTMATTER[@]}"; do
        if ! echo "$fm" | grep -qE "^${key}:"; then
            err "$name: front-matter missing key '${key}'"
        fi
    done

    # Required sections — use fixed-string match to avoid regex metachars ('?')
    for section in "${REQUIRED_SECTIONS[@]}"; do
        if ! grep -qFx "## ${section}" "$file"; then
            err "$name: missing required section '## ${section}'"
        fi
    done
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if [ $# -gt 0 ]; then
    for f in "$@"; do check_rfc "$f"; done
else
    shopt -s nullglob
    for f in "${RFC_DIR}"/*.md; do
        check_rfc "$f"
    done
fi

# Uniqueness of `id:` across all RFCs
duplicate_ids="$(
    for f in "${RFC_DIR}"/[0-9]*.md; do
        [ -f "$f" ] || continue
        extract_frontmatter "$f" | awk '/^id:/{ print $2 }'
    done | sort | uniq -d
)"
if [ -n "$duplicate_ids" ]; then
    err "duplicate RFC ids detected:"
    echo "$duplicate_ids" >&2
fi

if [ $ERR -ne 0 ]; then
    echo "lint-rfc: failed."
    exit 1
fi
echo "lint-rfc: OK"
