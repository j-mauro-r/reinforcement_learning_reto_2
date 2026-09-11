#!/usr/bin/env bash
set -euo pipefail

DWP_VERSION="v4.0.0"
DWP_REPO="https://github.com/DailybotHQ/deepworkplan-skill.git"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dwp-install.XXXXXX")"

cleanup() {
  rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

for command in git bash; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $command" >&2
    exit 1
  fi
done

cd "$ROOT"
mkdir -p .dwp/plans tmp

echo "Installing Deep Work Plan ${DWP_VERSION}..."
git clone --quiet --depth 1 --branch "$DWP_VERSION" "$DWP_REPO" "$TEMP_DIR/deepworkplan-skill"

(
  cd "$TEMP_DIR/deepworkplan-skill"
  ./setup.sh --host copilot

  if command -v codex >/dev/null 2>&1 || [ -d "${HOME}/.codex" ]; then
    ./setup.sh --host codex
  fi
)

echo "Created local workspaces: .dwp/plans/ and tmp/"

if [ -x "$ROOT/scripts/verify-dwp-harness.sh" ]; then
  "$ROOT/scripts/verify-dwp-harness.sh"
else
  bash "$ROOT/scripts/verify-dwp-harness.sh"
fi

echo "Deep Work Plan ${DWP_VERSION} is configured for this repository."
