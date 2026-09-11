#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

failures=0

pass() { printf 'PASS: %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1" >&2; failures=$((failures + 1)); }

require_file() {
  if [ -f "$1" ]; then pass "$1"; else fail "missing file $1"; fi
}

require_file AGENTS.md
require_file .github/copilot-instructions.md
require_file .codex/instructions.md
require_file .agents/README.md
require_file .agents/docs/catalog.md
require_file .agents/commands/dwp-create.md
require_file .agents/commands/dwp-execute.md
require_file .agents/commands/dwp-status.md
require_file .agents/commands/dwp-refine.md
require_file .agents/commands/dwp-resume.md
require_file .agents/commands/dwp-verify.md
require_file docs/dwp.md
require_file scripts/bootstrap-dwp.sh

if [ -L .claude ] && [ "$(readlink .claude)" = ".agents" ]; then
  pass ".claude -> .agents"
else
  fail ".claude must be a symlink to .agents"
fi

if [ -L .cursor ] && [ "$(readlink .cursor)" = ".agents" ]; then
  pass ".cursor -> .agents"
else
  fail ".cursor must be a symlink to .agents"
fi

if [ -L CLAUDE.md ] && [ "$(readlink CLAUDE.md)" = "AGENTS.md" ]; then
  pass "CLAUDE.md -> AGENTS.md"
else
  fail "CLAUDE.md must be a symlink to AGENTS.md"
fi

if git check-ignore -q .dwp/example 2>/dev/null; then
  pass ".dwp/ is gitignored"
else
  fail ".dwp/ must be gitignored"
fi

if git check-ignore -q tmp/example 2>/dev/null; then
  pass "tmp/ is gitignored"
else
  fail "tmp/ should be gitignored"
fi

if grep -q 'v4.0.0' AGENTS.md && grep -q 'v4.0.0' docs/dwp.md && grep -q 'DWP_VERSION="v4.0.0"' scripts/bootstrap-dwp.sh; then
  pass "DWP version pin is consistent"
else
  fail "DWP version pin must be v4.0.0 across AGENTS/docs/bootstrap"
fi

if grep -R --line-number --fixed-string '.dwp/drafts/' AGENTS.md .agents 2>/dev/null; then
  fail "operational repository instructions must not use legacy .dwp/drafts/"
else
  pass "no operational legacy drafts workflow"
fi

if [ "$failures" -ne 0 ]; then
  echo "DWP harness verification failed with ${failures} issue(s)." >&2
  exit 1
fi

echo "DWP harness verification passed."
