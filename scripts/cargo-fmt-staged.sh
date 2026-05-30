#!/usr/bin/env bash
# Auto-format staged Rust files with rustfmt and re-stage them, so every commit
# is rustfmt-clean without mixing formatting churn into logic commits.
# Invoked by prek/pre-commit (see .pre-commit-config.yaml); prek passes the
# staged *.rs paths as positional arguments.
set -euo pipefail

[ "$#" -eq 0 ] && exit 0

# Resolve rustfmt even when ~/.cargo/bin is absent from PATH (e.g. git GUIs,
# or shells that don't source the cargo env).
if command -v rustfmt >/dev/null 2>&1; then
  RUSTFMT=rustfmt
elif [ -x "$HOME/.cargo/bin/rustfmt" ]; then
  RUSTFMT="$HOME/.cargo/bin/rustfmt"
else
  echo "rustfmt not found (install with: rustup component add rustfmt)" >&2
  exit 1
fi

"$RUSTFMT" --edition 2021 "$@"
git add -- "$@"
