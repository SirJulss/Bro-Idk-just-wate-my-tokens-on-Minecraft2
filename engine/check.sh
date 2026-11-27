#!/bin/sh
# Acceptance check for the whole engine workspace (contract section 3.4).
# Lead / acceptance only: everybody else uses the crate-scoped cargo commands.
#
# Runs the steps below in order, stops at the first one that fails and exits with that step's
# exit code (0 when all pass). POSIX sh; run it as "sh check.sh" if the file is not executable.

set -eu

# Operate from the directory of this script, whatever the caller's working directory is.
CDPATH='' cd -- "$(dirname -- "$0")"

run() {
    echo "==> $*"
    "$@"
}

run cargo fmt --all -- --check
run cargo clippy --workspace --all-targets -- -D warnings
run cargo test --workspace

echo "check.sh: all checks passed"
