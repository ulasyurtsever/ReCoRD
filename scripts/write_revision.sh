#!/usr/bin/env bash
# Record the checkout's current commit in REVISION so that copies of this tree
# that are not git checkouts (the working copy, the server) still stamp their
# .meta.json sidecars with a revision instead of "unknown". Run from the git
# checkout after committing; copy the file along with scripts/ and src/.
#
#   bash scripts/write_revision.sh            # writes ./REVISION
#   bash scripts/write_revision.sh /path/to/copy   # writes there as well
set -euo pipefail
cd "$(dirname "$0")/.."
rev=$(git rev-parse --short HEAD 2>/dev/null) || { echo "not a git checkout: $PWD" >&2; exit 1; }
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "warning: working tree has uncommitted changes; REVISION=$rev names the last commit" >&2
fi
printf '%s\n' "$rev" > REVISION
for dest in "$@"; do printf '%s\n' "$rev" > "$dest/REVISION"; done
echo "REVISION=$rev"
