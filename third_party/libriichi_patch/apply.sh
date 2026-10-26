#!/usr/bin/env bash
# Apply the libriichi patches to a Mortal checkout with `git apply`.
#
#   bash apply.sh <mortal_repo> [all|0001|0002]      (default: all, in order)
#
#   0001-take-obs-raw.diff    data side: RawObsEvent + PlayerState::snapshot_raw() + Gameplay::take_obs_raw()
#   0002-mortal2-engine.diff  eval side: engine_type "mortal2" for the paper's LLMEngine (needs 0001)
#
# Both diffs are against upstream Mortal 0cff2b5. Each one is checked before it is applied, so the tree is
# never left half-patched; a diff that is already applied is skipped, so re-running is safe.
# To undo:  git -C <mortal_repo> apply -R <diff>   (0002 first, then 0001)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:?usage: apply.sh <mortal_repo> [all|0001|0002]}"
WHICH="${2:-all}"
BASE=0cff2b5

case "$WHICH" in
    all)  DIFFS=(0001-take-obs-raw 0002-mortal2-engine) ;;
    0001) DIFFS=(0001-take-obs-raw) ;;
    0002) DIFFS=(0001-take-obs-raw 0002-mortal2-engine) ;;   # 0002 builds on 0001
    *) echo "unknown patch set: $WHICH" >&2; exit 2 ;;
esac

cd "$REPO"
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "$REPO is not a git checkout" >&2; exit 1
fi
head=$(git rev-parse --short=7 HEAD)
[ "$head" = "$BASE" ] || echo "warning: HEAD is $head, the diffs were made against $BASE"

for d in "${DIFFS[@]}"; do
    f="$HERE/$d.diff"
    if git apply --reverse --check "$f" 2>/dev/null; then
        echo "= $d: already applied, skipping"
    elif git apply --check "$f"; then
        git apply "$f"
        echo "+ $d: applied"
    else
        echo "! $d: does not apply cleanly (tree differs from $BASE, or partly patched)" >&2
        exit 1
    fi
done
