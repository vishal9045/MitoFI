#!/usr/bin/env bash
# Back up the project and the blind key to the external drive, then verify.
#
# Three drvfs quirks this works around, all of which failed silently once:
#
#   1. rsync cannot create its temp files on this mount ("mkstemp ... Operation
#      not permitted"), so the copy reports success while transferring nothing.
#      --inplace avoids the temp file.
#   2. It cannot set mtimes either, so every later run re-copies everything.
#      Harmless, just slow; --size-only would skip that at the cost of missing a
#      same-size edit, so it is deliberately NOT used here.
#   3. --inplace once appended to an existing file instead of truncating it,
#      doubling its size. Only a checksum pass catches that, so this script
#      always runs one.
#
# Verification is by content, never by exit code: the first backup of this
# project exited 0 having copied nothing.

set -uo pipefail

# Paths follow the project layout, so they survive the folder moving:
#   mitofi/pipeline   this repo
#   mitofi/docs       manuscript, figures, rater package -- not in git
#   mitofi/_BLIND_KEY_do_not_open_until_rating_done
SRC_REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DOCS="$(cd "$SRC_REPO/../docs" && pwd)"
SRC_KEY="$(cd "$SRC_REPO/.." && pwd)/_BLIND_KEY_do_not_open_until_rating_done"
DST="/mnt/f/mito_tool_backup"
FLAGS=(-rlt --inplace --no-perms --no-owner --no-group --omit-dir-times)

[ -d /mnt/f ] && [ -n "$(ls -A /mnt/f 2>/dev/null)" ] || {
    echo "F: is not mounted. Plug the drive in, then:"
    echo "    sudo mount -t drvfs F: /mnt/f"; exit 1; }

echo "== repo =="
rsync "${FLAGS[@]}" --delete --exclude '.venv/' "$SRC_REPO/" "$DST/mitofi/" 2>&1 \
    | grep -v 'failed to set times' | grep -v '^$'

# the manuscript lives outside the repo, so git does not protect it either
echo "== docs =="
rsync "${FLAGS[@]}" --delete "$SRC_DOCS/" "$DST/docs/" 2>&1 \
    | grep -v 'failed to set times' | grep -v '^$'

# The key is small and irreplaceable, so copy it plainly rather than through
# rsync's temp-file path, and compare it byte for byte.
echo "== key =="
cp -rf "$SRC_KEY/." "$DST/_BLIND_KEY_do_not_open_until_rating_done/"
if diff -r "$SRC_KEY" "$DST/_BLIND_KEY_do_not_open_until_rating_done" >/dev/null; then
    echo "key: byte-for-byte identical"
else
    echo "KEY MISMATCH -- do not trust this backup"; diff -rq "$SRC_KEY" \
        "$DST/_BLIND_KEY_do_not_open_until_rating_done"; exit 1
fi

echo "== verify by checksum (reads both sides, takes a few minutes) =="
diffs=$(rsync -rn --checksum --itemize-changes --exclude '.venv/' \
        "$SRC_REPO/" "$DST/mitofi/" 2>/dev/null | grep -c '^>')
src=$(find "$SRC_REPO" -path "$SRC_REPO/.venv" -prune -o -type f -print | wc -l)
dst=$(find "$DST/mitofi" -type f | wc -l)

ddiffs=$(rsync -rn --checksum --itemize-changes "$SRC_DOCS/" "$DST/docs/" \
         2>/dev/null | grep -c '^>')
dsrc=$(find "$SRC_DOCS" -type f | wc -l)
ddst=$(find "$DST/docs" -type f | wc -l)

echo "files: source $src, backup $dst   docs: source $dsrc, backup $ddst"
echo "content differences: repo $diffs, docs $ddiffs"
echo "backup git head: $(git --git-dir="$DST/mitofi/.git" log --oneline -1)"

if [ "$diffs" -eq 0 ] && [ "$src" -eq "$dst" ] \
   && [ "$ddiffs" -eq 0 ] && [ "$dsrc" -eq "$ddst" ]; then
    echo "BACKUP VERIFIED"
else
    echo "BACKUP INCOMPLETE -- investigate before relying on it"; exit 1
fi
