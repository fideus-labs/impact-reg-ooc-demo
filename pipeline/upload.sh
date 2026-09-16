#!/usr/bin/env bash
# The competition pair's OME-Zarr stores to the public Filebase bucket, each under the key it has in this tree: the
# converted inputs (data/ome-zarr/<subject>/*.ome.zarr), the pair (out/<subject>/pair/*.ome.zarr) and, for every
# preset of the README's table, the moved FA and the displacement field
# (out/<subject>/<preset>/P000/{Moved,Transform}.ome.zarr). s5cmd syncs each store: a chunk that is already there,
# with the same size and a later date, is not sent again.
#   FILEBASE_KEY=... FILEBASE_SECRET=... bash pipeline/upload.sh [--dry-run] [SUBJECT ...]      (pixi run upload)
# FILEBASE_BUCKET and FILEBASE_ENDPOINT replace the bucket and the S3 endpoint it is served from.
set -euo pipefail
cd "$(dirname "$0")/.."
dry_run=(); if [ "${1:-}" = "--dry-run" ]; then dry_run=(--dry-run); shift; fi
: "${FILEBASE_KEY:?the Filebase access key}" "${FILEBASE_SECRET:?the Filebase secret}"
export AWS_ACCESS_KEY_ID=$FILEBASE_KEY AWS_SECRET_ACCESS_KEY=$FILEBASE_SECRET AWS_REGION=${AWS_REGION:-us-east-1}
bucket=${FILEBASE_BUCKET:-impact-reg-ooc-demo}
endpoint=${FILEBASE_ENDPOINT:-https://s3.filebase.io}
presets=(APEX_FA_RET_BSPLINE APEX_FIREANTS_SYN_CC APEX_FIREANTS_TS APEX_CONVEXADAM_MIND)
subjects=("$@"); [ ${#subjects[@]} -gt 0 ] || subjects=(subject_v subject_m)
shopt -s nullglob
missing=0
for subject in "${subjects[@]}"; do
  stores=(data/ome-zarr/"$subject"/*.ome.zarr out/"$subject"/pair/*.ome.zarr)
  for preset in "${presets[@]}"; do
    stores+=(out/"$subject"/"$preset"/P000/Moved.ome.zarr out/"$subject"/"$preset"/P000/Transform.ome.zarr)
  done
  for store in "${stores[@]}"; do
    if [ ! -d "$store" ]; then echo "== $store: not here, skipped"; missing=$((missing + 1)); continue; fi
    echo "== $store -> s3://$bucket/$store/"
    s5cmd "${dry_run[@]}" --endpoint-url "$endpoint" sync "$store/" "s3://$bucket/$store/"
  done
done
[ "$missing" -eq 0 ] || { echo "$missing stores were not here (pixi run register-all writes them)"; exit 1; }
