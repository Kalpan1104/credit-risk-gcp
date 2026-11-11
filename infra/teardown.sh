#!/usr/bin/env bash
# Delete everything this project created (so nothing keeps costing money).
source "$(dirname "$0")/common.sh"
read -r -p "Delete functions, scheduler job, bucket gs://${BUCKET} and dataset ${DATASET} in ${PROJECT_ID}? [y/N] " ok
[[ "$ok" == "y" ]] || exit 0
gcloud scheduler jobs delete credit-risk-monitor --location="$REGION" --quiet || true
gcloud functions delete score --region="$REGION" --quiet || true
gcloud functions delete monitor --region="$REGION" --quiet || true
gcloud storage rm --recursive "gs://${BUCKET}" || true
bq rm -r -f "${PROJECT_ID}:${DATASET}" || true
echo "Done. Service accounts and the Workload Identity pool are left in place (they cost nothing)."
