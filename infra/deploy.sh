#!/usr/bin/env bash
# Package the code and deploy both Cloud Run functions. Used locally and by GitHub Actions.
source "$(dirname "$0")/common.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build/function"

echo "==> Bundling function source"
rm -rf "$BUILD" && mkdir -p "$BUILD"
cp "$ROOT/function/main.py" "$ROOT/function/requirements.txt" "$BUILD/"
cp -r "$ROOT/src/credit_risk" "$BUILD/credit_risk"
find "$BUILD" -name "__pycache__" -prune -exec rm -rf {} +

ENV_VARS="REGISTRY_URI=${REGISTRY_URI},PREDICTION_SINK=bigquery:${TABLE},REPORTS_URI=${REPORTS_URI}"
common=(--gen2 --region="$REGION" --runtime=python312 --source="$BUILD" --trigger-http
        --no-allow-unauthenticated --service-account="$RUNTIME_SA" --set-env-vars="$ENV_VARS"
        --min-instances=0 --max-instances=3 --quiet)

echo "==> Deploying score"
gcloud functions deploy score --entry-point=score --memory=1Gi --timeout=60s "${common[@]}"
echo "==> Deploying monitor"
gcloud functions deploy monitor --entry-point=monitor --memory=1Gi --timeout=300s "${common[@]}"

for fn in score monitor; do
  echo "$fn: $(gcloud functions describe "$fn" --region="$REGION" --format='value(serviceConfig.uri)')"
done
