# Shared settings for the infra scripts. Sourced, not run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$HERE/config.env" ]]; then source "$HERE/config.env"; fi
: "${PROJECT_ID:?Set PROJECT_ID in infra/config.env}"
: "${REGION:=us-east1}"

BUCKET="${BUCKET:-${PROJECT_ID}-credit-risk}"
DATASET="credit_risk"
TABLE="${PROJECT_ID}.${DATASET}.predictions"
REGISTRY_URI="gs://${BUCKET}/registry"
REPORTS_URI="gs://${BUCKET}/reports"

RUNTIME_SA="credit-risk-fn@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOYER_SA="credit-risk-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_SA="credit-risk-scheduler@${PROJECT_ID}.iam.gserviceaccount.com"
POOL="github"
PROVIDER="github-provider"

gcloud config set project "$PROJECT_ID" >/dev/null
PROJECT_NUMBER="${PROJECT_NUMBER:-$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')}"
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
