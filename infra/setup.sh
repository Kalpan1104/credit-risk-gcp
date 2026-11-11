#!/usr/bin/env bash
# One-time Google Cloud setup. Safe to re-run: existing resources are left alone.
# Run from the repo root:  bash infra/setup.sh
source "$(dirname "$0")/common.sh"
: "${GITHUB_REPO:?Set GITHUB_REPO (owner/name) in infra/config.env}"

step() { echo; echo "==> $*"; }
exists() { "$@" >/dev/null 2>&1; }

step "1/7 Enabling APIs (takes a minute the first time)"
gcloud services enable \
  cloudfunctions.googleapis.com run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com storage.googleapis.com bigquery.googleapis.com \
  cloudscheduler.googleapis.com logging.googleapis.com compute.googleapis.com \
  iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com \
  cloudresourcemanager.googleapis.com

step "2/7 Cloud Storage bucket gs://${BUCKET} (model registry + reports)"
exists gcloud storage buckets describe "gs://${BUCKET}" || \
  gcloud storage buckets create "gs://${BUCKET}" --location="$REGION" --uniform-bucket-level-access
gcloud storage buckets update "gs://${BUCKET}" --versioning >/dev/null   # every overwrite of champion.json is recoverable

step "3/7 BigQuery dataset + partitioned predictions table"
exists bq show "${PROJECT_ID}:${DATASET}" || bq --location=US mk --dataset "${PROJECT_ID}:${DATASET}"
exists bq show "${PROJECT_ID}:${DATASET}.predictions" || \
  bq mk --table --time_partitioning_field=ts --time_partitioning_type=DAY \
     --description="Scored loan applications from the score function" \
     "${PROJECT_ID}:${DATASET}.predictions" "$(dirname "$0")/bq_schema.json"

step "4/7 Service accounts"
for sa in credit-risk-fn:"Runtime for score/monitor functions" \
          credit-risk-deployer:"GitHub Actions deploy + retrain" \
          credit-risk-scheduler:"Cloud Scheduler caller for monitor"; do
  name="${sa%%:*}"; desc="${sa#*:}"
  exists gcloud iam service-accounts describe "${name}@${PROJECT_ID}.iam.gserviceaccount.com" || \
    gcloud iam service-accounts create "$name" --display-name="$desc"
done
sleep 10   # new service accounts take a few seconds to become usable in IAM bindings

step "5/7 Permissions (least privilege for each account)"
bind_project() { gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$1" --role="$2" --condition=None >/dev/null; }
bind_bucket()  { gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member="serviceAccount:$1" --role="$2" >/dev/null; }
# runtime: read models / write reports in the bucket, stream + query predictions
bind_bucket  "$RUNTIME_SA"  roles/storage.objectAdmin
bind_project "$RUNTIME_SA"  roles/bigquery.dataEditor
bind_project "$RUNTIME_SA"  roles/bigquery.jobUser
bind_project "$RUNTIME_SA"  roles/logging.logWriter
# deployer (GitHub Actions): deploy functions, act as the runtime + build accounts, write the registry
bind_project "$DEPLOYER_SA" roles/cloudfunctions.developer
bind_project "$DEPLOYER_SA" roles/run.developer
bind_bucket  "$DEPLOYER_SA" roles/storage.objectAdmin
for target in "$RUNTIME_SA" "$COMPUTE_SA"; do
  gcloud iam service-accounts add-iam-policy-binding "$target" \
    --member="serviceAccount:${DEPLOYER_SA}" --role=roles/iam.serviceAccountUser >/dev/null
done
# builds: since mid-2024 new projects build functions as the default compute account, which needs this role
bind_project "$COMPUTE_SA" roles/cloudbuild.builds.builder

step "6/7 Workload Identity Federation (GitHub Actions logs in without a key file)"
exists gcloud iam workload-identity-pools describe "$POOL" --location=global || \
  gcloud iam workload-identity-pools create "$POOL" --location=global --display-name="GitHub Actions"
exists gcloud iam workload-identity-pools providers describe "$PROVIDER" --location=global --workload-identity-pool="$POOL" || \
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
    --location=global --workload-identity-pool="$POOL" --display-name="GitHub" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository=='${GITHUB_REPO}'"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GITHUB_REPO}" >/dev/null

step "7/7 Values for GitHub (Settings -> Secrets and variables -> Actions -> Variables)"
WIF_PROVIDER="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"
cat <<OUT

  GCP_PROJECT_ID      = ${PROJECT_ID}
  GCP_PROJECT_NUMBER  = ${PROJECT_NUMBER}
  GCP_REGION          = ${REGION}
  GCS_BUCKET          = ${BUCKET}
  WIF_PROVIDER        = ${WIF_PROVIDER}
  DEPLOYER_SA         = ${DEPLOYER_SA}

  Or set them from this terminal with the GitHub CLI:
    gh variable set GCP_PROJECT_ID --body "${PROJECT_ID}"
    gh variable set GCP_PROJECT_NUMBER --body "${PROJECT_NUMBER}"
    gh variable set GCP_REGION     --body "${REGION}"
    gh variable set GCS_BUCKET     --body "${BUCKET}"
    gh variable set WIF_PROVIDER   --body "${WIF_PROVIDER}"
    gh variable set DEPLOYER_SA    --body "${DEPLOYER_SA}"

Setup complete. Next: make train-cloud && make promote-cloud && make deploy
OUT
