#!/usr/bin/env bash
# Let Cloud Scheduler call the monitor function every day. Run once after the first deploy.
source "$(dirname "$0")/common.sh"
MONITOR_URL="$(gcloud functions describe monitor --region="$REGION" --format='value(serviceConfig.uri)')"
[[ -n "$MONITOR_URL" ]] || { echo "Deploy the functions first (make deploy)"; exit 1; }

echo "==> Allowing the scheduler account to invoke monitor"
gcloud functions add-invoker-policy-binding monitor --region="$REGION" \
  --member="serviceAccount:${SCHEDULER_SA}" >/dev/null

args=(--location="$REGION" --schedule="0 8 * * *" --time-zone="America/New_York"
      --uri="$MONITOR_URL" --http-method=POST
      --oidc-service-account-email="$SCHEDULER_SA" --oidc-token-audience="$MONITOR_URL")
if gcloud scheduler jobs describe credit-risk-monitor --location="$REGION" >/dev/null 2>&1; then
  gcloud scheduler jobs update http credit-risk-monitor "${args[@]}"
else
  gcloud scheduler jobs create http credit-risk-monitor "${args[@]}"
fi
echo "Daily monitoring scheduled for 08:00 New York time. Run it now with: make monitor-now"
