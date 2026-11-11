# make help  -> list everything. GCP targets read infra/config.env.
-include infra/config.env
export
REGION ?= us-east1
BUCKET ?= $(PROJECT_ID)-credit-risk
REGISTRY = gs://$(BUCKET)/registry
PY = python

.PHONY: help venv test train promote rollback serve-local traffic-local gcp-setup train-cloud promote-cloud \
        retrain-downturn rollback-cloud deploy schedule smoke traffic traffic-downturn monitor-now report \
        champion logs teardown

help:              ## list commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

# ---------- local (no cloud) ----------
venv:              ## create .venv with pinned dependencies (use Python 3.12)
	python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt && pip install --no-deps -e .

test:              ## run the test suite (no credentials needed)
	pytest

train:             ## train into ./local_registry
	$(PY) -m credit_risk.train

promote:           ## gate the newest local version
	$(PY) -m credit_risk.promote

rollback:          ## local rollback
	$(PY) -m credit_risk.promote --rollback

serve-local:       ## run the score function on localhost:8080 with functions-framework
	REGISTRY_URI=local_registry PREDICTION_SINK=jsonl:logs/predictions.jsonl \
	  PYTHONPATH=src functions-framework --source=function/main.py --target=score --port=8080 --debug

traffic-local:     ## send applications to the local function
	$(PY) scripts/send_traffic.py --url http://localhost:8080 --n 300

# ---------- Google Cloud ----------
gcp-setup:         ## ONE TIME: APIs, bucket, BigQuery, service accounts, IAM, Workload Identity
	bash infra/setup.sh

train-cloud:       ## train a new version into gs://BUCKET/registry
	$(PY) -m credit_risk.train --registry $(REGISTRY)

promote-cloud:     ## gate the newest cloud version; on success production switches within a minute
	$(PY) -m credit_risk.promote --registry $(REGISTRY)

retrain-downturn:  ## retrain with post-shock applications, then gate on recent data
	$(PY) -m credit_risk.train --registry $(REGISTRY) --regime downturn
	$(PY) -m credit_risk.promote --registry $(REGISTRY)

rollback-cloud:    ## point production back at the previous champion
	$(PY) -m credit_risk.promote --registry $(REGISTRY) --rollback

deploy:            ## package + deploy the score and monitor functions
	bash infra/deploy.sh

schedule:          ## ONE TIME after first deploy: daily Cloud Scheduler job for monitor
	bash infra/schedule.sh

smoke:             ## call the deployed score function (GET health + one application)
	@URL=$$(gcloud functions describe score --region=$(REGION) --format='value(serviceConfig.uri)'); \
	TOKEN=$$(gcloud auth print-identity-token); \
	curl -s -H "Authorization: Bearer $$TOKEN" $$URL; echo; \
	curl -s -X POST $$URL -H "Authorization: Bearer $$TOKEN" -H 'Content-Type: application/json' \
	  -d @scripts/sample_applicant.json; echo

traffic:           ## 500 normal applications to the deployed function
	$(PY) scripts/send_traffic.py --region $(REGION) --n 500

traffic-downturn:  ## 500 applications from an economic downturn
	$(PY) scripts/send_traffic.py --region $(REGION) --n 500 --regime downturn

monitor-now:       ## trigger the scheduler job now, then print the report
	gcloud scheduler jobs run credit-risk-monitor --location=$(REGION)
	@echo "waiting for the report..."; sleep 45
	gcloud storage cat gs://$(BUCKET)/reports/monitoring/latest.md

report:            ## print the latest monitoring report
	gcloud storage cat gs://$(BUCKET)/reports/monitoring/latest.md

champion:          ## show which version production serves
	gcloud storage cat $(REGISTRY)/champion.json

logs:              ## recent function logs
	gcloud functions logs read score --region=$(REGION) --limit=30
	gcloud functions logs read monitor --region=$(REGION) --limit=10

teardown:          ## delete functions, scheduler job, bucket and dataset
	bash infra/teardown.sh
