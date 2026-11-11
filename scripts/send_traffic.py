"""
Send synthetic loan applications to the score function.

    python scripts/send_traffic.py --n 500                      # deployed function, normal economy
    python scripts/send_traffic.py --n 500 --regime downturn    # after an economic shock
    python scripts/send_traffic.py --url http://localhost:8080  # local functions-framework, no auth
"""
import argparse
import subprocess
import sys
import time
from collections import Counter

import requests

sys.path.insert(0, "src")
from credit_risk import config as C
from credit_risk.data import generate


def gcloud(*args):
    return subprocess.check_output(["gcloud", *args], text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", help="defaults to the deployed score function")
    p.add_argument("--region", default="us-east1")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--regime", choices=["normal", "downturn"], default="normal")
    a = p.parse_args()

    headers = {}
    url = a.url
    if not url:
        url = gcloud("functions", "describe", "score", f"--region={a.region}", "--format=value(serviceConfig.uri)")
    if "localhost" not in url and "127.0.0.1" not in url:
        headers["Authorization"] = f"Bearer {gcloud('auth', 'print-identity-token')}"

    df = generate(a.n, seed=int(time.time()) % 100_000, regime=a.regime).drop(columns=[C.TARGET])
    decisions, versions = Counter(), Counter()
    start = time.time()
    for i in range(0, len(df), a.batch):
        r = requests.post(url, json={"applicants": df.iloc[i:i + a.batch].to_dict("records")}, headers=headers, timeout=60)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}: {r.text[:300]}")
            sys.exit(1)
        body = r.json()
        versions[body["model_version"]] += 1
        decisions.update(res["decision"] for res in body["results"])
    took = time.time() - start
    total = sum(decisions.values())
    print(f"scored {total} {a.regime} applications in {took:.1f}s via {url}")
    print("  model versions:", dict(versions))
    print("  decisions:", {k: f"{v} ({v / total:.0%})" for k, v in decisions.most_common()})


if __name__ == "__main__":
    main()
