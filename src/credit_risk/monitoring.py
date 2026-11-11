"""
Compare recent production traffic with the champion's training-time profile.

Checks (all standard in credit-risk model monitoring):
  * score PSI           has the distribution of scores moved?
  * feature PSI         which inputs moved (income, DTI, utilisation, inquiries, ...)?
  * approval-rate shift is the policy outcome changing?
  * mean predicted PD   is the book getting riskier?
"""
import numpy as np

from credit_risk import config as C
from credit_risk.model import share_in_bins

EPS = 1e-4


def psi(expected, actual):
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value, warn, alert):
    return "alert" if value >= alert else "warn" if value >= warn else "ok"


def analyse(profile, records):
    if len(records) < C.MIN_MONITOR_ROWS:
        return {"n": len(records), "alerts": [], "status": "insufficient_data",
                "note": f"need at least {C.MIN_MONITOR_ROWS} scored applications"}
    scores = [r["score"] for r in records]
    pds = np.array([r["pd"] for r in records], float)
    checks = {}
    v = psi(profile["score_share"], share_in_bins(scores, profile["score_edges"]))
    checks["score_psi"] = {"value": round(v, 4), "status": status(v, C.PSI_WARN, C.PSI_ALERT)}
    for col, ref in profile["features"].items():
        vals = [r["features"][col] for r in records if col in r["features"]]
        if vals:
            v = psi(ref["share"], share_in_bins(vals, ref["edges"]))
            checks[f"psi_{col}"] = {"value": round(v, 4), "status": status(v, C.PSI_WARN, C.PSI_ALERT)}
    approval = float(np.mean(pds < C.APPROVE_BELOW_PD))
    shift = abs(approval - profile["approval_rate"])
    checks["approval_rate_shift"] = {"value": round(shift, 4), "live": round(approval, 4),
                                     "training": round(profile["approval_rate"], 4),
                                     "status": "alert" if shift >= C.APPROVAL_RATE_SHIFT_ALERT else "ok"}
    checks["mean_pd"] = {"value": round(float(pds.mean()), 4), "training": round(profile["mean_pd"], 4),
                         "status": "info"}
    alerts = [k for k, c in checks.items() if c["status"] == "alert"]
    return {"n": len(records), "model_versions": sorted({r["model_version"] for r in records}),
            "checks": checks, "alerts": alerts, "status": "alert" if alerts else "ok"}


def to_markdown(result, champion, window_hours):
    if result["status"] == "insufficient_data":
        return f"# Monitoring report\n\n{result['note']} (got {result['n']} in the last {window_hours} h).\n"
    icon = {"ok": "🟢", "warn": "🟡", "alert": "🔴", "info": "ℹ️"}
    lines = ["# Monitoring report", "",
             (f"Champion **{champion}** · {result['n']} applications in the last {window_hours} h · "
              f"versions seen: {', '.join(result['model_versions'])}"), "",
             "| Check | Value | Status |", "|---|---|---|"]
    for name, c in result["checks"].items():
        extra = f" (live {c['live']} vs training {c['training']})" if "live" in c else (
            f" (training {c['training']})" if "training" in c else "")
        lines.append(f"| {name} | {c['value']}{extra} | {icon[c['status']]} {c['status']} |")
    lines.append("")
    lines.append("**🔴 Action needed:** " + ", ".join(result["alerts"]) +
                 ". Investigate the shift; if it is a real change in the applicant population, retrain on "
                 "recent data (`train --regime downturn`) and let the gate compare on recent applications."
                 if result["alerts"] else "**🟢 No action needed.**")
    return "\n".join(lines) + "\n"
