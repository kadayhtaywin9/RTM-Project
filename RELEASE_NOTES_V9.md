# GeoVision AI v9 — Bandwidth Scenario & Telecom Review

## Overview map

- Click **Show bandwidth mask** to replace the standard Overview overlays with a continuous red–yellow–green scale on filled geographic cells. Tower markers, candidate points, recommendations, study-boundary traces and click-details are absent in this mode. OpenStreetMap stays underneath for orientation.
- Click **Back to overview** to restore the original map. The bandwidth assumptions survive a round trip through the normal view within the session.
- Adjust **Assumed effective capacity / site (Mbps)**, **Assumed active population (%)** and **User target (Mbps)**. The defaults (100 Mbps/site, 5% active population and 10 Mbps/user target) are illustrative assumptions, not operator measurements or validated local engineering values.
- Red means low/zero scenario throughput; yellow is half the target; green is at or above the target. The target changes comparison and color saturation, never the calculated throughput. Hover values can exceed the legend's green endpoint.
- Maps use population-weighted means in 1 km display cells clipped to the selected study boundary. Hover shows the pixel range, population below target, population outside the assumed radius, and 2020 population context. Locations without populated display cells remain uncolored/unknown. This is an aggregate display, not a claim of uniform performance throughout a square kilometre.

## Meaning and assumptions

The dataset contains no measured MHz allocation, throughput, busy-hour traffic, active users, SINR, PRB utilization or backhaul capacity. This feature is therefore explicitly a **capacity-sharing scenario**, not a live bandwidth map or another AI model.

Each population pixel uses its nearest site proxy if it falls inside the sidebar service radius. Per-site population is summed over the **entire packaged project** before choosing which area to display, so filtering the map cannot manufacture spare capacity.

```text
expected active users = full-project served catchment population × active percentage / 100
scenario Mbps per active user = assumed site Mbps / max(1, expected active users)
```

Every eligible pixel assigned to the same site shares that rate. Per-site allocated throughput cannot exceed its assumed budget. Pixels outside the radius receive zero only within this hypothetical scenario, not an observed outage label. Sites without population demand have no supported estimate. Demand outside the packaged project boundary remains unknown.

All operator/technology observations are pooled and all site proxies are assumed active and independent with equal usable capacity. There is no RF, interference, sector, operator-eligibility, shared transport, power or failure simulation. Counting several observations of one actual site as independent budgets could overstate supply. More accurate bandwidth mapping requires verified operator measurements and engineering data.

## Expert review and remaining work

`TELECOM_EXPERT_REVIEW.md` combines code, model and telecom audits. The major open issues include authored/synthetic training targets, absent observed-outage validation, an Admin-4 code collision affecting site lookup, missing RF/operator/dependency constraints, raw public exception details, stale retained results and unquantified uncertainty.

This release adds the map and review; it does **not** claim to correct those pre-existing issues or retrain/validate the AI. Public credential-bearing errors in particular need remediation before wider public use. GEE authentication behavior is unchanged. The new bandwidth scenario needs no external API credentials.

## Validation

On 9 September 2026, **259 regression tests passed**, including 43 new scenario tests for capacity conservation, full-catchment demand, radius boundaries, missing values, geographic clipping, weighted aggregation, legend limits and invalid assumptions. Nine pre-existing NumPy/pandas timedelta deprecation warnings remain.

Streamlit smoke tests passed for the bandwidth toggle, restored overview, adjustable/persistent inputs and all four local hazard workspaces. The latter rendered in approximately 4.5–5.5 seconds per workspace in this environment. Scoped Ruff checks passed. Browser inspection confirmed the colored mask and absence of the usual markers, including in the narrow in-app preview.

These checks establish software behavior, not accuracy against real service measurements, authenticated hosted GEE access or operational readiness.

## Run or deploy

Extract into a new folder and double-click `run_windows.bat`, or run `python -m streamlit run app.py` in an environment with `requirements.txt` installed. Keep the Streamlit process running. Restart an existing process after upgrading to avoid old imported modules.

For an existing v8 deployment, the map change requires `app.py`, `engine/bandwidth_scenario.py` and `ui/bandwidth_map.py`. No new dependency is added. Local changes do not automatically update a Streamlit-hosted GitHub repository; commit/upload the updated project to your deployment repository and reboot the hosted app. Keep credentials only in Streamlit Secrets.
