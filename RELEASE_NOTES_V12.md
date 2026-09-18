# v12 — Load-Aware Bandwidth

## Use the update

Extract the new ZIP to a fresh folder, then open a terminal in its inner `GeoVision_AI_Multi_Hazard_Disaster_Model` directory:

```powershell
python -m streamlit run app.py
```

Use the Python environment where the project's dependencies are installed. This update adds no dependency or API key. On Streamlit hosting, update the application files and reboot; keep credentials in the hosting Secrets settings, never in a ZIP or source control. The package does not include local service-account credentials or secrets.

In **Overview → Show bandwidth mask**, use **Load-aware sharing (scenario)** or compare with **Nearest site (baseline)**. **Back to overview** restores the original map. The university/team header, map design, technology controls and measured-download mode remain.

## Changes

- Fractional cohort allocation across up to five nearby, same-radio/single-matching-network site proxies, using full-project demand.
- Correct location-key crosswalk between the distinct population-site and cell-observation ID systems.
- Unknown or ambiguous compatibility stays anchored; no invented cross-network roaming or device upgrades.
- Demand and site-capacity conservation checks; nearest-baseline comparison, shifted-demand percentage and visible convergence diagnostics.
- Fractional population shortfall and assigned-user ranges, so an average cannot conceal the proportion below target.
- Fixed-routing sensitivity explicitly labelled; the log/linear selector still changes colors only.
- A capacity-budget diagnostic explaining when low predictions are unavoidable under the selected inputs.

## Important interpretation

This improves allocation logic, **not validated real-world speed accuracy**. Presets are unchanged. The default full-project mean rises from roughly 0.283 to 0.325 Mbps/user, but the low predictions remain a consequence of the hypothetical capacity/activity inputs. Total assumed capacity permits an optimistic mean ceiling of only about 0.329 Mbps/user in that scenario. It is not evidence that real Yangon users receive these speeds.

The default solver reaches its 120-iteration limit before strict convergence, and the UI discloses this. Its output remains feasible and capacity-conserving; it is not presented as an optimal network configuration. Some users/threshold summaries can worsen when scarce capacity is redistributed. See [method and full audit](BANDWIDTH_LOAD_SHARING.md).

## Verification

The release includes synthetic conservation/compatibility/shortfall regression tests, a real-grid aggregate audit (`scripts/load_sharing_audit.py`), an isolated bandwidth UI check (`scripts/bandwidth_smoke_test.py`) and a full-app check (`scripts/app_smoke_test.py --block-rasterio`). The latter simulates the previously reported terrain DLL block without changing Windows security policy. No live external API success or hosted performance is claimed by those checks.

Verified locally: **396 pytest tests passed** (nine existing NumPy/pandas timedelta deprecation warnings); scoped Ruff checks passed; the actual-grid bandwidth UI check and synthetic-upload check passed; the full app check passed in the user's Anaconda environment, including all four demo hazard workspaces, site checker and return to overview. UI verification was programmatic through Streamlit AppTest, not a browser screenshot review.

Historical release notes describe their own versions; the v12 method supersedes earlier nearest-only assignment descriptions. No trained model, original data file, secret or external service configuration is changed by this update.
