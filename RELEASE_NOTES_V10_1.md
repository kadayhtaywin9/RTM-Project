# GeoVision AI v10.1 — Optional Terrain Startup Fix

## Cause and scope

The reported startup traceback ended in `site_ai.py → rasterio → rasterio._base`: Windows Application Control blocked a native library before Streamlit could render the dashboard. This is separate from GEE authentication or the cyclone feed. v10.1 keeps the v10 bandwidth, map and header features and isolates this optional terrain dependency.

## Changes

- Rasterio is imported only when a selected-coordinate terrain sample is requested. Missing-package, DLL import and OS loading errors return an unavailable capability; they no longer prevent the whole dashboard from starting. Failures are cached until process restart.
- The Windows launcher check treats Rasterio as optional, printing a capability warning instead of entering dependency-repair/startup-failure handling. Core dependency failures remain fatal. The regular requirements file still includes Rasterio for installations that support it.
- When a terrain sample is unavailable, the existing area-candidate elevation proxy may still be used. Its provenance is now visible in the warning, elevation metric, model-input table and downloaded technical report. It must not be interpreted as a fresh point sample or a field measurement. If neither a usable sample nor a valid stored terrain feature exists, the site score is withheld rather than inventing a zero.
- Raster sampling rejects masked, nodata, nonfinite and out-of-raster values. A genuine valid zero-metre elevation remains valid.
- No security settings, DLL allowlists, Anaconda packages, credentials, GEE initialization or model artifacts are changed. The blocked DLL remains blocked. No accuracy improvement is claimed for a proxy-backed score.

Overview, the two bandwidth evidence modes, stored candidate ranking and the local hazard workspaces do not require this terrain reader. Their existing scientific limitations still apply. Live GEE/JTWC/USGS availability is a separate concern and is not certified by the offline regression tests.

## Verification

- Full pytest suite: **304 passed**, with nine pre-existing NumPy/pandas timedelta deprecation warnings in earthquake/rainfall tests.
- Scoped Ruff checks passed for the modified model adapter, runtime checker, app smoke script and startup/terrain tests.
- Full Streamlit smoke test passed in the project environment: bandwidth-only/normal map transitions, operator and evidence-mode changes, persistent assumptions, and all four local hazard workspaces.
- The full Streamlit smoke test also passed with `--block-rasterio`, simulating an Application Control import failure without modifying OS policy. The site-checker warning, area-proxy metric and report control rendered, and bandwidth and all four local hazard workspaces remained functional. Unit tests check cached failure handling, explicit proxy provenance, safe report text, valid/missing terrain and the absence of mandatory Rasterio imports at startup.
- The updated site adapter, model metadata and proxy-backed site assessment were also checked successfully with the user's Anaconda Python executable. The full Streamlit smoke test then passed in that Anaconda environment with `--block-rasterio`, including the site checker, bandwidth controls and all four local hazard workspaces.

## Install

1. Stop the old Streamlit process with **Ctrl+C** in its terminal.
2. Extract the complete v10.1 ZIP into a new folder. This preserves the old release and any local configuration.
3. Open your Anaconda Prompt in the extracted `GeoVision_AI_Multi_Hazard_Disaster_Model` folder containing `app.py`.
4. Run `python -m streamlit run app.py` and keep the terminal open. Use the URL printed by that process.

Existing compatible dependencies are sufficient; reinstalling into Anaconda base is not needed for this fix. If you keep credentials in a local secrets file, retain them securely in your own deployment; no real secrets are included in the release. Streamlit Cloud needs the updated files committed to its configured repository; this ZIP does not automatically update a hosted app.

For exact point terrain sampling, ask the device administrator to investigate the blocked file through the approved software/policy process. Microsoft documents the relevant **CodeIntegrity → Operational** events (including 3077 and associated 3089 signature details) in its [Application Control troubleshooting guide](https://learn.microsoft.com/en-us/windows/security/application-security/application-control/app-control-for-business/operations/appcontrol-debugging-and-troubleshooting). Do not disable Windows protection to launch this app.

Runtime changes: `site_ai.py`, `app.py`, `scripts/check_runtime.py`, and the explanatory comment in `requirements.txt`. New/updated regression coverage: `tests/test_optional_terrain.py`, `tests/test_startup.py`, `scripts/app_smoke_test.py`.
