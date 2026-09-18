# GeoVision AI v11 — Technology-Based Bandwidth

## Requested change

Bandwidth operator selectors have been removed. The planning map now supports automatic recorded-technology assignment, GSM/2G, UMTS/3G and LTE/4G. NR/5G support is available only if NR/5G records are supplied; none are present in the bundled site inventory. Other dashboard tabs retain their inventory information. No external account, hosted deployment or local credential was changed.

## Behaviour

- The single uniform capacity control is replaced with editable per-technology shared site budgets: GSM 0.2, UMTS 10, LTE 100 Mbps. These assume EDGE/HSPA where stated; they are scenario starting points, not measured or standardized cell rates.
- Automatic assignment counts a mixed-technology site once, using its highest recorded generation. Selecting one technology recomputes catchments using only matching sites and uses that technology's budget even at mixed sites. Cell-observation counts never multiply capacity.
- Per-site budget arrays preserve unique site IDs and full-project demand. Unknown radio/capacity remains unassessed, not zero Mbps. Unknown-nearest sites in automatic mode are not silently reassigned to more distant recognized sites.
- The display still uses the original basemap, gradient and 500 m/1 km aggregation choices. The independently editable per-user target remains 10 Mbps by default; it is not the total site capacity.
- Chosen technology, individual capacity presets, active percentage, target and display resolution persist across evidence-mode and overview transitions.
- The measured-speed view also filters by radio technology, with a new `radio` column in its empty CSV template. It uses supplied readings only. An old operator-only upload is not sufficient evidence of the technology used by a test. Technology readings are pooled across networks, explicitly labelled as such.
- The previous optional Rasterio loading/startup fix, GEE setup, hazard modes, university/team header and other map functions are retained. No AI model was retrained and no current operator capacity or speed dataset was added.

See [BANDWIDTH_TECHNOLOGY_ASSUMPTIONS.md](BANDWIDTH_TECHNOLOGY_ASSUMPTIONS.md) for the exact formulas, preset caveats, inventory counts and primary-source context. Scenario changes are not proof of increased scientific accuracy or actual network performance.

## Run

Stop your old Streamlit process with Ctrl+C, extract this complete ZIP into a new folder, and run `python -m streamlit run app.py` from the extracted project folder containing `app.py`. Keep the terminal open. Existing compatible dependencies suffice; none were added. Your previous release is left untouched. Real secrets are not packaged; preserve your own local/cloud configuration securely.

Streamlit Cloud needs the updated files committed to its configured repository and the app rebooted; a local ZIP does not update a hosted app automatically.

## Changed runtime files

`engine/bandwidth_radio.py` is new. `engine/bandwidth_scenario.py` now accepts per-site capacity arrays. `engine/bandwidth_evidence.py` adds radio-aware measurement validation/aggregation while retaining legacy programmatic operator helpers for compatibility. `ui/bandwidth_evidence_map.py` uses only radio selectors and the new template. Documentation and regression/smoke tests are updated; model files are unchanged.

## Verification (10 September 2026)

- Full automated suite: **337 tests passed**. Nine existing NumPy/pandas timedelta deprecation warnings remain in earthquake/rainfall tests.
- Bandwidth subset: **98 tests passed**, including 33 new technology/preset/measurement cases. Tests cover variable-budget conservation, full-project demand, stable IDs, mixed radios, unknown capacity, unavailable technologies, aliases and technology-filtered measured evidence.
- Scoped Ruff checks passed for all changed bandwidth runtime code and regression/smoke scripts.
- The isolated Streamlit test passed on the actual population grid for automatic/GSM/UMTS/LTE views, persistent capacity presets, gray unsupported locations and evidence-mode changes. Synthetic test-only uploads verified technology filtering, sample counts and the measured map; these fixtures are not shipped as measured project data.
- The full Streamlit smoke test passed using the user's Anaconda Python with a simulated Rasterio Application Control block: bandwidth/overview transitions, persistent LTE capacity, site-checker proxy provenance, all four local hazard workspaces and result controls. No Windows security setting was changed.

These checks validate application behaviour, not real RF accuracy or external-feed availability. No screenshot-based browser visual review or live API certification is claimed for this release.
