# v14 — Full AOI Estimates

Extract the full project into a new folder, keeping your previous version. Stop the old Streamlit process with Ctrl+C, then use `run_windows.bat` or run `python -m streamlit run app.py` from the extracted project. A browser refresh alone does not reliably replace imported modules.

Open **Overview → Show bandwidth mask → Show full AOI estimate**. The new red–yellow–green map fills every selected AOI polygon using nearby collected tile averages as examples. Select **All project areas** for the four study areas and all 36 configured township boundaries, including Seikgyikanaungto, which lacks inventory tower records. Adjust the source quarter, minimum tests, 500/1000 m display grid or color maximum. Check **Color by distance to measurements** to inspect data support. Download the selected full-AOI estimates from the method expander.

The original measured map remains available and is still the default. Its speed values, source counts and blank unsampled areas have not been altered. Existing scenario allocation, manual measurement upload, ordinary maps, hazard/site AI models and university/team header are preserved. No new dependency, GEE credential or API key is required for this feature.

The default full-project layer contains 11,203 generated cells across 2,622.47 km². These are estimates, not extra measurements. Its spatial check produced 20.1 Mbps mean absolute error, worse than the 17.1 Mbps simple-mean baseline; 81.1% of its area requires extrapolation. The app displays that limitation prominently. This release adds full visual extent, not demonstrated prediction accuracy. See [method, evaluation and license](FULL_AOI_ESTIMATES.md).

For Streamlit hosting, deploy the complete extracted project including the new `engine/aoi_bandwidth.py`, `ui/aoi_bandwidth_map.py` and unchanged `data/mobile_performance/` dataset, then reboot the app. Never include your private service-account JSON or real secrets in a public repository.

Verification: 449 automated tests passed, along with the full-AOI interaction check, existing bandwidth/scenario/upload interaction check, changed-code lint and a browser inspection of the complete map. Existing NumPy timedelta deprecation warnings remain in unrelated hazard code. These checks establish software behavior, not prediction accuracy.
