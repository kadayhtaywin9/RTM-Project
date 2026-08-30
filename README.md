# GeoVision AI — Multi-Hazard Telecom Resilience Dashboard

This version contains two AI systems:

1. **Tower Recommendation AI (Model 1)** — recommends candidate locations for new telecom towers.
2. **Disaster Impact AI (Model 2)** — estimates disaster exposure/impact for existing towers using four selectable AI modes:
   - Flood / Heavy Rain
   - Earthquake
   - Cyclone
   - Compound

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

For local Earth Engine use, authenticate first and set your project:

```bash
earthengine authenticate
set GEE_PROJECT_ID=YOUR_PROJECT_ID
streamlit run app.py
```

See `GEE_SETUP.md` for Streamlit Cloud/service-account deployment.

## Important interpretation

The disaster models are hackathon/MVP planning models. Current flood, earthquake, cyclone and compound outputs are exposure/impact scores with pseudo-label limitations, not calibrated disaster or tower-failure probabilities. See `models/MULTI_HAZARD_MODEL_CARD.md`.
