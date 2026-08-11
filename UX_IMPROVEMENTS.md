# UX improvements in this package

This package keeps the same Yangon data, trained XGBoost model, calculations and project scope, but makes the dashboard easier for non-technical users to understand.

## Main changes

- Renamed the dashboard to **Yangon Telecom Coverage & Resilience Planner**.
- Replaced technical map labels such as "Observed tower sites" and "Recommended new sites" with **Existing telecom towers** and **Suggested tower locations**.
- Replaced `Admin-3`/`Admin-4` wording in the main user view with **Township** and **Ward / Village Tract**.
- Replaced raw 0-1 planning scores in the normal recommendation table with plain-language bands such as **Low / Moderate / High**.
- Changed **OPTIMAL CANDIDATE** style wording to **Recommended for field review**, so the dashboard does not imply final engineering approval.
- Added a plain-language **Why this location** explanation for every selected suggested site.
- Reduced the normal recommendation table to the fields a planner needs first: rank, place, nearest-tower distance, population need, disaster risk, overall score and recommendation.
- Moved raw model fields, coordinates, probabilities, feature values and model explanations into expandable **Technical details** sections.
- Rewrote the Site Checker so it first gives a practical recommendation and explanation, while keeping XGBoost inputs and model contributions available under Technical model details.
- Simplified disaster-simulation wording around affected population, backup sites and potential loss of access.
- Kept the detailed CSV export for engineering/analysis work.

## Important modeling note

The model and source datasets were not retrained or replaced in this UX revision. The changes are primarily presentation, terminology and explainability improvements. Suggested locations remain planning-screening candidates and require RF, land/access, structural, power, backhaul and regulatory checks before construction.

- Added a page footer with the university logo, **University of Technology (Yatanarpon Cyber City)**, and **Team GeoVisionaries**.
