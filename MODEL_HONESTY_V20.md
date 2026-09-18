# Model honesty and limitations — v20

## What the audit actually found

An offline check of all 8,029 packaged tower records sets only the earthquake/cyclone `event_intensity` feature to zero, leaving the same model and historical/site features in place:

| Model | Median score with zero event input | Towers at or above 50/100 with zero event input |
| --- | --- | --- |
| Earthquake | 44.11 / 100 | 1,208 |
| Cyclone | 43.49 / 100 | 392 |

The cyclone count of 392 matches the user's screenshot, but that numerical match alone does not establish which live JTWC inputs their run used. It demonstrates that the old scenario could produce that count without any cyclone event contribution. The screenshot's earthquake count may also include a recent-event signal; the screenshot alone cannot identify that signal or its date.

These are not observed failed towers. The models were trained to reproduce proxy scoring formulas, which deliberately include historical and site factors. In particular, the earthquake training formula gives substantial weight to historical exposure; older radio technology and mapped cell counts are also used as vulnerability/redundancy proxies. They are not measurements of mast strength, power resilience or functioning backup coverage.

The existing model is not suddenly “unbiased” because the colors became softer. v20 therefore preserves the trained model outputs, shows their reference contribution, and no longer automatically promotes score thresholds into prominent outage/population totals.

## Meaning of the new reference values

- **Model score:** the original prediction using the current selected inputs.
- **Zero-event reference:** the same model with only `event_intensity = 0`; all historical/site inputs stay fixed.
- **Event-input change:** original score minus reference, kept signed. It is a sensitivity diagnostic, not event damage, causation or a probability.

The reference is available for earthquake and cyclone only. Flood has a different feature contract; its source summary explains that 24-hour and 72-hour rain totals are not direct trained-model inputs. Compound results inherit all three component models' limitations.

The audit also compared event input zero versus one and found no decreasing endpoints across the packaged tower set. That is a limited endpoint sanity check, not proof of monotonic behavior everywhere, unbiased decisions or forecast accuracy. Reproduce it with:

```powershell
& ".\.venv\Scripts\python.exe" scripts/hazard_honesty_audit.py
```

Results are stored in `HAZARD_MODEL_AUDIT_V20.json`. No external APIs are called by that audit.

## What v20 improves without manufacturing accuracy

- Continuous, fixed-scale map coloring keeps adjacent score values visually close; numeric values and 30/50/70 category thresholds are unchanged.
- Hypothetical tower-outage calculations require an explicit user choice. When off, affected-population fields remain unknown rather than being reported as zero.
- Source counts, time windows, individual earthquake records and cache clocks are visible and exported. A forecast track point, a rain image and a tower sample are not treated as interchangeable evidence.
- Delayed GSMaP products are labelled; the 73-hour tolerance is not described as real-time rainfall coverage.
- A no-event earthquake result no longer gets the analysis clock substituted for a missing observation timestamp.
- The main result states that validation against real outcomes and calibrated uncertainty are unavailable. It does not invent confidence percentages.

## What remains biased or uncertain

1. **Proxy targets:** high cross-validation scores mainly measure reproduction of authored formulas. There are no verified independent tower-outage labels establishing real-world predictive performance.
2. **Geographic and inventory coverage:** historical tower records, mapped cell counts and the 2020 population baseline can be incomplete/outdated. Missing or substituted features can distort rankings; defaults in legacy preprocessing still need a dedicated missing-data audit.
3. **Physical simplification:** RF propagation, antenna sectors, structural condition, grid power, backhaul and repair status are not fully modeled. The nearest tower is not necessarily the serving or operational tower.
4. **Time mismatch:** a 30-day earthquake maximum is not a current warning; a delayed rainfall image misses the period since acquisition. Cyclone history is limited and is not full IBTrACS-based outcome training.
5. **Uncertainty and calibration:** output scores are not probabilities, and verified prediction intervals/error rates are not available. The new reference values do not fill that gap.
6. **Operational safety:** the SOS prototype still needs fail-closed public access controls, abuse limits, duplicate protection, active-incident pagination, reliable acknowledgement and a real responder workflow. The requested receipt display does not establish delivery to responders.

## What is needed for a defensible model upgrade

Collect verified, time-matched event/outage/non-outage records and sector/asset condition data. Hold out entire disasters, future periods and geographic areas; compare against transparent engineering baselines. Report errors separately by township, hazard severity, site type and data quality. Calibrate only against genuine outcomes, and explicitly abstain or flag unsupported inputs. These steps can measure and reduce bias; no software edit alone can promise its absence.

This release does not lower scores to satisfy a desired map appearance, silently relabel historical information as current evidence, or claim improved physical prediction accuracy.
