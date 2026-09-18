# Technology-based bandwidth: what the numbers mean

## Presets are assumptions, not verified tower specifications

The app now uses recorded radio technology instead of a bandwidth operator selector. The inventory records site proxies and radio labels; it does not establish independently provisioned physical cells, sector counts, channel widths, antenna configurations, active users or backhaul capacity. A site's technology name cannot determine its actual Mbps.

These **author-chosen starting budgets** are intentionally editable:

| Recorded type | Initial shared budget per mapped site proxy | Extra assumption / limitation |
| --- | ---: | --- |
| GSM / 2G | 0.2 Mbps | Assumes EDGE-enabled packet data; a GSM label does not confirm EDGE or even packet-data availability. |
| UMTS / 3G | 10 Mbps | Assumes HSPA-enabled service; a plain UMTS label does not confirm HSPA or its configuration. |
| LTE / 4G | 100 Mbps | Illustrative shared budget, not a verified LTE sector/cell rate. Spectrum, MIMO, bands and backhaul are unknown. |
| NR / 5G | 500 Mbps | Only offered when NR/5G is actually recorded. Band, bandwidth, MIMO and backhaul are unknown. No NR site is bundled. |

These values are not theoretical maxima, standard-mandated rates, validated statistical estimates or an assertion that every site of a generation has the same real capacity. Edit them under **Technology capacity assumptions (editable)** if you have better configuration or measured-capacity evidence. The bundled data cannot supply calibrated upper/lower bounds. The half/double sensitivity table is not a confidence interval.

The preset values are modelling choices, not figures prescribed by the references below. The sources explain why technology/configuration distinctions matter:

- [ETSI / 3GPP TR 45.912](https://www.etsi.org/deliver/etsi_tr/145900_145999/145912/16.00.00_60/tr_145912v160000p.pdf) distinguishes GSM/EDGE capabilities across radio-resource configurations. It does not provide a universal capacity for a site labelled GSM.
- [3GPP-hosted UMTS/HSPA white paper](https://www.3gpp.org/ftp/pcg/pcg_15/docs/pdf/PCG15_22.pdf) describes HSPA as an enhancement to earlier UMTS. Consequently a UMTS inventory label alone does not establish an HSPA data rate.
- [Ericsson: spectrum and transport capacity](https://www.ericsson.com/en/blog/2020/5/sizing-up-spectrum-using-transport-the-best-way-with-new-5g-assets) explains substantial capacity differences with spectrum assets, MIMO and backhaul. A newer-generation label is not an exact capacity measurement.

## Assignment and sharing

**Automatic** retains each population pixel's nearest mapped site as its anchor. That site receives the preset for its highest recorded generation: NR, then LTE, then UMTS, then GSM. This ordering is a scenario convention, not an observed connection or a guarantee that the newest radio is operational or gives the best real service. A site marked `GSM, UMTS, LTE` receives one LTE budget, not the sum of three budgets. Its `cell_count` does not multiply the budget.

**A technology filter** selects sites explicitly recording that type and recomputes nearest-site assignment for all project population. For example, a mixed GSM/UMTS/LTE site uses the UMTS preset in the UMTS view. Selecting LTE does not upgrade other sites or presume they support LTE. Demand is not reduced merely because the displayed study area is smaller.

An unrecognized nearest-site radio in automatic mode has **unknown capacity**, not a default or zero-capacity estimate. Unsupported pixels and partially supported display cells remain gray. Radius-based assignment is still geographic, not RF propagation. All types currently share the user's chosen geographic-radius assumption; there is no invented per-technology RF range.

In v12, **Load-aware sharing (scenario)** divides population cohorts across up to five radius-eligible compatible sites. Both sites must have the same assigned radio and exactly one matching recorded network for that radio; unknown/multi-network matches remain at the anchor. No subscriber market shares, roaming, technology upgrades or actual handovers are inferred. **Nearest site (baseline)** retains the earlier allocation for comparison. See [the full allocation method and audit](BANDWIDTH_LOAD_SHARING.md).

Within the assumed radius:

```text
Equivalent active users = full-project allocated population (2020) × active percentage
Scenario Mbps per active user = selected site's assumed budget / max(active users, 1)
```

The minimum-one-user denominator prevents an isolated fractional-population pixel from exceeding a site's assumed budget. Capacity is conserved separately for every site ID. No demand from outside the project is available. Adoption and simultaneous activity are combined into one editable active-population assumption.

The operator selector remains removed, but v12 uses joint cell-level radio/network observations internally to prevent cross-network offloading. Cohorts are hypothetically associated with their nearest anchor's network, not verified subscriber identities. This remains a regional technology planning scenario, not the coverage or throughput a particular subscriber can access. Physical-site duplication, device compatibility, interference, scheduling, power outages and shared backhaul remain unresolved. The half/double sensitivity table holds routing fixed and does not re-optimize sharing; changing the actual input controls does re-run the allocation.

**The per-user target is separate.** The initial 10 Mbps target controls comparisons only. Changing it to 25 Mbps does not increase capacity; it makes the success threshold stricter. Gray is unassessed/partial support, and uncolored areas have no displayed population data. Display-cell size controls visualization detail, not measurement accuracy.

In v11.1, **Speed differences (log)** is the default planning color scale. It spreads small cell-mean speeds across red, amber and yellow while reserving a separate green endpoint for means reaching the target. Yellow is still below target, not proof of adequate service. Actual Mbps ticks are shown with nonlinear spacing; the transform is fixed relative to the target rather than automatically re-ranked for the current visible data. **Target comparison (linear)** retains the original red/zero, yellow/half-target and green/target mapping. Switching the scale does not change capacity, active users, catchments, mean Mbps, unknown locations or population shortfall. A green display-cell mean can coexist with slower individual population pixels; the hover gives the range and below-target share. See [v11.1 notes](RELEASE_NOTES_V11_1.md) for the diagnostic and exact mapping.

## Bundled inventory audit (10 September 2026)

There are 8,029 mapped site proxies. Counts recording each type overlap: GSM 1,939; UMTS 6,461; LTE 527; NR 0. In automatic mode, the one-budget-per-site assignments are GSM 1,290; UMTS 6,212; LTE 527. These are historical inventory counts, not a current network census. In particular, missing 5G records do not prove that no 5G exists today.

## Measured-download view

The CSV now requires:

```text
latitude,longitude,download_mbps,measured_at,radio,location_accuracy_m
```

Use one observed technology per test: GSM/2G, UMTS/3G, LTE/4G or NR/5G (recognized aliases are normalized). Mixed, missing and unrecognized labels are rejected instead of guessed. Existing operator-only files must be supplemented with the actual test's radio technology, or cannot be used for this filter. Merely renaming an operator column to radio is not valid evidence.

The measured map uses supplied speeds, never technology presets. It filters by selected radio, observation window, area and location accuracy. It does not interpolate or fill missing data. Identical normalized rows are deduplicated. Recent, sufficiently precise readings of the same technology are pooled across networks and devices; medians therefore do not establish any single network's performance. The existing 5 MB/50,000-row limits, timezone requirement, nonnegative finite speeds and 500 m bins remain. See the app's empty template; no synthetic speed observations are shipped as real project data.
