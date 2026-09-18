# Bandwidth load sharing: method, limits and audit

## What v12 changes

The old bandwidth calculation did not connect the entire city to one tower: it used many site proxies, but assigned each population pixel wholly to its nearest site. v12 adds fractional cohort allocation across nearby compatible sites. This addresses rigid attachment, not the missing evidence needed to predict real speeds.

Open **Overview → Show bandwidth mask → Planning scenario**. **Load-aware sharing (scenario)** is the default; **Nearest site (baseline)** restores the prior assignment. Radio technology, editable capacity budgets, display size, active percentage, target, log/linear colors and the separate measured-download view are retained. No operator selector returns. No new key or API is needed for this feature.

The comparison cards use the same inputs and displayed eligible population. The model always allocates demand from the whole project before filtering the display area. A smaller viewport cannot create extra capacity by removing neighboring users.

## Compatibility and the site-ID crosswalk

`tower_sites_population.csv` has 8,029 population-model IDs (`tower_id`). Cell observations in `yangon_tower_cells_with_site.csv` use a different ID system (`yangon_tower_id`). Direct equality links the correct physical grouping for only 2,631 of the 8,029 population IDs.

The new compatibility module links population sites to `tower_sites_yangon_all.csv` using the original unique `(adm3_name, lat_r, lon_r)` grouping keys, then links that lookup's `yangon_tower_id` to cell observations. All 8,029 bundled population sites match. Duplicate keys/IDs or absent required columns fail closed. Unmatched sites do not acquire invented network identities.

For each site and assigned radio, collect the jointly recorded network names. Offloading requires:

- The same assigned generation at anchor and destination. Automatic mode does not route a UMTS cohort onto a mixed site's LTE budget merely because it also records UMTS.
- Exactly one matching recorded network at both sites for that radio. Multi-network or unknown anchor compatibility locks that cohort to its anchor.
- A destination within the user's assumed geographic radius. The anchor must itself have radius and technology support; unknown/outside-anchor locations remain unassessed.

The graph queries up to five nearest **compatible** sites, including the anchor. It does not select five arbitrary neighbors and then discard incompatible ones. Tied/co-located positions retain the anchor without duplicating a site ID. The cap is a computational convention, not evidence that other sites cannot serve a location.

This is conservative inventory compatibility, not measured subscriber attachment. The nearest anchor's network and assigned generation are cohort assumptions; real SIM networks, subscriptions, roaming, devices and serving cells are absent. Network labels may be historical, incomplete or stale. The uniform geographic radius is not a signal-strength or RF-feasibility test.

## Capacity-conserving calculation

For population pixel `i`, let `p_i` be baseline population and `n_i = p_i × active_pct / 100` be its assumed simultaneous-user cohort. Let `x_ij` be the fraction assigned to compatible site `j`.

```text
For each supported positive-population pixel: sum_j x_ij = 1, x_ij >= 0
For unsupported pixels: all x_ij = 0; throughput remains unknown
Site active users L_j = sum_i n_i * x_ij
Site assumed Mbps/user r_j = C_j / max(L_j, 1)
Pixel mean Mbps/user = sum_j x_ij * r_j
Allocated site throughput = L_j * r_j <= C_j
```

Fractions divide aggregate population among sites. They do not split a phone into multiple users or bond the speeds of several towers. There is one assumed budget per site ID: no automatic multiplication by `cell_count`, antenna sectors, radio labels or carriers. Sites with less than one equivalent active user retain the denominator floor and cannot exceed their budget per user.

The objective is an author-implemented convex congestion proxy:

```text
F(L) = sum_j [L_j + 0.5 * max(L_j - 1, 0)^2] / C_j
gradient_j = max(L_j, 1) / C_j
```

Starting at the nearest-only allocation, each iteration compares a Frank–Wolfe assignment toward lowest-cost candidates with a pairwise move from the most expensive used candidate. Bisection line search selects a feasible objective-decreasing step. The solver keeps fractions nonnegative, counts each supported cohort once and checks every site's budget before allowing results to display.

The default is at most 120 iterations with a relative Frank–Wolfe objective-gap tolerance of 0.001. If the limit is reached, the feasible approximation is shown with an explicit warning and its residual gap. This gap is a numerical optimality diagnostic for this objective, not a confidence interval or percentage error on real Mbps. An optimum of this proxy would still not validate the assumptions. It does not maximize every user's throughput or guarantee a lower share below any chosen target.

For context, [Ericsson's RAN load-balancing discussion](https://www.ericsson.com/en/blog/2022/10/how-csps-can-achieve-balance-in-ran-environments-a-demo) describes real balancing using network and performance information. GeoVision does not implement or validate an operational RAN controller. The algorithm here is a planning allocation, not evidence of an operator's live configuration.

## Honest map summaries

The map colors each supported display cell's population-weighted mean. Hover reports the smallest/largest assigned-group rate and the fraction of eligible population below the target. With split cohorts, shortfall is calculated as `sum_j x_ij * (r_j < target)` before aggregating people. Half the users at 5 Mbps and half at 15 Mbps means a 10 Mbps average **and 50% below 10 Mbps**, not zero shortfall.

Green therefore refers to a mean, not a promise about everyone in that cell. Gray continues to mean incomplete support or unassessed throughput, never measured zero. A color-scale change affects only encoding, not the allocation, raw speeds or shortfall percentages. No percentile-based recoloring or increased capacity preset is used to make this release appear better.

The sensitivity table changes capacity and activity while holding the selected routing fixed; it explicitly does not re-optimize those three cases. Changing the actual capacity/activity controls computes a new allocation. Target and color controls reuse the same allocation.

## Reproducible bundled-data audit — 10 September 2026

Run `python scripts/load_sharing_audit.py` from the extracted project directory. The following is the full-project automatic-technology scenario, 5 km radius, 5% active population and unchanged GSM 0.2 / UMTS 10 / LTE 100 Mbps site budgets. These are **not observations of Yangon network speeds**.

| Diagnostic | Nearest baseline | Load-aware approximation |
| --- | ---: | ---: |
| Eligible population-weighted mean, Mbps/user | 0.28294 | 0.32515 |
| Eligible population below 1 Mbps | 95.0205% | 95.0237% |
| Eligible population below 10 Mbps | 99.8419% | 99.8792% |
| Congestion objective | 29,714,845.77 | 13,301,888.81 |

The calculation includes 312,302 population pixels and 8,029 site proxies. Approximately 7,004,742 baseline people have scenario support; 2.9533% of baseline population remains unassessed. Of supported assumed demand, 88.4513% has multiple compatible candidates, 9.7265% has unknown/ambiguous compatibility and remains anchored, and 69.1018% moves from its original anchor. A site with known compatibility can still have no additional in-radius candidate.

The run reached 120 iterations with a relative objective gap of 0.02341; it did **not** meet the 0.001 tolerance. Runtime was approximately 15 seconds in the local audit; hosting performance is not guaranteed. The script verifies conservation and candidate compatibility independently of chart colors.

The mean improves, yet population shortfall at these particular thresholds does not. That is possible: redistributing scarce capacity changes who shares it and minimizes a congestion proxy, not the count exceeding 1 or 10 Mbps. We do not hide this by showing only a favorable statistic.

### Why this cannot by itself remove low predictions

The unchanged assumed budgets total **115,078 Mbps**, shared by approximately **350,237 assumed active users**. Even granting use of every budget, the optimistic aggregate mean ceiling is only `115078 / 350237 ≈ 0.329 Mbps/user` under these inputs. Geography, compatibility and the one-user floor can only reduce utilization of that budget. This is an arithmetic limit of the scenario, not a limit on the real city.

More accurate real Mbps requires current sector/carrier inventory, spectrum widths, MIMO and radio configuration, independent backhaul budgets, actual busy-hour users/traffic by operator, device/attachment evidence, signal/interference information and time/location/technology-stamped measurements. Population is not an observed count of simultaneous mobile users. A historical site inventory and assumed per-site rates cannot substitute for that evidence.

No AI model was retrained, no RF model was added, and no new measured speeds were invented in v12. Existing hazard and siting scores retain their prior limitations.
