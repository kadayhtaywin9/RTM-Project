# GeoVision v19 — focused engineering review

Reviewed 14 September 2026 against the supplied v18 archive. This is a source review plus local regression/UI testing, not a penetration test, field RF survey or validation of a live deployment.

## Verdict

GeoVision is a useful research and planning demonstrator with a stronger interface than its underlying evidence can currently justify. The distinction between measured observations, hypothetical capacity allocation and estimated full-area surfaces must remain visible. The SOS feature is a demo workflow, not a dependable emergency service. Improving its appearance does not resolve its delivery, privacy or operational risks.

## Regressions corrected in this release

- **Overview bypassed the evidence tools.** v18 routed Overview through `ui/network_bandwidth_map.py`, a distance-only speed illustration, instead of the existing evidence workspace. v19 restores the separate measurement/scenario/estimate entry point and the older map layers. The legacy module remains in the package for compatibility but is not called by Overview.
- **Selected area and displayed towers disagreed.** v18 passed the complete site inventory into Overview. v19 uses the selected tower set for the ordinary map. A regression test checks Hmawbi filtering.
- **Tower points lost useful detail.** Large opaque points and a generic hover obscured dense areas. The compact circles, useful metadata and explicit display-sampling caption are restored.
- **The clock mixed timezones.** The resident's clock used device-local time alongside a UTC date. v19 formats both consistently in Myanmar time and renders immediately, without needing GPS or map initialization.
- **SOS polling was attached to the wrong function.** The timer decorated the sound helper rather than the incident panel. The corrected panel now owns polling and its fragment-scoped actions.

## Highest-priority remaining risks

### 1. Protect precise SOS locations before public hosting

In `sos_service/app.py`, `require_dashboard_key` rejects unauthorized reads/updates only when `SOS_API_KEY` is nonempty. With the default empty key, anyone able to reach the service can read incident locations or change incident status. Set a nonempty server-side key before exposure. Next, add authenticated responder accounts, least-privilege roles, an audit trail, retention/deletion policies and a tested deployment configuration that refuses unsafe public defaults.

Resident submission remains public. A dashboard key does not stop spam, repeated submissions or a resource-exhaustion attack. Introduce server-side rate limits, bounded request processing, idempotency keys and abuse monitoring. Origin restrictions alone do not authenticate callers. The service also stores user-agent strings; disclose and justify any retained metadata alongside coordinates.

### 2. Make SOS delivery and incident lifecycle reliable

The current success message means the API stored an incident. It does not mean a responder saw it, accepted it or dispatched help. Five-second polling depends on an active Streamlit session. SQLite, a single process and a browser toast are not a durable dispatch pipeline.

The dashboard requests only the latest 50 incidents. Older unresolved incidents can disappear from that window; its counts are not a complete active-incident backlog. Add status-filtered pagination, persistent queues, retries, delivery/acknowledgement tracking, escalation and responder operating procedures.

In `sos_service/static/index.html`, GPS callbacks can re-enable the send button during an in-flight request/cooldown. A cached location can also outlive an error. Add an explicit sending state, freshness/accuracy checks, request timeouts and server-side duplicate protection. Test poor connectivity, permission loss, page reloads and repeated taps. The bundled sound payload contains no audio samples, and browser playback restrictions also apply; do not promise a reliable audible alarm.

Do not advertise this version as an alternative to established emergency channels. No telecom/SMS/satellite/offline delivery or emergency-agency integration was verified. The visible clock is informational, sourced from the device; incident chronology should use server receipt and verified operational timestamps.

### 3. Do not call geometric proximity verified telecom coverage

The tower inventory is historical, and a mapped site is not necessarily an active physical tower, radio sector or available serving cell. A radius or nearest-site assignment does not capture antenna azimuth/tilt, frequency, bandwidth, interference, terrain/buildings, user device, subscriber network, power or backhaul constraints.

Technology labels alone do not establish throughput. In particular, the 100 Mbps LTE budget is an editable scenario input, not the speed each LTE user receives or a measurement of each site's capacity. Load sharing conserves assumed demand/capacity but does not validate those inputs. Real RF/capacity work needs verified sector inventory, spectrum configurations, busy-hour counters, outages/backhaul data and representative field measurements.

### 4. Treat the AI scores as proxy scores until validated against outcomes

The packaged tower model learns labels derived from earlier planning rules. Hazard models use proxy targets and synthetic event scenarios; repeated tower/scenario rows are not independent observed disasters. Good agreement with those labels does not establish tower-site feasibility, failure probability or prediction accuracy in a future event.

Next steps are verified deployment/outage labels, event/time/geography-separated evaluation, simple engineering baselines, uncertainty calibration and explicit out-of-distribution handling. Cyclone evaluation needs a substantially broader historical event set with independent held-out storms. This release does not retrain the models or test live GEE/JTWC/USGS availability.

### 5. Keep missing bandwidth evidence visible

The bundled Ookla data comprises sampled quarterly tiles, not all users or complete RF coverage. A sample mean is not a guaranteed individual speed. Filling an AOI changes the extent of a visualization, not the amount of measured evidence.

The existing full-AOI documentation reports a default spatial-holdout error of about 20.1 Mbps versus 17.1 Mbps for a simple mean baseline. That interpolation is not demonstrated to outperform the baseline. Retain estimate labels and source-distance/holdout diagnostics; do not silently convert it into a measured map. For defensible estimates, collect more spatially and temporally representative observations, validate out of sample and publish error/coverage limits.

## Recommended next upgrade

Prioritize SOS access control and send-state/idempotency fixes, followed by active-incident pagination and reliable acknowledgement. For telecom accuracy, build a verified sector/measurement dataset before changing the AI architecture. Keep measurements, engineering scenarios and inferred surfaces separate in both the UI and exports.

## What the checks establish

454 Python tests, a deterministic JavaScript clock test, a Streamlit UI smoke test and browser inspection passed. API tests used temporary local data and a test key. No real emergency was submitted. These checks establish the tested UI/API behavior; they do not certify field accuracy, security completeness, phone compatibility across all devices or production readiness.

Implementation references: [Overview and SOS dashboard](app.py), [resident service](sos_service/app.py), [resident page](sos_service/static/index.html), [clock](sos_service/static/sos-clock.js), [existing telecom review](TELECOM_EXPERT_REVIEW.md), [full-AOI limitations](FULL_AOI_ESTIMATES.md).

The clock uses explicit timezone formatting documented by [MDN](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/DateTimeFormat), served through FastAPI's [static-file support](https://fastapi.tiangolo.com/tutorial/static-files/).
