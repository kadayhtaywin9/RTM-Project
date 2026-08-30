# GeoVision AI architecture

```text
                            GEOVISION AI
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
 MODEL 1: Tower Recommendation AI       MODEL 2: Disaster Impact AI
             |                                     |
     new tower locations              +------------+------------+
                                      |            |            |
                                      v            v            v
                                  Flood AI    Earthquake AI  Cyclone AI
                                      |            |            |
                                  GEE GSMaP       USGS       GEE IBTrACS
                                  GEE SRTM       events       + local context
                                      |            |            |
                                      +------------+------------+
                                                   |
                                                   v
                                             Compound AI
                                                   |
                                                   v
                                       tower exposure / impact
                                                   |
                                                   v
                                      population rerouting model
                                                   |
                                                   v
                                               Dashboard
```

The previous manual disaster-severity simulator is no longer the primary disaster engine. The user selects one of the four AI hazard modes, runs Model 2, and the selected AI score drives the telecom coverage-impact analysis.
