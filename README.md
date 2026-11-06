GB electricity demand

| Column | Contains |
|---|---|
| SETTLEMENT_DATE | The date |
| SETTLEMENT_PERIOD | Half-hour of the day, 1-48 (1 = 00:00-00:30) |
| ND | National demand: GB electricity used from the grid. The series searched for anomalies |
| TSD | ND plus the grid's own use, pumping and exports |
| ENGLAND_WALES_DEMAND | ND for England and Wales only |
| EMBEDDED_SOLAR_GENERATION | Estimated rooftop/small solar output |
| EMBEDDED_SOLAR_CAPACITY | Estimated installed small solar |
| EMBEDDED_WIND_GENERATION | Estimated small wind farm output |
| EMBEDDED_WIND_CAPACITY | Estimated installed small wind |
| PUMP_STORAGE_PUMPING | Power used to pump water uphill for hydro storage |
| NON_BM_STOR | Standby reserve generators |
| SCOTTISH_TRANSFER | Power flow between Scotland and England |
| IFA_FLOW, IFA2_FLOW, ELECLINK_FLOW | Cable flow to/from France |
| BRITNED_FLOW | Cable flow to/from Netherlands |
| NEMO_FLOW | Cable flow to/from Belgium |
| NSL_FLOW | Cable flow to/from Norway |
| VIKING_FLOW | Cable flow to/from Denmark |
| MOYLE_FLOW | Cable flow to/from Northern Ireland |
| EAST_WEST_FLOW, GREENLINK_FLOW | Cable flow to/from Republic of Ireland |

Each row is one half-hour. All values are in MW. Cable flows: positive = import into GB, negative = export.

Source: [NESO Historic Demand Data](https://www.neso.energy/data-portal/historic-demand-data)