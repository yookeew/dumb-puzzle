# data/derived/

`stands_empirical.csv` -- median parked position per (airport, STAND_mvt), from the
>= 3 min stationary segment that a matched departure's ADS-B surface run starts in
(PROGRESS.md §51 Part B; `tests/step51_partB_stands_and_parked.py`). Exploratory; not
used by the detector or any submission.

Licence: this table is a derived database of adsb.lol data and is made available under
the Open Database License (ODbL) 1.0 -- https://opendatacommons.org/licenses/odbl/1-0/ --
with attribution "adsb.lol contributors". Any public use or redistribution of it (or of a
database derived from it) must keep this licence and attribution (share-alike). The
`nearest_gateway` column holds stand names from the X-Plane Scenery Gateway (GPLv2, see
`data/external/stands_LICENSE_GPLv2.txt`).
