# Release History

## 0.1.0 (2024-MM-DD)

- :rocket: First release

## 2.4.4 (2025-04-09)
- Use full_like from xarray instead of odc.geo
- Delete the empty field FER_RE_av from the final statitstics FER_ER_ave.shp
- In the README. Add remarks for the european method due to absent data from countries in the european method such as Andorra, Switzerland, Croatia and others.

## 2.4.5 (2025-04-15)
- Add a try + time.sleep() on wbw.fill_depressions to avoid panicking errors when ran in FTEP

## 2.4.6 (2025-04-16)
- Delete: Add a try + time.sleep() on wbw.fill_depressions to avoid panicking errors when ran in FTEP
- Return of pysheds method for computing the ls_factor as a backup for FTEP processing.

## 2.4.7 (2025-04-16)
- Keep pysheds as the only method for computing the ls_factor in the FTEP.