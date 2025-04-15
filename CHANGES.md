# Release History

## 0.1.0 (2024-MM-DD)

- :rocket: First release

## 2.4.4 (2025-04-09)
- Use full_like from xarray instead of odc.geo
- Delete the empty field FER_RE_av from the final statitstics FER_ER_ave.shp
- In the README. Add remarks for the european method due to absent data from countries in the european method such as Andorra, Switzerland, Croatia and others.

## 2.4.5 (2025-04-15)
- Add a try + time.sleep() on wbw.fill_depressions to avoid panicking errors when ran in FTEP