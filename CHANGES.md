# Release History

## 2.4.13 (2025-mm-dd)

- LINT: Add linting for `.pyt` files
- CI: Add pre-commit bot
- 
## 2.4.12 (2025-07-31)
- CI: Use new groupware CI template for triggering sertit_atools update
- Change arcgispro toolbox from category RRM to CEMS RRM

## 2.4.10 (2025-05-13)
- Change print errors to loggers

## 2.4.9 (2025-05-06)
- Add exception control for error in reading rasters with rasterio and vectors with pyogrio at the FTEP, assumed to be related with problems of timeouts, failed reads, networking issues, rate limits, etc.
- DOC: Modify in CHANGES.md organization.

## 2.4.8 (2025-04-17)
- Change the approach for computing the ls_factor in the FTEP. Compute fill_depressions with pysheds to avoid panicking error, then go back to the regular whitebox_workflows computations.

## 2.4.7 (2025-04-16)
- Keep pysheds as the only method for computing the ls_factor in the FTEP.
- Mask the c_factor to the image footprint to avoid random pixels.

## 2.4.6 (2025-04-16)
- Delete: Add a try + time.sleep() on wbw.fill_depressions to avoid panicking errors when ran in FTEP
- Return of pysheds method for computing the ls_factor as a backup for FTEP processing.

## 2.4.5 (2025-04-15)
- Add a try + time.sleep() on wbw.fill_depressions to avoid panicking errors when ran in FTEP

## 2.4.4 (2025-04-09)
- Use full_like from xarray instead of odc.geo
- Delete the empty field FER_RE_av from the final statitstics FER_ER_ave.shp
- In the README. Add remarks for the european method due to absent data from countries in the european method such as Andorra, Switzerland, Croatia and others.

## 0.1.0 (2024-MM-DD)

- :rocket: First release