# TDEW Gap-Fill — Scientific Plan & Progress Tracker

> **Purpose of this file.** Single source of truth for the effort to turn the per-zone×doy
> dew-point (Td) tuning pipeline into a **defensible scientific result for a paper**.
> A fresh Claude session should read this top-to-bottom, then continue from the first
> unchecked task in §7. Update the checkboxes and the "Progress log" (§12) as work lands.
>
> **Last updated:** 2026-07-16 · **Owner:** Piero Palacios (CIP) · **Assistant:** Claude Code

---

## 1. Scientific goal (what we are actually doing)

**Gap-fill the PISCO gridded daily dew-point temperature (Td) product forward in time.**

- PISCO **v1.1** provides both **Tmin and Td** for **1981–2016**.
- PISCO **v1.2** provides **Tmin for 1981–2020** but **no Td after 2016**.
- We learn the **Tmin → Td** mapping on the 1981–2016 overlap (Td from v1.1, Tmin from **v1.2**),
  then **predict Td for 2017–2020** at every grid cell using the continued v1.2 Tmin.

The model is per-(ID, doy) **weighted local linear regression on climate anomalies** (tricube DOY-window
kernel, half-window `h`), with per-**SENAMHI-zone × doy** feature/`h` selection (MultiLLR-style,
arXiv:1809.07394; LOYOCV cosine-skill objective). This is **NOT** a "predict at new locations" method —
it predicts **new years at the same grid cells**. That distinction drives the validation design (§5).

**Paper framing.** "A method to extend the PISCO gridded dew-point record (2017–2020) from the
continued minimum-temperature record, with per-climate-zone seasonal model selection." A perfectly
publishable outcome includes a *negative/neutral* result (per-zone×doy tuning does not beat a strong
climatology/Td≈Tmin baseline) **iff** the baselines and the temporal validation are rigorous.

---

## 2. Data inventory (verified 2026-07-16)

| item | path | coverage | notes |
|---|---|---|---|
| **Full national grid root** | `/media/ppalacios/Data1/henry_simcast_peru_full` (`$FULL`) | 2,818,404 IDs | dense 0..N-1 |
| Target Td (v1.1) | `$FULL/td/Outputs/td_daily_YYYY_MM.parquet` | 1981–2016 | the label |
| Grid coords | `$FULL/td/Outputs/grid_index.parquet` | — | ID→lon/lat |
| **v1.2 Tmin (gap-fill input)** | `/media/ppalacios/Data1/henry_simcast_peru/tmin_v12/Outputs` | **1981–2020** | 480 files incl. all of 2017–2020 |
| v1.1 Tmin | `/media/ppalacios/Data1/henry_simcast_peru/tmin_v11/Outputs` | 1981–2016 | 432 files; comparison only |
| Tuning results root | `$FULL/results_tuning` (`$RES`) | — | see §3 |
| Zone shapefile (zip) | `/home/ppalacios/Downloads/clasif_clima_peru.zip` | 41 zones | SENAMHI Thornthwaite; 15,700 polygon parts |

**Version-consistency decision (locked):** train and predict both use **v1.2 Tmin** (`--tmin-var tmin_v12`).
`reports/results/compare_v11_v12.md` shows the Tmin sensitivity differs by version (coeff mean 0.45 vs 0.37)
and switching versions injects ~0.3 °C bias, so a single Tmin source on both sides is required. State this
in the paper.

---

## 3. Method & pipeline status — what is already DONE

Full-grid pipeline has been run end-to-end on 1981–2016 (v1.2 Tmin → v1.1 Td). Artifacts under `$RES`:

- [x] **Prep** — `daily_climatology.parquet` (1.03 B rows), `bucketed_training_data/` (8192 buckets × 36 yr),
      `climatology_by_bucket/` (8192). OOM fixes committed (`86936a3` on `feat/hpc-code-tunning`).
- [x] **Zone table** — `$RES/zone_table.parquet` (2,818,404 rows, 41 zones, `unassigned=0`).
      (`zones.py` `sjoin_nearest` reprojects to EPSG:32718 — CRS-warning fix; verify it's committed.)
- [x] **Select** — `$RES/tuning/manifest.parquet` = **15,006 recipes** (41 zones × 366 doy):
      `[zone_id, zone_label, doy, h, feature_list, n_features, skill, skill_baseline, skill_uplift]`.
      Overview plot: `$RES/tuning/manifest_overview.png`.
      ⚠️ Used the **old h-grid `{7,11,15,21}`**. We are now extending it (§9) → this manifest and the coeffs
      below are **stale for the final config** and must be re-selected with the extended grid before the
      production fill (Phase B0, §7).
- [x] **Train** — `$RES/tuning/zoned_coeffs/id_bucket=XXXX/coeffs.parquet` (8192 buckets, **8.14 B** rows,
      74 GB). Schema `[ID, zone_id, doy, feature_name, coeff, r_squared_anom, h]`, no NaNs.
      (Same staleness caveat — trained from the old-h-grid manifest.)

**Manifest findings (need SME validation):** `h=21` chosen 56 % (window saturates at the max grid value →
consider extending the h-grid); selection prunes little (`n_features=11` kept 71 % → the fixed-5 baseline is
*under*-featured); **mean skill uplift only +0.013 (median +0.007), and 22.5 % of zone-days are equal or
worse** than the fixed-5/h=11 baseline. Existing fixed-model accuracy vs observed Td (from
`reports/results/accuracy_v11_v12.md`): **RMSE ≈ 1.13 °C**, strong **December warm bias +1.45 °C**.

> ⚠️ These skill numbers are **selection-optimistic** (LOYOCV was the objective the recipes were chosen to
> maximize) and **in-sample in time** (all 36 years used). They are NOT the paper's headline. §5 fixes this.

---

## 4. What makes it paper-grade (gaps to close)

1. **Ground truth — RESOLVED.** Target = the **gridded v1.1 Td** (that is the product being extended;
   validating against it is correct, not a compromise). Caveat to state: gridded Td and gridded Tmin both use
   **LST + DEM** covariates (regression-kriging; Td does *not* use Tmin directly — PISCOeo_pm paper,
   DOI 10.1038/s41597-022-01373-8), so they share some structure. Station-level validation is out of scope.
2. **Incumbent baselines — TODO.** Must beat **climatology** (the gap-fill incumbent), **Td = Tmin**
   (FAO-56 / Allen 1998), and a **zone OLS `Td~Tmin`**, not just the internal fixed-5/h=11 ablation.
3. **Validation is TEMPORAL, not spatial — TODO.** Forward backtest (train early years → predict held-out
   recent years), run **autoregressively** (see §5). Spatial blocking is a *secondary* robustness check only.
4. **Effect sizes over p-values — TODO.** ΔRMSE °C + block-bootstrap CIs, skill-vs-climatology, per-location
   distribution, Cohen's d as companion, and a **permutation/null test** (shuffle years, re-select) to show the
   +0.013 uplift is/ isn't distinguishable from selection noise.

---

## 5. Validation design (the paper's core evidence)

**Primary: forward temporal backtest, autoregressive.**

- **Holdout to mimic the real 4-year gap (CONFIRMED):** train on **1981–2012**, predict **2013–2016**
  (leave-last-4), score vs observed v1.1 Td. Also run leave-last-1 and leave-last-2 to show error growth
  with lead time.
- **Autoregressive requirement (critical).** The model uses `TD_anom_lag1/lag2` as predictors. In 2017–2020
  there is **no observed Td**, so those lags must come from the model's **own previous-day prediction**
  (sequential recursion — this is the P6 `forecast.py` gap, §6). The backtest **must also run autoregressively**
  (no observed-Td lags inside the held-out block) or it flatters the model vs what production can do.
- **Selection honesty.** For the winner's-curse fix in the *time* dimension, selection must not see the
  held-out years. Two variants to run and compare:
  - (a) **Refit-only** (fast): keep the existing manifest, refit coefficients on train-years, predict holdout.
  - (b) **Full re-select** (slower, honest): re-run `--stage select` on train-years only, then train+predict.
  Report both; the gap between them *is* the selection-bias estimate.
- **Cost control.** Do the backtest on a **stratified ID sample** (reuse cached `zone_table.parquet`;
  consider `--per-zone-n 500`), not the full 2.8 M grid. Full grid is only for the deliverable (§6).

**Baselines (all scored on the identical holdout):** climatology → Td=Tmin → zone OLS `Td~Tmin` →
fixed-5/h=11 → tuned.

**Metrics & effect sizes:** RMSE, MAE, bias, Pearson r (report these as primary, not cosine skill);
**skill score vs climatology** and vs Td=Tmin; **ΔRMSE °C with block-bootstrap CIs** (resample years/spatial
blocks to respect autocorrelation); **distribution of per-location improvements** + **fraction of cells where
tuned wins**; **Cohen's d** on paired per-location error diffs; **permutation/null test**.

**Secondary (robustness): spatial block CV** — partition Peru into contiguous tiles with a buffer dead-zone;
block *within* zone (recipes are zone-specific). Only needed to support the bootstrap CIs / show spatial
stability; not the primary holdout.

---

## 6. Deliverable — production gap-fill (2017–2020)

- [x] **P6: generalize the forecast to the tuned recipes.** DONE 2026-07-16 —
      `HPC_code_tunning/forecast_zoned.py` consumes the long/tidy `zoned_coeffs/` (variable per-zone×doy
      feature sets, TD lags to 30 d) + `climatology_by_bucket/` and runs the **autoregressive** recursion
      (predicted Td feeds next-day lags). Single-ID reference core + vectorised-across-IDs/loop-over-days
      bucket path (14× faster), kept in sync by `tests/test_forecast_zoned.py` (parity Δ<1e-9, independent
      hand-recursion, autoregressive-feed). No manifest needed at forecast time — the coeffs already encode
      each cell's retained features. Verified on real bucket-0 data (parity Δ=2.5e-14; day-1 math Δ=0).
- [ ] **Prep 2017–2020 v1.2 Tmin inputs — this is a REGRID, not a copy.** ⚠️ The raw v1.2 Tmin at
      `/media/ppalacios/Data1/henry_simcast_peru/tmin_v12/Outputs` is on a **302,449-ID grid** (max ID
      302448); the full product/coeffs use the **2,818,404-ID** grid. Feeding the raw file directly to the
      forecast mismatches the ID space (verified: produced a spurious −8.7 °C 2016→2017 seam jump; with the
      correctly-regridded full-grid Tmin the seam is −0.06 °C). The **regridded** full-grid v1.2 Tmin lives at
      `$FULL/tmin_v12/Outputs` but only covers **1981–2016** (432 files). B2 must run the same regrid/prep
      that built those, extended to 2017–2020, then bucketise into `future_tmin_root/id_bucket=XXXX/`
      (`[ID, FECHA, TMIN]`). Find the regrid step in `$FULL/extract_tmin.log` / the prep pipeline.
- [ ] **Run the full-grid fill** for 2017–2020 using the coeffs trained on all 1981–2016 → the extended Td
      product. Save under `$RES/tuning/predictions_2017_2020/` (persistent disk, never /tmp).
- [ ] Sanity/QC the filled product (ranges, seasonal cycle continuity across the 2016→2017 seam, maps).

---

## 7. Task checklist (work from the first unchecked item)

**Phase A — Validation harness (start here; no full re-run needed)**
- [ ] A1. Build the **baseline ladder** module (climatology, Td=Tmin, zone OLS) on existing artifacts.
- [ ] A2. Build the **forward-backtest harness**: train-years vs holdout-years split, **autoregressive**
       prediction of the holdout block, on a stratified ID sample. (Depends on P6, §6 — build P6 first.)
- [ ] A3. Build the **effect-size + bootstrap + null-test** reporting module.
- [ ] A4. Run backtest **variant (a) refit-only** for tuned + all baselines; produce the results table + plots.
- [ ] A5. Run backtest **variant (b) full re-select** on train-years; compare to (a) → selection-bias estimate.

**Phase B — Production deliverable**
- [ ] B0. **Re-select + re-train the full grid with the extended h-grid** `{7,11,15,21,31,45}` (§9), replacing
       the stale manifest/coeffs (§3). Do this once Phase A confirms tuning is worth it. ~25 h select + ~6 h train.
- [x] B1. Implement **P6 autoregressive forecast** for tuned recipes (§6). DONE 2026-07-16 —
       `HPC_code_tunning/forecast_zoned.py` + `tests/test_forecast_zoned.py` (3 passing). *(unblocks A2)*
- [ ] B2. Prep 2017–2020 v1.2 Tmin future inputs — **REGRID** raw 302k-ID Tmin → full 2.8M grid (see §6).
- [ ] B3. Full-grid fill 2017–2020; QC.

**Phase C — Write-up**
- [ ] C1. Draft Quarto/Typst paper: method, data, temporal validation, effect sizes, filled product.
- [ ] C2. **AI-use disclosure** + **SME sign-off** (CIP org policy).

---

## 8. Environment & reproduction

- **Working dir:** `/home/ppalacios/Documents/tdew_estimation`
- **Code branch:** the `HPC_code_tunning/` source is on **`main`** (also `feat/hpc-code-tunning`). Confirm with
  `ls HPC_code_tunning/*.py`; if empty, `git checkout main` (or `feat/hpc-code-tunning`).
- **Python:** `.venv/bin/python` (set `export PY=$PWD/.venv/bin/python`).
- **GPU:** single RTX A2000 12 GB; `cupy` works. Selection/training are GPU (`cupy`).
- **Runs must be detached** so they survive the session: `setsid nohup $PY -m ... > log 2>&1 < /dev/null &`
  (plain `nohup` inside the Claude session gets killed on session exit — use `setsid`). Outputs go to the
  **persistent** `$RES`, never `/tmp`.
- **Standard env block:**
  ```bash
  cd /home/ppalacios/Documents/tdew_estimation
  export PY=$PWD/.venv/bin/python
  export FULL=/media/ppalacios/Data1/henry_simcast_peru_full
  export RES=$FULL/results_tuning
  export ZIP=/home/ppalacios/Downloads/clasif_clima_peru.zip
  export TMIN12=/media/ppalacios/Data1/henry_simcast_peru/tmin_v12/Outputs
  ```
- **Reference commands (already-run full pipeline, for reuse with different `--train-years`):**
  ```bash
  # select (per-zone×doy feature/h search); ~25 h on full sample, reuses zone_table if present
  $PY -m HPC_code_tunning.run_tuning_hpc --base "$RES" \
    --coords "$FULL/td/Outputs/grid_index.parquet" --zones-zip "$ZIP" \
    --tmin-var tmin_v12 --train-years "1981 2016" \
    --per-zone-n 2000 --id-chunk 96 --h-grid 7,11,15,21,31,45 --granularity doy --stage select
  #   ^ extended h-grid (§9). The already-run manifest used the old {7,11,15,21}; re-run to adopt.
  #   For the Phase-A backtest re-select, also pass --train-years "1981 2012".
  # train (fit coeffs on full grid); ~6 h, ~2.67 s/bucket, output tuning/zoned_coeffs/
  $PY -m HPC_code_tunning.run_tuning_hpc --base "$RES" \
    --coords "$FULL/td/Outputs/grid_index.parquet" --zones-zip "$ZIP" \
    --tmin-var tmin_v12 --train-years "1981 2016" \
    --id-chunk 96 --granularity doy --stage train
  ```
  Progress signal for train (no per-bucket logging): `find $RES/tuning/zoned_coeffs -name coeffs.parquet | wc -l` (/8192).

---

## 9. Decisions locked
- Target = gridded v1.1 Td (product being extended). Station-level validation out of scope.
- Tmin source = **v1.2 throughout** (train + predict).
- Validation = **temporal forward backtest, autoregressive**; spatial CV is secondary.
- Effect sizes (ΔRMSE °C, skill scores, bootstrap CIs, null test) lead; Cohen's d as companion.
- Gap to fill = **2017–2020** (4 years).
- **Backtest holdout = leave-last-4 (train 1981–2012, predict 2013–2016)** [confirmed 2026-07-16], plus
  leave-last-1 and -2 for the lead-time curve.
- **h-grid extended to `{7,11,15,21,31,45}`** [confirmed 2026-07-16] — the old run saturated at 21, so give
  selection room to find the plateau. NOTE the trade-off: a very wide DOY window pools nearly the whole year
  and loses seasonal specificity, so if selection keeps picking the max that is itself a finding (it wants
  annual pooling). Top values `31,45` are a starting choice — adjustable if 45 is still the modal pick.

## 10. Open questions / inputs needed from the user
- Where should the paper draft + figures live (`reports/paper/`?).

## 11. Org policy (CIP) — must hold throughout
Validate AI-generated content with SMEs; no confidential/personal data; no unpublished third-party IP;
be mindful of bias; **disclose AI use** in the paper.

## 12. Progress log
- 2026-07-16 — Plan created. Prep/select/train DONE on 1981–2016 (§3). Data for 2017–2020 fill located (§2).
  Next action: **Phase A / B1** (build P6 autoregressive forecast, then the validation harness).
- 2026-07-16 — Confirmed decisions: backtest holdout = leave-last-4 (2013–2016); h-grid extended to
  `{7,11,15,21,31,45}`. Existing manifest/coeffs flagged stale (old h-grid) → Phase B0 re-run before the fill.
- 2026-07-16 — **B1/P6 DONE.** `HPC_code_tunning/forecast_zoned.py`: generalized autoregressive Td forecast
  over the long/tidy zoned coeffs (arbitrary per-zone×doy feature sets, TD lags to 30 d). Reference +
  vectorised paths, 3 passing tests. Verified on real bucket-0 (parity Δ=2.5e-14, day-1 math Δ=0). Pseudo-
  forecast of 2016 on correct full-grid Tmin (60 cells, full AR year): RMSE 1.535 / MAE 1.11 / r 0.944,
  **beats climatology (2.082)** — optimistic (coeffs saw 2016); honest number awaits the leave-last-4 backtest.
  **Key finding for B2:** raw v1.2 Tmin is a 302k-ID grid, product is 2.8M-ID → B2 is a REGRID (see §6);
  regridded full-grid v1.2 Tmin exists only for 1981–2016 at `$FULL/tmin_v12/Outputs`. A2 does NOT depend on
  B2 (backtest Tmin for 2013–2016 is already in the prepared shards). Methodological choice for A2: train-only
  climatology (1981–2012) to avoid holdout leakage. Next: A1 baselines + A2 backtest harness.
