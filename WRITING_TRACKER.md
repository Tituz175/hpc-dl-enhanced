# Writing Tracker

Thesis: *Deep Learning Enhanced Prediction of Execution Time, Memory Usage, and Power Consumption in Heterogeneous HPC Systems*

Tracks execution of **Track A** from the conceptualization plan (`~/.claude/plans/i-have-a-master-s-encapsulated-cat.md`). Update this as writing happens — it's a living document, not a one-time writeup. See `EXPERIMENT_TRACKER.md` for the code/experiment side.

Status values used throughout: `Not Started` / `Drafting` / `Drafted` / `Advisor Review` / `Revised` / `Final`.

Last updated: 2026-07-21

---

## Chapter Status

| Chapter | Title | Status | Draft % | Depends on (Track B) | Advisor Review Round | Notes |
|---|---|---|---|---|---|---|
| — | Front matter (title/approval/abstract) | Not Started | 0% | — | | Format per NMHU sample (Track A2) |
| 1 | Introduction | Not Started | 0% | — | | |
| 2 | Literature Review | Not Started | 0% | — | | Citation checklist below |
| 3 | Datasets and Feature Engineering | Not Started | 0% | Notebooks 01–02 | | |
| 4 | Methodology | Not Started | 0% | Notebooks 02–07 | | Roofline (F-DATA) vs. calibrated power model (PM100) must be described as distinct, not comparable |
| 5 | Experimental Setup and Results | Not Started | 0% | Notebooks 03–09 | | Full model-performance figure set → appendix; headline figures in body |
| 6 | Discussion, Threats to Validity, Conclusion & Future Work | Not Started | 0% | All notebooks | | One-by-one literature tie-back style (Track A2); PM100's Power-only Hybrid scope + FLOP-data gap goes in Future Work |
| — | References | Not Started | 0% | — | | |

---

## Advisor/Committee Open Items

Mirrors Track A5. Status: `Open` / `Raised with advisor` / `Resolved`.

| Item | Status | Resolution Notes |
|---|---|---|
| Sign-off on Tier A/B feature-leakage resolution (changes what RQ1 claims) | Open | |
| Whether F-DATA (homogeneous) + PM100 (heterogeneous) framing satisfies "heterogeneous HPC systems" in the title | Open | |
| 6-chapter/separate-Results-chapter format vs. the sample's 5-chapter format | Open | |
| Publication alongside the thesis | Resolved | Decided thesis-only for now |
| NMHU Graduate Division formatting requirements (margins, font, spacing, pagination) | Open | Distinct from department-level content template |

---

## Citation Checklist

One row per citation from Track A1 (Literature Review Grounding). Track whether it's been (re-)read in full, cited in Chapter 2, and whether the one-by-one Discussion-chapter tie-back (adopted from the NMHU sample's rhetorical style) has been written.

| Citation | Theme | Read in Full | Cited in Ch.2 | Discussion Tie-back Written |
|---|---|---|---|---|
| Yang et al. 2019 (Hierarchical Roofline) | HPC performance modeling | | | |
| Konstantinidis et al. 2017 (Quantitative/microbenchmark Roofline) | HPC performance modeling | | | |
| Ding et al. 2021 (Instruction Roofline) | HPC performance modeling | | | |
| Lopes et al. 2017 (Cache-aware Roofline w/ power) | HPC performance modeling | | | |
| Newaz et al. 2023 (RF/XGBoost for memory prediction) | ML for system performance | | | |
| Zhang et al. 2025 (RF/LightGBM/XGBoost ensemble) | ML for system performance | | | |
| Fatima et al. 2023 (RF vs. XGBoost comparison) | ML for system performance | | | |
| Pentsos et al. 2025 (LSTM-Transformer power forecasting) | DL for performance prediction | | | |
| Kai et al. 2025 (CNN-LSTM memory prediction) | DL for performance prediction | | | |
| Sabyasachi et al. 2023 (CNN+LSTM workload prediction) | DL for performance prediction | | | |
| Amirkhanloo et al. 2024 (CNN+LSTM+Bi-LSTM power prediction) | DL for performance prediction | | | |
| Aslam et al. 2025 (ML-driven scheduling) | Resource allocation/scheduling | | | |
| Zarif et al. 2025 (LSTM + PSO scheduling) | Resource allocation/scheduling | | | |
| Bal et al. 2022 (hybrid ML resource allocation) | Resource allocation/scheduling | | | |
| Bailey et al. 1991 (NAS Parallel Benchmarks) | HPC benchmarks & tracing | | | |
| Okada et al. 2016 (NPB in cloud environments) | HPC benchmarks & tracing | | | |
| Farrell et al. 2021 (MLPerf HPC) | HPC benchmarks & tracing | | | |
| Antici et al. (F-DATA / PM100 papers) | Prior work on these datasets | | | |
| Shwartz-Ziv & Armon 2021 (tabular DL vs. trees) | Model-choice grounding | | | |
| Grinsztajn et al. 2022 (tabular DL vs. trees) | Model-choice grounding | | | |
