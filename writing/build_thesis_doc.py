"""Build a Word draft of thesis content from the work done so far.
Run with: uv run python build_thesis_doc.py
"""
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

FIG_DIR = "/home/bizon/Documents/tobi/hpc-dl-enhanced/writing/figures"
OUT_PATH = "/home/bizon/Documents/tobi/hpc-dl-enhanced/writing/thesis_draft_notes.docx"

doc = Document()

# --- basic styling ---
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)

def h1(text):
    doc.add_heading(text, level=1)

def h2(text):
    doc.add_heading(text, level=2)

def p(text):
    doc.add_paragraph(text)

def fig(filename, caption):
    doc.add_picture(f"{FIG_DIR}/{filename}", width=Inches(5.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(10)

# ============================================================
doc.add_heading("Working Draft: Chapter 3 and Chapter 5 Material", level=0)
p("This document collects the writing-ready material produced while building the "
  "data pipeline for this thesis, organized by the chapter each section is "
  "expected to land in. "
  "None of this is final prose (line numbers, wording, and even section order "
  "will change once the surrounding chapters exist), but the substance, the "
  "numbers, and the reasoning behind each decision are accurate as of this draft "
  "and worth keeping rather than reconstructing later from memory.")

# ============================================================
h1("Chapter 3: Datasets and Feature Engineering")

h2("3.1 Data Acquisition and Schema Verification")
p("Two datasets anchor this study: F-DATA, drawn from the Fugaku supercomputer, "
  "and PM100, drawn from CINECA's Marconi100 system. Both are published "
  "job-accounting traces, but they record different things, and I wanted that "
  "difference established from the schema itself rather than assumed from the "
  "papers describing them.")
p("F-DATA arrived as 38 monthly parquet files, one per month from March 2021 "
  "through April 2024, totaling 25,866,900 job records, a number I confirmed "
  "directly from parquet metadata rather than the approximately-24-million figure "
  "quoted in the dataset's own paper. The file count is worth stating carefully: "
  "when this pipeline first ran, only one of the 38 files had actually been "
  "downloaded, and the loading code printed how many of the expected 38 it found "
  "rather than assuming the full set was present. All 38 were confirmed present "
  "by listing the directory directly (March 2021 through April 2024 with no "
  "gaps) before any of the numbers in this chapter that depend on the full "
  "date range were computed. PM100 is a single file of 231,238 jobs from "
  "Marconi100, covering May through October 2020.")
p("The original project proposal described both systems as heterogeneous "
  "CPU-GPU workloads. That framing does not survive contact with the schema. "
  "Fugaku's A64FX processors are CPU-only, with no GPU field to speak of "
  "in F-DATA, while PM100's nodes do carry GPUs. F-DATA's columns include six "
  "hardware performance counters (labeled perf1 through perf6), a FLOP count, "
  "memory bandwidth, and an operational-intensity field, none of which appear "
  "anywhere in PM100. I searched PM100's 35 columns for anything matching flop, "
  "perf, cycle, instr, mbwidth, or opint as a final check before ruling out "
  "Roofline-style analysis there, and found nothing. The two datasets are "
  "complementary, not symmetric: F-DATA supports genuine Roofline analysis "
  "because it has the hardware counters that method needs, and PM100 does not.")

h2("3.2 Target Definitions and a Units Discrepancy")
p("F-DATA's three prediction targets are duration (execution time), mmszu "
  "(memory used), and avgpcon (power). PM100's are run_time, mem_alloc, and "
  "node_power_consumption. PM100 has no field recording memory actually used, "
  "only requested and allocated amounts, so mem_alloc is not a preference "
  "between two options; it is the only one available.")
p("avgpcon needed a closer look before I trusted it as a target. Its documented "
  "name, average node power consumption, implies a single node's draw, but the "
  "maximum value in a 5,000-row sample was 259,097 watts, far beyond anything "
  "a single A64FX node could draw. Restricting to single-node jobs (nnuma == 1) "
  "brought the range down to 39.6–149.9 watts, consistent with published A64FX "
  "power figures, and avgpcon correlated with the number of allocated nodes at "
  "0.98. The field is a job-level total, summed across every allocated node and "
  "averaged over time, not a per-node figure. I checked the unit consistency a "
  "second way: dividing avgpcon multiplied by duration by the recorded energy "
  "field (econ) gave a ratio clustering around 1/3600 across the sample, which "
  "is exactly what you would expect if avgpcon is in watts, duration is in "
  "seconds, and econ is recorded in watt-hours rather than joules. The numbers "
  "are internally consistent once the units are understood correctly; the "
  "column name is just misleading about scope. This has one concrete "
  "consequence for later chapters: F-DATA's avgpcon and PM100's "
  "node_power_consumption turn out to be the same category of field. "
  "Both measure a job's total power draw, summed across every node it "
  "used, not a single node's reading. Section 4.4 confirmed this "
  "directly for node_power_consumption, by dividing each job's mean "
  "power by its allocated node count and finding the result holds close "
  "to constant, 677–732 watts, across node counts from 1 to 32. A true "
  "per-node reading would not need that division to land in the same "
  "range at every scale. The two fields measure the same kind of "
  "quantity, but not the same hardware or the same job scale, and any "
  "table that puts them side by side still needs to say so.")

p("Execution time, memory, and power are all heavy-tailed in both datasets: "
  "most jobs are short and small, and a long tail of much larger jobs pulls "
  "the mean well above the median. F-DATA's duration in a 5,000-row sample "
  "ranges from 1 second to 255,677 seconds (about 71 hours), with a median "
  "around 2,000 seconds, the kind of spread that will dominate a "
  "mean-squared-error loss if targets are trained on in raw units. All six "
  "targets across both datasets are therefore modeled in log1p space, with "
  "metrics reported back in real units afterward for interpretability. I "
  "did not assume the transform was safe; expm1(log1p(x)) was checked "
  "against the original values for all six targets and matched to within "
  "floating-point precision.")
fig("nb01_cell15_img1.png",
    "Figure 3.4. F-DATA execution time, raw versus log1p-transformed, "
    "illustrating the heavy-tailed distribution that motivates training in "
    "log space.")
p("The same pattern holds for the other five targets. F-DATA's memory and "
  "power, and all three of PM100's targets, show the same near-zero-dominated "
  "raw histogram spread into something usable once log1p-transformed.")
fig("nb01_cell15_img2.png",
    "Figure 3.5. F-DATA memory (mmszu), raw versus log1p-transformed.")
fig("nb01_cell15_img3.png",
    "Figure 3.6. F-DATA power (avgpcon), raw versus log1p-transformed.")
fig("nb01_cell16_img4.png",
    "Figure 3.7. PM100 execution time (run_time), raw versus log1p-transformed.")
fig("nb01_cell16_img5.png",
    "Figure 3.8. PM100 memory (mem_alloc), raw versus log1p-transformed.")
fig("nb01_cell16_img6.png",
    "Figure 3.9. PM100 power (node_power_consumption), raw versus "
    "log1p-transformed.")
p("Only one month of F-DATA was available when this pipeline was first "
  "built; all 38 months were obtained shortly afterward. The two "
  "job-count-over-time plots below cover the full range now available for "
  "both datasets and will matter directly in Chapter 4, where the "
  "chronological train/test split depends on there being enough time span "
  "to split meaningfully.")
fig("nb01_cell18_img7.png", "Figure 3.10. F-DATA jobs per day across the sampled files.")
fig("nb01_cell18_img8.png", "Figure 3.11. PM100 jobs per day across its full six-month span.")

h2("3.3 Submission-Time and Post-Hoc Feature Tiers")
p("The proposal's original feature diagram feeds measured FLOPs, memory "
  "bandwidth, and power directly into a network that predicts duration and "
  "power for the same job. That is circular if the intended use case is "
  "predicting a job's behavior before it runs, since none of those measured "
  "quantities exist until the job has already executed.")
p("I split every feature into two tiers to make this distinction impossible to "
  "blur by accident. Tier A holds only what is known at submission time: "
  "requested cores, nodes, and memory; queue and priority; requested wall time; "
  "an anonymized job-name embedding; and, for F-DATA, a rolling average of each "
  "user's recent job durations computed using only jobs strictly before the "
  "current one. On the current dev sample, that rolling average was populated "
  "for 4,426 of 4,721 F-DATA jobs (93.8 percent) and 3,717 of 3,952 PM100 jobs "
  "(94.1 percent). Coverage is high at this scale, but notebook 02 flags it as "
  "provisional rather than final: with only a handful of scattered months "
  "loaded so far, many of these rolling averages rest on very few prior jobs "
  "per user rather than a full window, and both the coverage rate and the "
  "values themselves are expected to firm up once more of the dataset is "
  "loaded. Tier B holds everything measured during or after execution: "
  "allocated versus used resources, exit codes, the hardware counters, and the "
  "power fields. Tier A is what a scheduler could actually use to predict a "
  "job's behavior before running it; Tier B exists for characterizing what "
  "happened after the fact and for the residual-learning step of the hybrid "
  "model described in Chapter 4. The two tiers are built by separate functions "
  "rather than one function with a flag, specifically so that mixing them "
  "requires deliberately writing code to do it. An assertion checks for column "
  "overlap between the two lists every time the module loads, and a second "
  "assertion re-checks after every transformation step in the feature pipeline, "
  "not just once at the start.")

h2("3.4 Embedding Dimensionality Reduction")
p("F-DATA's job-name field is anonymized and replaced with a 384-dimensional "
  "Sentence-BERT embedding. Feeding 384 raw dimensions into a random forest or "
  "gradient-boosted tree model is impractical, so the original thesis proposal "
  "called for reducing it with PCA to \"a handful of components.\" I had "
  "originally set that handful to 10 without checking what fraction of the "
  "embedding's variance 10 components actually capture.")
p("It turned out to be 68 percent on a single month's data, nowhere near "
  "enough to justify the number. I replaced the fixed default with an "
  "empirical check: fit PCA with up to 100 components, look at cumulative "
  "explained variance, and keep the smallest number that reaches a 90 percent "
  "threshold. On a sample drawn from one month, that threshold needed 43 "
  "components. On a sample drawn from five months spread across the full "
  "date range, it needed 59. The gap makes sense once you think about what "
  "the embedding encodes: a wider span of months carries more distinct job "
  "names and, presumably, more distinct applications, so capturing the same "
  "share of variance takes more components. The number is not fixed until a "
  "fit is done against the full 38-month dataset; treat 59 as current, not "
  "final.")
fig("nb02_cell22_img3.png",
    "Figure 3.1. Cumulative explained variance of F-DATA's job-name embedding "
    "against number of PCA components retained, computed on a 5-file sample. "
    "The dashed line marks the 90% threshold used to select the component count.")
p("I also decided, after an earlier round of discussion, to use the "
  "PCA-reduced embedding everywhere (for the tree-based models and for the "
  "neural network architectures alike) rather than giving the deep learning "
  "models the option of the raw 384-dimensional vector, which the original "
  "proposal allowed for. The reason is practical rather than statistical: "
  "loading the raw embedding for the full F-DATA dataset measures at over 100 "
  "gigabytes of resident memory, against roughly 27 gigabytes for every other "
  "column combined, because each row's embedding is stored as an individually "
  "allocated array rather than one contiguous block. Section 4.2 covers this "
  "measurement and the loading strategy it led to in more detail.")

h2("3.5 Feature Vetting: PM100's Missing-by-Design Fields")
p("Three PM100 columns are Tier A candidates and all three have missing "
  "values, but at very different rates: num_tasks is missing in 4.1 percent "
  "of jobs, req_nodes in 88.6 percent, and threads_per_core in 99.6 percent. "
  "Treating all three the same way (impute and move on) would have missed "
  "what turned out to be a fairly consequential distinction, so I worked "
  "through a four-part check on each candidate before deciding whether its "
  "derived feature belonged in the active Tier A set.")
p("The first check is simply whether the missingness reflects an optional "
  "field or an incomplete one. req_nodes, when populated, holds a specific "
  "node identifier (values like [763] or [878]), which is Slurm's explicit "
  "node-pinning flag (--nodelist). Most jobs do not ask for a particular "
  "machine, so the field is null by design for them, not because logging "
  "failed. threads_per_core is the same story: its only non-null values are "
  "1, 2, or 4, matching Slurm's explicit hyperthreading override "
  "(--threads-per-core), a setting almost nobody touches. num_tasks has no "
  "such documented optional-flag semantics; its gaps look like ordinary "
  "incomplete logging.")
p("The second check is effect size against the three prediction targets, "
  "using median comparisons rather than leaning on p-values alone: at "
  "roughly 180,000 completed jobs, a Mann-Whitney test will report "
  "significance for almost any difference, trivial or not. All three derived "
  "flags (node_pinned, threads_per_core_set, and num_tasks_missing) showed "
  "real median differences. threads_per_core_set showed the largest of the "
  "three by a wide margin: median run_time roughly four times higher, median "
  "memory roughly four times lower, when the flag is set.")
fig("nb02_cell11_img1.png",
    "Figure 3.2. Median run_time, mem_alloc, and power for each candidate flag, "
    "true versus false, on a log scale. threads_per_core_set shows the largest "
    "separation of the three despite being the rarest.")
p("The third check is redundancy: does an existing categorical column already "
  "carry the same information as the flag being considered? I cross-tabulated "
  "threads_per_core_set against partition, quality-of-service class, and GPU "
  "allocation. None of them explained it: the flag is set almost exclusively "
  "within the system's single dominant partition and QoS class, which is "
  "where nearly every job lives regardless, and GPU allocation among "
  "flagged jobs was close to the overall rate. Ruling out redundancy raised "
  "my confidence in the flag rather than lowering it, which is why the fourth "
  "check mattered more than I expected going in.")
p("The fourth check is user concentration: is the signal broad, or is it a "
  "handful of individual users' behavior showing up as if it were a general "
  "pattern? Both datasets are naturally dominated by a small number of heavy "
  "users (PM100's top five account for 45.4 percent of all completed jobs, "
  "F-DATA's top five for 41.4 percent in a five-month sample), so some "
  "concentration is expected and not automatically disqualifying. The "
  "question is how much concentration is too much. node_pinned turned out to "
  "be far more concentrated than that baseline: only 19 users ever set it, "
  "and the top five account for 99.3 percent of every pinned job, 73.5 "
  "percent from one user alone. threads_per_core_set was similarly narrow: "
  "only 6 users, 85 percent of occurrences from a single one of them. "
  "num_tasks_missing, by contrast, spans 101 distinct users with the top "
  "five accounting for 47.1 percent, close enough to the dataset's own "
  "baseline that it reads as a genuine pattern rather than a few users' "
  "signature.")
fig("nb02_cell17_img2.png",
    "Figure 3.3. Top-1 and top-5 user share of jobs for each candidate flag, "
    "compared against the dataset's overall baseline concentration (dashed "
    "line). node_pinned and threads_per_core_set both sit well above the "
    "baseline; num_tasks_missing sits close to it.")
p("Only num_tasks_missing passes all four checks, and it is the feature "
  "promoted into the active Tier A set. node_pinned and threads_per_core_set "
  "are both computed and kept in the processed dataset (along with the raw "
  "req_nodes and threads_per_core columns) but excluded from the general "
  "feature set, reserved for a narrower, explicitly scoped analysis later "
  "(a power-specific submodel is the most likely candidate, given "
  "threads_per_core_set's relationship to SMT settings) where the "
  "single-user-concentration caveat can be addressed directly rather than "
  "quietly inherited by every model that uses Tier A. threads_per_core_set "
  "having the largest raw effect size of the three, and still being the one "
  "excluded, is worth stating plainly: effect size on its own was not "
  "sufficient evidence, and would have been misleading without the "
  "concentration check that followed it.")

p("That decision covers three candidate features. It says nothing about "
  "whether the same problem was already sitting undetected somewhere in the "
  "feature set built before this vetting process existed, so I went back "
  "and ran the same concentration check against every PM100 Tier A column "
  "that predates it. One failed outright: req_switch, a Slurm switch-count "
  "field already in active use, has only 13 non-zero values across 180,310 "
  "completed jobs, and all 13 come from a single user, a worse "
  "concentration ratio than threads_per_core_set, which had already been "
  "excluded for the same reason. req_switch has now been moved to the same "
  "reserved category, with a derived req_switch_set flag preserved "
  "alongside it for consistency. The rest of the audit came back clean: "
  "shared's two populated categories each hold tens of thousands of jobs, "
  "nowhere near single-user territory, and the smallest categories of qos "
  "and partition, while occasionally tiny, are legitimate multi-valued "
  "scheduling fields rather than purpose-built rare flags, so ordinary "
  "small-category noise there is not the same concern. F-DATA's one "
  "comparable column, jobenv_req, showed a milder version of the same "
  "pattern (its minority value spans 21 users with a 39.2 percent top-user "
  "share, above F-DATA's own baseline concentration of 16.0 percent but "
  "well short of PM100's excluded flags) and is left active for now, "
  "flagged for a second look once more months of F-DATA are loaded.")

p("There is a second, broader question this whole exercise raised that is "
  "worth stating on its own rather than folding into the feature-by-feature "
  "discussion above: given that a handful of users account for close to "
  "half of all jobs in both datasets, does the chronological train/test "
  "split planned for later chapters risk giving an unrepresentative slice "
  "of any one heavy user's jobs to training versus testing? I checked this "
  "for PM100's five heaviest users directly, rather than leaving it as an "
  "open question. All five are active across 97 to 100 percent of the "
  "dataset's full 159-day span, with one exception reaching 76 percent "
  "after starting roughly a month into the window. None of them are "
  "concentrated in a narrow slice of time that a chronological split could "
  "isolate entirely on one side. That is a reassuring result, but it is "
  "specific to PM100. F-DATA spans 38 months rather than 6, which is a "
  "much longer window for a user's activity to plausibly shift, taper off, "
  "or start partway through before the dataset ends, and that check has "
  "not yet been run there. The Feature Vetting method described in this "
  "section should be treated as an ongoing discipline for this thesis "
  "rather than a step that was completed once and closed: the fact that "
  "it caught req_switch on a second pass, after already being applied "
  "once, is itself the argument for continuing to apply it as new features "
  "and the full-scale dataset come into the pipeline.")

h2("3.6 Handling Incomplete Jobs")
p("Both datasets record jobs that failed, were cancelled, or hit resource "
  "limits before completing. Their duration and power figures reflect the "
  "failure, not ordinary workload behavior, so both datasets are filtered to "
  "completed jobs only before any target is computed. F-DATA's exit-state "
  "field is a clean binary and excludes 5.1–5.6 percent of jobs depending on "
  "the sample; PM100's job-state field has six values (completed, failed, "
  "cancelled, timeout, out-of-memory, and node failure) and excludes a "
  "notably larger 21.0 percent. That gap between the two systems' failure "
  "rates is itself worth stating explicitly rather than treating the two "
  "datasets as symmetric on this point.")

# ============================================================
doc.add_page_break()
h1("Chapter 4 (partial): Methodology")
p("This chapter covers how the work in this thesis was built, set up, and "
  "validated: the computing environment, the memory constraints that "
  "shaped the data-loading strategy, the notebook-execution tooling, and "
  "the construction and validation of the two analytical baselines. Model "
  "architectures and training procedures for the classical ML, deep "
  "learning, and hybrid models will be added here once that work exists; "
  "this is not the complete chapter.")

h2("4.1 Computing Environment")
p("Work on this thesis began on a single workstation with an Intel i9-10900X "
  "(10 cores, 20 threads), 125 gigabytes of RAM, and one NVIDIA RTX 3090. It "
  "moved partway through to a replacement machine, an Intel i9-10980XE with "
  "18 cores and 36 threads, 251 gigabytes of RAM, and four NVIDIA RTX 3090 "
  "Ti cards, because the original's memory could not hold F-DATA's embedding "
  "matrix at full scale and a single GPU bottlenecks the later deep-learning "
  "stage. Every full-scale result reported here was produced on the "
  "replacement machine, and the microbenchmark validation of Section 4.5 was "
  "re-measured on one of its RTX 3090 Ti cards. The environment is managed "
  "with uv rather than plain pip, "
  "pinned to Python 3.12 rather than the newer 3.13 available on the "
  "system. That choice was not arbitrary: numba, a dependency of the SHAP "
  "interpretability library used later in this work, does not yet build "
  "cleanly under Python 3.13. More specifically, resolving the full "
  "dependency set under the newer Python version led the package resolver "
  "to select numba 0.53.1, a version old enough that its own dependency, "
  "llvmlite, refuses to build at all above Python 3.9. The fix needed an "
  "explicit lower bound on numba in the project's dependency list; without "
  "it, the resolver's preference for a version whose metadata claims "
  "unbounded compatibility, even when that claim is stale, wins over a "
  "modern release that correctly declares its actual constraints. PyTorch "
  "is installed from the CUDA 13.0 build, matching the workstation's driver "
  "version exactly, and its GPU detection was confirmed directly rather than "
  "assumed from the installation succeeding.")

h2("4.2 Memory Constraints at Full Scale")
p("F-DATA's 38 monthly files total roughly 26 gigabytes on disk, comfortably "
  "within the workstation's storage, but disk size is not the same question "
  "as memory footprint once loaded. Measuring one month's file directly: "
  "loading all 45 columns costs 5.54 gigabytes of resident memory, while "
  "loading the same file without the embedding column costs 1.39 gigabytes. "
  "Scaled to the full 25,866,900-row dataset, that difference becomes "
  "significant: upwards of 100 gigabytes with the embedding included, "
  "against roughly 27 gigabytes without it. Given 125 gigabytes of total "
  "RAM, the first scenario would leave little headroom for anything else "
  "running on the machine, let alone for the model training this dataset "
  "exists to support.")
p("The inefficiency is specific to how the embedding is stored: each row's "
  "384-dimensional vector is an individually allocated NumPy array rather "
  "than a slice of one contiguous block, which carries far more overhead per "
  "row than the memory the numbers themselves would need. The fix separates "
  "fitting the PCA transform from applying it. The transform is fit once, on "
  "a sample small enough to hold comfortably in memory, and the fitted model "
  "is then applied to one monthly file at a time in a loop, so the raw "
  "embedding for more than one month's worth of rows is never resident at "
  "once. A function that loads every other column across all 38 files "
  "without touching the embedding at all covers the case where "
  "embedding-derived features are not needed for a given step.")

h2("4.3 Notebook Execution and Monitoring")
p("Early runs of the data-loading notebook appeared to hang for extended "
  "periods with no visible output, which turned out to have two distinct "
  "causes rather than one. The first was calling a generic describe() "
  "across a dataframe that still included the embedding column: pandas "
  "attempts to compute uniqueness and mode statistics on that column, which "
  "means comparing unhashable NumPy arrays across a million rows, and the "
  "fallback path for that is slow rather than an outright error. The second "
  "was treating PM100's node_power_consumption as if it were a single "
  "number per job, when it is in fact a time series, power sampled roughly "
  "every 20 seconds for the duration of the job, which is exactly the kind "
  "of sequence data the LSTM and TCN architectures later in this chapter "
  "are meant to use. Both were fixed at the source rather than worked "
  "around: the embedding column is excluded from any generic summary call, "
  "and any scalar treatment of the power column reduces the sequence to a "
  "per-job mean explicitly first.")
p("Notebooks are now executed with papermill rather than plain jupyter "
  "nbconvert, specifically because nbconvert only writes output once "
  "execution finishes entirely, which gives no way to distinguish a slow "
  "cell from a stalled one while it is running. papermill logs each cell's "
  "start and end as it happens, which is what actually resolved the "
  "hanging-versus-slow question above rather than guesswork based on "
  "process CPU time.")

p("Sections 4.4 and 4.5 are one methodological thread, not two separate "
  "topics. F-DATA's Roofline ceiling in 4.4 is built entirely from the "
  "dataset's own reported FLOP and bandwidth counters, which leaves the "
  "construction itself unverified: does the ceiling formula and its "
  "compute/memory-bound classification actually recover the right answer, "
  "on hardware where the right answer is independently known, rather than "
  "only looking plausible because the same source produced both the "
  "inputs and the label it's checked against? Section 4.5 exists to "
  "answer exactly that, using controlled GPU kernels timed directly on "
  "this workstation rather than F-DATA's self-reported figures. 4.4 "
  "describes what was built; 4.5 describes why it can be trusted. The "
  "actual outcomes of both (the R² and MAPE numbers, the calibrated "
  "coefficients, the classification results) are reported afterward, in "
  "Chapter 5.")

h2("4.4 Analytical Baseline Construction")
p("Notebook 03 built the two analytical baselines this study needs: a "
  "Hierarchical Roofline model for F-DATA (Yang et al. 2019), and a "
  "calibrated resource-utilization power model for PM100. The two "
  "datasets need different treatments here, not just different numbers: "
  "F-DATA has the hardware performance counters Roofline requires, and "
  "PM100 does not, a distinction established already in Section 3.1.")
p("Notebook 03 was first developed and run on a six-month slice of F-DATA: "
  "the consecutive files from March through August 2021, rather than the "
  "scattered-month sampling notebooks 01 and 02 used at that point. A "
  "chronological train/test split only makes sense across a contiguous span "
  "of time, and six scattered months would not have provided one. That "
  "six-month slice held 2,897,734 F-DATA rows; the completed-jobs filter "
  "from Section 3.6 removed 10.5 percent of them, leaving 2,593,124. PM100 "
  "was loaded in full, as everywhere else in this thesis: 231,238 rows, "
  "22.0 percent removed by the same filter, leaving 180,310.")
p("The chronological split was implemented here for the first time and then "
  "reused unchanged by every later notebook. Submission time is sorted, the "
  "70th percentile becomes the boundary, jobs before it train and jobs "
  "after it test, with the boundary computed as a quantile rather than "
  "hand-picked. On the six-month F-DATA slice the boundary fell on July 1, "
  "2021, giving 1,815,596 training and 777,528 test rows. On PM100 it fell "
  "on September 19, 2020, giving 126,217 training and 54,093 test rows. The "
  "split lives in a shared module, so every model in this thesis is "
  "measured against the same boundary.")
p("F-DATA's Roofline ceiling comes from the A64FX processor's published "
  "specifications: 3.3792 teraFLOP/s double-precision peak and 1024 "
  "gigabytes per second of peak HBM2 bandwidth, both per node. I checked "
  "two things directly against the data before trusting the ceiling, "
  "rather than taking them from the column documentation. The first is "
  "that opint, F-DATA's own operational-intensity field, equals flops "
  "divided by mbwidth. It does, exactly, to a relative error of 0.00e+00 "
  "on the six-month slice, so F-DATA's authors precompute it and it does "
  "not need recomputing. The second is that flops and mbwidth are job-wide "
  "totals rather than per-node figures, the same question Section 3.2 "
  "raised about avgpcon. On the six-month slice, 977 jobs exceeded the "
  "A64FX's single-node peak FLOP/s when node count was ignored, and none "
  "exceeded it once each job's rate was divided by its node count. Both "
  "fields are job-wide totals, so the ceiling is scaled by each job's node "
  "count before any comparison.")
p("I later re-ran the notebook across all 38 months of F-DATA on the "
  "replacement workstation described in Section 4.1. At that scale the "
  "completed-jobs filter removed 9.9 percent of 25,866,900 rows, leaving "
  "23,308,381, split 70/30 by submission time into 16,315,867 training and "
  "6,992,514 test rows. The two direct checks above were repeated on the "
  "full population and held: opint matched flops divided by mbwidth exactly "
  "across all 23,308,381 jobs, and no job exceeded the per-node compute "
  "peak after scaling by node count. Every Roofline number reported in "
  "Chapter 5, including its accuracy against held-out execution time, comes "
  "from this full-scale run. The six-month figures above describe only the "
  "development slice the construction was built and debugged on.")
p("PM100 has no FLOP, instruction, or performance-counter field anywhere "
  "in its schema (confirmed directly against documentation/job_features.md, "
  "not assumed from the absence noted in Section 3.1), so no Roofline "
  "variant is computable there. This section builds a calibrated power "
  "model instead, fixed in functional form by hardware reasoning (an idle "
  "baseline plus a linear contribution per allocated resource) with only "
  "the coefficients fit to data.")
p("Building that model meant revisiting an assumption from Section 3.2, "
  "which described PM100's node_power_consumption as genuinely per-node, "
  "in contrast to F-DATA's job-total avgpcon. That assumption does not "
  "hold. Dividing each job's mean power by its allocated node count gives "
  "a per-node figure that stays close to constant across every node count "
  "from 1 to 32: 731.1 watts at 1 node, 720.1 at 2, 717.4 at 4, 701.8 at 8, "
  "677.3 at 16, and 732.1 at 32. A reading that were actually per-node "
  "would not need dividing by node count to land in the same range at "
  "every scale; the fact that it does only after dividing means the raw "
  "field is already summed across every allocated node at each 20-second "
  "sample, the same kind of quantity avgpcon turned out to be in F-DATA. "
  "This is a correction to Section 3.2's framing, not an extension of it, "
  "and the power model below is built for the corrected, job-total "
  "interpretation: the idle-power term scales with the number of nodes "
  "allocated, rather than being added once per job.")
fig("nb03_cell19_img2.png",
    "Figure 4.1. PM100's mean node_power_consumption divided by allocated "
    "node count, for jobs at each node count from 1 to 32. A per-node "
    "reading would not need this division to stay flat across scales.")
p("Before fitting anything, I checked how the model's four predictors "
  "(allocated nodes, cores, GPUs, and memory) relate to each other, since "
  "Marconi100 assigns cores, GPUs, and memory to a job in close to a fixed "
  "ratio per node. They are correlated with each other at 0.85 to 0.97 "
  "across every pair. That is worth knowing before looking at the "
  "calibrated coefficients, reported in Chapter 5, not after.")
p("F-DATA's Roofline baseline and PM100's power model are not directly "
  "comparable to each other: different formulas, different physical "
  "quantities, and different targets, execution time against power. Any "
  "table in this thesis that places both side by side, including the one "
  "notebook 03 builds for its own recordkeeping, exists for convenience of "
  "layout, not as a claim that the two numbers mean the same thing.")
p("Section 3.3 called re-checking assertions after every transformation "
  "step a deliberate discipline rather than a formality, and this notebook "
  "kept it up: opint's non-negativity and finiteness, the Roofline ceiling "
  "never exceeding the compute-bound peak, positive predicted durations, "
  "finite calibration coefficients, the log1p/expm1 round-trip, and no "
  "leakage across the chronological split all passed before any number "
  "reported in Chapter 5 was trusted.")
p("Two things are deferred past this notebook. The rolling-window "
  "temporal-drift check across F-DATA's full span, training on one window "
  "and testing on the next while sliding forward, is separate from the "
  "single chronological split used here and has not been run. And the "
  "baselines built here, Roofline and the naive median for F-DATA and the "
  "calibrated power model for PM100, are meant to be reused in the full "
  "evaluation sweep planned for notebook 08, in the same comparison tables "
  "as the random forest, gradient-boosted tree, feedforward, recurrent, "
  "and hybrid models covered in later chapters.")

h2("4.5 Microbenchmark Validation Methodology")
p("Section 4.3 raised this question without answering it. This section "
  "answers it directly: Notebook 04 is a small, self-contained "
  "side-study, independent of F-DATA and PM100 entirely, that runs a "
  "handful of GPU kernels locally, times them directly, and checks "
  "whether the same Roofline construction recovers the right answer on "
  "hardware where the right answer is known in advance rather than "
  "merely self-reported.")
p("It runs on the replacement workstation's own RTX 3090 Ti, confirmed "
  "directly rather than assumed: a full GA102 die, 84 streaming "
  "multiprocessors, compute capability 8.6, 25.3 gigabytes of memory. "
  "Published specs put its double-precision peak at 0.625 teraFLOP/s and "
  "its single-precision peak at 40.0 teraFLOP/s (consumer Ampere runs "
  "FP64 at 1/64 the FP32 rate, a restriction a datacenter part like the "
  "A100 doesn't share), and its memory bandwidth at 1008 gigabytes per "
  "second across a 384-bit GDDR6X bus clocked at 21 gigabits per second. "
  "The original workstation's plain RTX 3090 put these at 35.58 and 0.556 "
  "teraFLOP/s and 936.2 gigabytes per second; the first run of this "
  "validation used those figures.")
p("NAS Parallel Benchmarks and Rodinia's kernel suite were the natural "
  "source material for this side-study, following Konstantinidis et al. "
  "2017's microbenchmark-based Roofline approach. Neither suite was actually "
  "compiled and run here; four standard operations implemented directly "
  "in PyTorch substitute for them: vector_add (SAXPY/DAXPY), "
  "dot_product, gemv (matrix-vector multiply), and gemm (matrix-matrix "
  "multiply). These four were chosen to span the same low-to-high "
  "arithmetic-intensity range Rodinia's kernels do, in less time than "
  "standing up an external C/CUDA build. None of these four resemble "
  "anything in F-DATA's actual "
  "job mix, and that's not an oversight: this notebook validates the "
  "Roofline construction itself, not F-DATA's workload characteristics, "
  "and validating a construction requires kernels whose FLOP count and "
  "byte count are known exactly, so their true compute- or memory-bound "
  "regime is known in advance. F-DATA's own jobs offer no such "
  "independent ground truth, only the dataset's own precomputed pclass "
  "label, which risks confirming the same method it's meant to check "
  "rather than testing it against something external. Konstantinidis et "
  "al. 2017's own approach uses this same logic: synthetic, controlled "
  "kernels trace a GPU's achievable-performance curve independent of "
  "whatever workload eventually runs against it. Dense linear algebra "
  "isn't an arbitrary stand-in either (GEMM and GEMV underlie a large "
  "share of real scientific computing, including several NAS Parallel "
  "Benchmarks kernels from Bailey et al. 1991), but no claim is made that "
  "these four kernels reproduce F-DATA's actual jobs; that would need profiling "
  "real F-DATA-representative application code, which is out of scope "
  "here.")
p("Each kernel ran across several problem sizes, in both FP32 and FP64, "
  "timed with CUDA events rather than wall-clock time so host-side "
  "Python overhead around the kernel launch wouldn't be included: five "
  "warmup calls discarded, then twenty further calls averaged.")
p("Building this validation surfaced a real finding along the way, not a "
  "bug. The first version of the sanity checks failed: vector_add at the "
  "smallest FP32 size (1,000,000 elements, an 8-megabyte working set "
  "across its two arrays) measured 115.1 percent of the RTX 3090 Ti's 1008 "
  "gigabyte-per-second DRAM peak. The same kernel at 5,000,000 elements "
  "(40 megabytes) and above measured a normal 87 to 90 percent, so the "
  "excess was confined to a single point. On the original RTX 3090 the "
  "same measurement came out at 124.6 percent of that card's lower "
  "936.2 gigabyte-per-second peak: a larger excess, because the L2 "
  "cache's fixed contribution is a bigger fraction of a smaller "
  "denominator. The explanation is the GPU's "
  "memory hierarchy, not measurement error: the timing harness calls the "
  "same kernel on the same tensors twenty times in a row, and GA102's "
  "6-megabyte L2 cache is large enough to serve a meaningful share of an "
  "8-megabyte working set's repeated accesses, so the achieved rate "
  "reflects a blend of cache and DRAM bandwidth rather than DRAM alone. "
  "dot_product at the identical working set didn't show the same excess "
  "(52.9 percent of peak), and FP64 vector_add at the same element count "
  "(a 16-megabyte working set, twice the bytes) didn't either (82.9 "
  "percent), which ruled out working-set size alone as the explanation. "
  "The specific memory-access pattern matters too: a reduction kernel "
  "apparently doesn't benefit from repeated-call cache residency the way "
  "a straightforward elementwise write does. I fixed "
  "this by adding a working-set-size field to each kernel result and "
  "excluding rows within twice the L2 cache size from the DRAM-bandwidth "
  "check, a carve-out justified by the specific row it was built to "
  "explain rather than a tolerance loosened until the assertion stopped "
  "failing.")
p("This ties directly back to why Yang et al. 2019's Hierarchical "
  "Roofline, introduced in Section 4.4, models multiple bandwidth levels "
  "(L1, L2, device memory, system memory) rather than a single DRAM "
  "figure. A roofline built against DRAM bandwidth alone is exactly the "
  "kind of model a small, cache-friendly problem can appear to violate, "
  "when what actually happened is the problem ran against a faster level "
  "of the memory hierarchy instead. F-DATA's own Roofline used a single "
  "measured bandwidth field and couldn't have caught this the way a "
  "controlled microbenchmark can; it is worth remembering when reading "
  "Section 4.4's ceiling as if it captured the whole hierarchy, since it "
  "doesn't.")
p("The same scope limitation stated for F-DATA's own Roofline applies "
  "here too, for the same reason: this workstation's RTX 3090 Ti is neither "
  "Fugaku's A64FX nor Marconi100's V100s, so nothing here is a "
  "hardware-matched ceiling for either dataset. What it validates is the "
  "construction: does a real, measured achieved-performance-versus-"
  "intensity relationship take the shape the Roofline theory predicts, "
  "on hardware where the correct answer is known in advance. This "
  "notebook is self-contained and doesn't feed into any other notebook, "
  "which made it the cheapest piece of this work to have cut entirely "
  "had time run short elsewhere. It didn't, so it stayed in scope.")

# ============================================================
doc.add_page_break()
h1("Chapter 5 (partial): Results")
p("This chapter reports what the methodology in Chapter 4 actually "
  "produced: the analytical baselines' accuracy against held-out data, "
  "the microbenchmark validation's classification results, and the "
  "classical machine-learning baselines for execution time, power, and "
  "memory. Deep-learning and hybrid-model results will be added once "
  "notebooks 06 through 09 exist; this is not yet the complete chapter.")

h2("5.1 Analytical Baseline Results")
fig("nb03_cell12_img1.png",
    "Figure 5.1. F-DATA Roofline, per-node operational intensity against "
    "per-node performance, for 23,308,381 completed jobs across all 38 months. The red ceiling combines the compute-bound peak "
    "(3.3792 TFLOP/s) and the memory-bound line (operational intensity "
    "times 1024 GB/s); the dashed line marks the ridge point at 3.30 "
    "FLOP/byte.")
p("Classifying each job as compute-bound or memory-bound from its position "
  "relative to the ridge point recovered F-DATA's own precomputed pclass "
  "label in 99.974 percent of jobs, which is as close to a direct "
  "confirmation of the ceiling's correctness as this data can give.")
p("The Roofline ceiling implies a best-case duration for every job: how "
  "long it would take running at the fastest rate its own measured "
  "operational intensity allows. Evaluated against actual duration on the "
  "held-out test split, this scored an R² of -0.164 and a MAPE of 99.98 "
  "percent. A naive baseline (each job's own user's median duration over "
  "their training-period jobs, falling back to the training-period global "
  "median for users with no training history) scored an R² of 0.080 and "
  "a MAPE of 8,369 percent on the same split. The Roofline figure being "
  "negative is not a bug in the baseline. The Roofline ceiling assumes a job "
  "spends its entire wall-clock duration running at peak achievable rate, "
  "with no time lost to I/O, synchronization, or any other non-compute "
  "phase, so its predicted durations come out roughly five orders of "
  "magnitude below actual ones: a job predicted to finish in a fraction "
  "of a second actually running for thousands of seconds is the expected "
  "outcome of that assumption, not a sign the ceiling was computed wrong. "
  "The naive baseline's enormous MAPE is a known property of that metric "
  "when actual values sit close to zero (F-DATA's minimum recorded "
  "duration is 1 second) rather than evidence the median itself is a bad "
  "estimate; RMSE and R² tell a more usable story for this baseline than "
  "MAPE does on its own. Both baselines are exactly the floor the "
  "classical ML, deep learning, and hybrid models in later chapters need "
  "to clear, and a physics-informed model landing this far below even a "
  "naive median is itself worth stating plainly in this thesis rather than "
  "only reporting the numbers that make later models look good.")
p("The power model was calibrated by ordinary least squares on the "
  "training split only, never touching the test split: an idle-power "
  "coefficient of 747.6 watts, 0.256 watts per allocated core, -23.674 "
  "watts per allocated GPU, and -0.126 watts per megabyte of allocated "
  "memory. The idle-power figure lands close to the 677-732 watt range "
  "measured in Section 4.4, which is a reasonable plausibility check on "
  "the fit. The GPU and memory coefficients coming out negative is "
  "physically backwards for resources that should only add to a node's "
  "power draw, and the correlation checked in that same section is the "
  "direct explanation: with four predictors this closely tied to each "
  "other, ordinary least squares has no clean way to attribute the "
  "outcome to one of them over its near-duplicate neighbors, even when "
  "the resulting fit predicts well overall. I am reporting the "
  "coefficients as the fit produced them rather than adjusting them to "
  "look more physically sensible, since that would misrepresent what the "
  "calibration actually found.")
fig("nb03_cell25_img3.png",
    "Figure 5.2. PM100 calibrated power model, predicted against actual "
    "mean power on the training-period calibration subset. The dashed "
    "line marks perfect prediction.")
p("On the held-out test split, the calibrated model scored an R² of 0.910 "
  "and a MAPE of 21.82 percent, against the naive per-user-median "
  "baseline's R² of 0.096 and MAPE of 49.75 percent on the same jobs, a "
  "wide margin in the calibrated model's favor. Read together with the "
  "Roofline results above, the two analytical baselines behave very "
  "differently relative to their own naive floors: PM100's power target "
  "turns out to be far more predictable from allocated resources alone "
  "than F-DATA's duration is from a user's own recent history, across the "
  "full F-DATA span and the full PM100 dataset. The model's "
  "predictions are trustworthy on this evidence; its individual "
  "coefficients, for the reason given above, are not.")

h2("5.2 Microbenchmark Validation Results")
p("Unlike F-DATA, the RTX 3090 Ti kernels described in Section 4.5 have a "
  "regime known in advance by construction: vector_add and dot_product "
  "are memory-bound at any problem size, and gemm should move toward "
  "compute-bound as the matrix size grows, since its FLOPs scale with "
  "the cube of the dimension against memory traffic that only scales "
  "with the square. The classification logic recovered exactly that in "
  "every run: vector_add and dot_product classified memory-bound in 8 "
  "of 8 runs at both precisions, and the largest gemm (8192×8192) "
  "classified compute-bound, reaching 74.7 percent of the FP32 peak and "
  "95.1 percent of the FP64 peak. The crippled consumer FP64 rate is low "
  "enough that a modestly sized GEMM nearly saturates it. Because these kernels' true regime is known "
  "independently rather than only cross-checked against a dataset's own "
  "label, this is a stronger validation of the Roofline construction "
  "than the F-DATA Roofline's own 99.974 percent pclass agreement in "
  "Section 5.1 was.")
fig("nb04_cell10_img1.png",
    "Figure 5.3. RTX 3090 Ti measured Roofline, FP32. Achieved performance "
    "for all four kernels across their tested problem sizes, against the "
    "40.0 TFLOP/s compute-bound ceiling and the 1008 GB/s "
    "memory-bound line.")
fig("nb04_cell11_img2.png",
    "Figure 5.4. RTX 3090 Ti measured Roofline, FP64. Same construction as "
    "Figure 5.3, against FP64's much lower 0.625 TFLOP/s compute-bound "
    "ceiling. GEMM saturates it at far smaller matrix sizes than in "
    "FP32.")

h2("5.3 Classical Machine-Learning Baselines: Execution Time and Power")
p("Notebook 05 fits three tree-ensemble models, a random forest and the "
  "gradient-boosted libraries XGBoost and LightGBM, against F-DATA's "
  "execution time and PM100's power, using only submission-time (Tier A) "
  "features. Each model gets the same fixed search budget: a 50-trial "
  "Optuna sweep over its own hyperparameters, tuned on a chronological "
  "validation carve-out that never touches the test split. Targets are fit "
  "in log1p space and metrics are reported back in raw units. For F-DATA "
  "the sweep runs on a one-million-row sample stratified by job size, and "
  "the final model is refit on the full 16,315,867-row training split, "
  "except the random forest, which is capped at five million rows for the "
  "final fit alone (a disclosed deviation: sklearn's forest has no GPU "
  "path and gains little past a few million rows). PM100's training split "
  "is already smaller than that sample would be, so it is used in full. "
  "The naive per-user-median baseline from Section 5.1 carries over as the "
  "floor.")
p("On F-DATA execution time, with a five-job rolling mean of each user's "
  "recent durations as the one historical feature, all three models clear "
  "both analytical baselines on R² by a wide margin. LightGBM leads at "
  "0.824, XGBoost follows at 0.815, and the random forest trails at 0.771, "
  "against the naive baseline's 0.080 and the Roofline ceiling's -0.164 "
  "from Section 5.1. The random forest reaching only 0.771 despite roughly "
  "ten times the tuning wall-clock time of either boosted library is worth "
  "stating plainly: the extra search budget did not buy a better model "
  "here. MAPE tells a different story, and the disagreement is real rather "
  "than a reporting quirk. Roofline's MAPE of 99.98 percent beats all "
  "three ML models, which land between 1,888 and 2,133 percent, even "
  "though Roofline loses badly on R². MAPE is dominated by the many short "
  "jobs where a small absolute miss is a large percentage; R² and RMSE are "
  "dominated by the few very long jobs. Neither metric is wrong, and this "
  "thesis reports both rather than choosing whichever one flatters a given "
  "model.")
p("Widening that historical feature helps. Adding 20-job and 50-job "
  "rolling means alongside the 5-job one and refitting at the same tuned "
  "hyperparameters lifts every model on every metric: XGBoost to 0.836, "
  "LightGBM to 0.831, the random forest to 0.821, with the share of "
  "predictions landing within 20 percent of actual rising by several "
  "points to around 61 percent. The best model flips from LightGBM to "
  "XGBoost in the process. Because this refit reused hyperparameters tuned "
  "with only the 5-job feature present, the gain is a lower bound on what "
  "a full re-tune would give.")
p("On PM100 power the three models sit close together, XGBoost and "
  "LightGBM both at an R² of 0.924 and the random forest at 0.902, against "
  "the naive baseline's 0.096 and the calibrated power model's 0.910 from "
  "Section 5.1. The margin over the analytical baseline is real but "
  "narrow, unlike the F-DATA duration case; XGBoost's MAPE of 16.6 percent "
  "against the calibrated model's 21.8 percent is the clearer separation. "
  "Two findings from this run matter beyond the headline numbers. The "
  "single feature carrying the most weight in XGBoost is the requested "
  "node count, at 61 percent of total signal by SHAP and 70 percent by "
  "split gain, which is the same quantity the Section 5.1 calibrated model "
  "was hand-built around when it scaled its idle-power term by node count. "
  "A hand-derived physical form and a data-driven tree ensemble "
  "independently identified the same dominant driver. Separately, an "
  "XGBoost model stripped to just the four resource-request fields scores "
  "an R² of 0.901 and a MAPE of 20.3 percent, a near-tie with the "
  "calibrated model on equivalent inputs. The full model's edge therefore "
  "comes from the broader Tier A feature set carrying information the "
  "four-field analytical form was never scoped to use, not from trees "
  "fitting the request-to-power relationship better than a calibrated "
  "linear form does.")
p("Across these two targets the best-performing model is already "
  "configuration-dependent: LightGBM for F-DATA duration with the single "
  "rolling feature, XGBoost once the wider windows are added, XGBoost for "
  "PM100 power. That is worth carrying forward rather than settling on one "
  "library as the winner.")

h2("5.4 Classical Machine-Learning Baselines: Memory, and What mem_req Was Doing")
p("Notebook 05b applies the same structure to memory: F-DATA's mmszu "
  "(memory actually used) and PM100's mem_alloc (allocated memory, the "
  "only memory field PM100 records). F-DATA's target has a documented "
  "fallback to allocated memory for rows where used memory is missing, but "
  "used memory has no missing values at all across the 23.3 million "
  "completed jobs, so the fallback never fires in practice. Both memory "
  "targets were fit with a real 50-trial Optuna sweep per model, no reused "
  "hyperparameters. A five-job rolling window won for both, unlike "
  "execution time, where the wider windows helped.")
p("On F-DATA memory the ranking is LightGBM at an R² of 0.867, the random "
  "forest at 0.825, XGBoost at 0.807. The naive per-user-median baseline "
  "scores -0.249 here, worse than simply predicting the global mean, which "
  "is the only target in this thesis where the per-user median does that "
  "badly. The historical rolling-mean feature accounts for roughly 88 "
  "percent of LightGBM's gain over that floor.")
p("PM100 memory needed a longer look, because the first run of it looked "
  "wrong in a specific way. With all 17 features present, the random "
  "forest's median absolute error was exactly zero and LightGBM's R² "
  "reached 0.985. Numbers that clean usually mean a feature is standing in "
  "for the target rather than predicting it, and there was an obvious "
  "suspect: mem_req, the requested memory, which on a batch-scheduled "
  "system is often granted exactly as asked. This is the same concern "
  "raised earlier about requested-versus-allocated fields on PM100 power, "
  "but sharper, because here the suspect feature could be a copy of the "
  "target itself rather than of another feature.")
p("A direct check settled it. Requested memory equals allocated memory "
  "exactly on 93.76 percent of test jobs. On the other 6.24 percent, "
  "allocation always exceeds the request, by a median of 230 megabytes and "
  "a tail out to 58,900. A model that does nothing but echo mem_req to its "
  "output scores an R² of 0.705 with no training at all, so most of the "
  "17-feature headline was mechanical rather than learned.")
p("The stripped-feature ablation I had set up to catch exactly this was "
  "itself compromised: mem_req was one of its four fields. Rebuilding it "
  "cleanly, with requested nodes, requested GPUs, requested cores, and the "
  "user's rolling mean of past allocations, mem_req removed entirely, "
  "scored an R² of 0.968 with XGBoost. That beats both the contaminated "
  "ablation and the full 17-feature model. So mem_req was doing more than "
  "failing to help on the hard cases; it was degrading the fit across the "
  "board.")
p("Adding the other twelve production features back on top of the clean "
  "four made accuracy worse, not better, with median absolute error rising "
  "from under one megabyte to about 69. XGBoost's column subsampling keeps "
  "drawing weak columns into each tree, which dilutes the one feature that "
  "carries real signal. The usable signal on this target sits in exactly "
  "four features.")
p("A full 50-trial search specific to the four-feature set moved the "
  "result barely at all, from 0.968 to 0.966, so the clean-four advantage "
  "is not a tuning artifact. The search surfaced one more thing: the "
  "validation metric used during tuning had itself been fooled by "
  "mem_req's mechanical echo. The 17-feature model scored a much better "
  "tuning-validation error yet did worse on the real test set, and worst "
  "of all on the divergent jobs.")
p("The contaminated-versus-clean gap held across all three models in the "
  "four-field comparison, with divergent-case R² around 0.73 contaminated "
  "against 0.95 clean. A supplementary check with all 17 features found a "
  "genuine per-model split, though. The random forest and LightGBM partly "
  "recover on the divergent jobs when given the full feature set to lean "
  "on; XGBoost does not, and its share of divergent-job predictions within "
  "20 percent of actual collapses to 17.9 percent.")
p("All three models were set to the clean four-field set, mem_req dropped "
  "entirely. For XGBoost the case is straightforward. For the random "
  "forest and LightGBM it is a deliberate trade: the full feature set "
  "gives a higher aggregate R² (0.977 and 0.985 against the clean set's "
  "0.951 and 0.960), but the clean set gives far more reliable accuracy on "
  "the roughly 6 percent of jobs where request and allocation diverge, "
  "with median absolute error under one megabyte against 35 to 44. Those "
  "divergent jobs are exactly what a genuine submission-time predictor is "
  "for, so that is where accuracy was prioritised. On the adopted "
  "clean-four set, XGBoost is best at an R² of 0.966, LightGBM at 0.960, "
  "the random forest at 0.951; the 17-feature set's best had been LightGBM "
  "at 0.985.")
p("This is the fourth target-and-dataset pairing in this thesis where the "
  "best model changed with the configuration: LightGBM then XGBoost on "
  "F-DATA execution time, XGBoost on PM100 power, LightGBM on the "
  "17-feature PM100 memory set, XGBoost on the clean-four set. Best model "
  "is a property of the configuration here, not a fixed choice the "
  "pipeline can settle once.")

doc.save(OUT_PATH)
print("saved to", OUT_PATH)
