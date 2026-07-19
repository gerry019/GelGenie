# AGENTS.md — GelGenie

Orientation guide for AI agents (and humans) working in this repository. Read this first.

## Overview

**GelGenie** is an AI-based system for gel-electrophoresis image analysis: it segments gel images
with deep learning models and automates the downstream band-quantitation workflow. The project has
two independent components:

- **`python-gelgenie/`** — a PyTorch package for preparing datasets, building segmentation
  architectures, and training / evaluating / running models.
- **`qupath-gelgenie/`** — a [QuPath](https://qupath.github.io) extension (Java + JavaFX) that gives
  end users a one-click GUI for running GelGenie models and analysing/exporting the results.

> **⚠️ The `qupath/` directory at the repo root is NOT part of GelGenie.** It is an untracked local
> clone of the upstream QuPath application (currently v0.7.0), kept only as a reference for QuPath's own build,
> CI, and API patterns. **Never edit files under `qupath/`** — it is not committed to this repo
> (`git ls-files qupath/` returns nothing).

### Current development context

This checkout is a **fork** (`origin = gerry019/GelGenie`, Gertrude's fork). Since the last upstream
commit by Matthew (`23fae2f`, Jul 2025), Gertrude extended GelGenie to **3-class segmentation**
(background / bands / **wells**) so that lanes can eventually be anchored to gel wells. That work is
functional but partly incomplete on the Java side and carries some leftover debug code — see
[Known issues](#known-issues) and `ROADMAP.md`.

## Repository layout

```
├── python-gelgenie/     # PyTorch training/inference package (see below)
├── qupath-gelgenie/     # QuPath Java extension (see below)
├── catalog.json         # QuPath extension catalog (points Extension Manager at release jars)
├── logo/                # Branding assets
├── README.md            # User-facing project README
├── ROADMAP.md           # Phased plan for upcoming work + git integration strategy
└── qupath/              # ⚠️ Untracked upstream QuPath clone — reference only, do not edit
```

## Python component (`python-gelgenie/`)

Package root is `python-gelgenie/gelgenie/`, split into:

- **`classical_tools/`** — non-ML segmentation (watershed / multi-Otsu), mostly legacy.
- **`segmentation/`** — the deep-learning core:
  - `networks/` — U-Net / MONAI model gateways (`model_configure`).
  - `data_handling/` — datasets (`ImageDataset`, `ImageMaskDataset`), augmentations, dataloader prep.
  - `training/` — `core_training.py` (`TrainingHandler`), setup helpers, and a large
    `config_files/` tree of TOML configs.
  - `evaluation/` — inference/eval (`core_functions.py`), the standalone `gel_analysis.py` lane
    pipeline, and the `quick_seg` CLI.
  - `helper_functions/`, `checkpoint_handling/` (2→3-class transfer-learning scripts),
    `bioimage_io_handling/`, `nnunet_scripting_analysis/`.

Sibling non-package dirs (`paper_figure_generation/`, `other_visualizations/`, `prototype_frontend/`)
are reproducibility/scratch material, not part of the installed package.

### Install & run

```bash
# Install PyTorch first per https://pytorch.org/get-started/locally/ (OS/GPU-specific), then:
pip install -e python-gelgenie
```

CLI entry points (defined in `setup.py`):

| Command          | Purpose                                    |
|------------------|--------------------------------------------|
| `gelseg_train`   | Train a segmentation model                 |
| `quick_seg`      | Run inference / evaluation on a folder     |
| `pull_model`     | Fetch model/server data                    |
| `export_model`   | Export a model (e.g. to TorchScript)       |
| `gen_eddie_qsub` | Generate an EDDIE-cluster batch script     |

**Training is TOML-config-driven.** Configs live under `segmentation/training/config_files/`.
Parameters are merged in priority order: **CLI flags > `--parameter_config` > `--user_default_config`
> `global_defaults.toml`**. Note the paper configs contain hardcoded EDDIE-cluster absolute paths.

## Java component (`qupath-gelgenie/`)

A standalone Gradle project (its own wrapper) implementing a QuPath extension. Key classes:

- **`GelGenie.java`** — extension entry point; registers the `Extensions > GelGenie` menu.
- **`ui/UIController.java`** (+ `resources/.../ui/gelgenie_control.fxml`) — the main control panel:
  model/device selection, inference, background correction, band visualisation & editing toggles,
  embedded bar chart.
- **`ui/TableController.java`** (+ `gelgenie_table.fxml`) — the data table; `computeTableColumns()`
  does the band **volume quantitation** (raw + global/local/rolling-ball background-corrected, plus
  normalisation). `ui/BandEntry.java` is its row model.
- **`models/`** — `ModelInterfacing` (downloads the HuggingFace model registry), `GelGenieModel`
  (per-model descriptor, incl. `num_classes`), `ModelRunner` (inference: an **OpenCV/ONNX** path and
  a **DJL/TorchScript** path).
- **`djl_processing/`** — Deep Java Library translators & transforms (`GelSegmentationTranslator`,
  `NnUNetSegmentationTranslator`, `ChannelSquisher`, `ImageInvert`, `DivisibleSizePad`, `MpsSupport`).
- **`tools/`** — `SegmentationMap` (export label images), `BandSorter` (group bands into lanes via
  X-overlap sweep, assign `LaneID`/`BandID`), comparators `CentroidCompareX` / `LaneBandCompare`,
  `ImageTools` (pixel extraction).
- **`graphics/EmbeddedBarChart.java`** — band-volume bar charts & intensity histograms.

### Build

```bash
cd qupath-gelgenie
./gradlew build        # produces build/libs/qupath-gelgenie-<version>.jar (shadow/fat jar)
```

- Targets **QuPath 0.7.0** (`settings.gradle.kts`), via plugin
  `io.github.qupath.qupath-extension-settings` v0.2.1 and `com.gradleup.shadow` 8.3.5.
- Extension coordinates: `group = io.github.mattaq31`, `version = 1.1.0`.

### Distribution

Build jar → attach to a GitHub **Release** tagged `vX.Y.Z` → add/update the matching entry in the
root **`catalog.json`**, which QuPath's Extension Manager reads to install/update the extension.

## 3-class model status (bands / wells / background)

Class indices: **0 = background, 1 = Gel Band, 2 = Well**. Gertrude threaded a `num_classes`
parameter through both components (models default to 2 for backward compatibility).

- **Python:** dataloaders, augmentations, training metrics (per-class Dice), visualisation, and the
  `gel_analysis.py` lane pipeline support 3 classes.
- **Java — works:** the **DJL/TorchScript** inference path splits the output into separate
  **"Gel Band"** and **"Well"** annotation classes (distinct colours; `BandID` vs `WellID`).
- **Java — does NOT yet work:** the **OpenCV/ONNX** path is still hardcoded to 2 classes; and wells
  never reach the **data table, charts, or export** (all band-only). Treat wells as "created but
  downstream-dead" until the roadmap work lands.

## Conventions

- **Python:** Apache-2.0 license header at the top of source files; TOML configs for training.
- **Java:** package root `qupath.ext.gelgenie`; UI strings in `resources/.../ui/strings.properties`
  (i18n); FXML for layout; extension registered via
  `META-INF/services/qupath.lib.gui.extensions.QuPathExtension`.
- **Git / authorship:** this is a multi-author project (Matthew + Gertrude + others). When
  integrating cross-author branches upstream, **use a regular merge, not squash**, so per-commit
  authorship is preserved. See `ROADMAP.md`.

## Known issues (confirmed; don't rediscover)

**Python**
- No `pyproject.toml`; `setup.py` has **no `install_requires`** and version `1.0.0` (mismatches the
  extension's `1.1.0`).
- `requirements.txt`: `opencv>=4.5.1` is not a valid PyPI name (should be `opencv-python`); missing
  `torch`, `torchvision`, `tifffile`, `scikit-learn`.
- `segmentation/training/core_training.py`: left-in `DEBUG:` `rprint(...)` + `sys.stdout.flush()`
  lines (e.g. 436, 452, 468, 543, 553); W&B hardcoded to `entity='gertrude-university-of-malta'`,
  `project='Wells'` (lines 144/148).
- `segmentation/evaluation/gel_analysis.py:257`: blocking `input()` for ladder sizes — hangs batch
  runs. Lane clustering is **one-well-per-lane**, not DBSCAN.
- `segmentation/checkpoint_handling/*.py`: hardcoded Colab `sys.path`/paths; no `__init__.py`.
- `segmentation/evaluation/core_functions.py` `segment_and_plot`: hardcodes band/well colours,
  ignoring the passed arguments.
- `gelgenie_catalog_creation.py:24`: writes to a hardcoded `/Users/matt/Desktop/catalog.json`.

**Java**
- `GelGenie.java:53`: leftover `System.err.println("GelGenie.installExtension HIT")` debug print.
- `models/ModelRunner.java` OpenCV path (~194-262): hardcoded 2-class; 3-class models only work via
  DJL.
- `ui/TableController.java:470`: band-only filter — wells excluded from quantitation/table.
- `tools/LaneBandCompare.java`: sorts on `BandID` only; doesn't understand `WellID`.
- `num_classes` relies on the registry JSON field with no validation/UI surface (silently 2).
- `settings.gradle.kts` 0.6.0→0.7.0 bump is currently **uncommitted**; `catalog.json` still declares
  `version_range.min: v0.6.0`.
