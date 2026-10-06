GelGenie Python
==============================

This is the CLI and Python version of GelGenie, which allows for model training, analysis and quick inference.

## Installation

**Thesis development version:** clone the [project fork](https://github.com/gerry019/GelGenie) instead of the original repository (`git clone https://github.com/gerry019/GelGenie.git`). This version was tested with Python 3.11 and 3.12.

To install the GelGenie python package and its dependencies in development mode:

- Create a new python environment (Conda or otherwise, python 3.7+),
- Install PyTorch using the correct command for your OS and hardware (GPU or otherwise) from here: https://pytorch.org/get-started/locally/
- Install dependencies from `./python-gelgenie/requirements.txt` (pip or conda should both work)
- clone this repository to your local machine (`git clone https://github.com/mattaq31/GelGenie.git`)
- Finally, run the following command from the `./python-gelgenie` directory:

`pip install -e .`

- The above process will allow you to import the gelgenie functions anywhere, as well as access the CLI commands.  Any changes you make to the code will be immediately reflected in the package.

## Code Organization
The root directory for all python code is `python-gelgenie`.  Within this directory, the code is organized as follows:
- `gelgenie` contains the main package code, including the model architectures, training and evaluation scripts, and the CLI commands.
- `prototype_frontend` contains an initial draft setup for the GelGenie GUI, built using Electron.  We eventually discarded this prototype in favour of the QuPath extension, but it should still be functional.  Unfortunately, only classical tools are available with this prototype.
- `paper_figure_generation` contains the entire analysis code (a mixture of Python scripts and Jupyter notebooks) used to generate the results and plots in the main GelGenie paper. This code is not intended for general use, but is provided for transparency and reproducibility.  The code here makes use of the functions within the `gelgenie` package. 

The `gelgenie` package is organized as follows:

- `classical_tools` contains a suite of functions for using watershed or multiotsu thresholding for segmentation.  The main file containing the most important functions is `watershed_segmentation.py`.  The other files contain various early implementations of a band detection system that are now unused by the main codebase.
- `segmentation` contains the main deep learning architectures and training systems.   Functionality is split up into folders as follows:
    - `data_handling`: Contains functions for preparing and augmenting datasets, as well as the dataset classes themselves.
    - `networks`:  Contains the main U-Net architectures.  Most networks are imported from dependencies with minor tweaks, but the setup allows for the easy addition of new custom architectures.
              - **Thesis development version:** adds the DAUNet architecture (`networks/DAUNet`).
    - `training`: Contains functions for training models, including the main training loop and the training configuration class.  Also contains several example configuration files.
              - **Thesis development version:** adds three-class training support and TPU training via PyTorch/XLA (in `core_training.py`).
    - `evaluation`: Contains functions for model inference and evaluation, along with comparisons with classical methods.
              - **Thesis development version:** adds the well-guided and well-independent post-segmentation analysis pipelines (`quantitative_analysis_well_guided.py` and `quantitative_analysis_well_independent.py`).
    - `helper_functions`: Contains a variety of helper functions used throughout the package such as loss functions, converters and model exporters.
    - `checkpoint_handling` *(thesis development version)*: Contains scripts for inspecting a two-class checkpoint and converting it into a three-class model for transfer learning.
    - Other: `EDDIE_scripts` contains scripts for running the code on the EDDIE cluster, `notebook_testing_and_evaluation` contains scratch-style Jupyter notebooks for testing and evaluating the code, and `nnunet_scripting_analysis` contains scripts for preparing datasets for nnunet, converting results and exporting the nnunet model to torchscript format.

## Training a Model

Training a model involves several steps, including preparing the dataset, setting up the training configuration and running the training loop.  

### Preparing Data

Preparing data for training is simple - you simply need to create two folders, one with the original gel images and another with the corresponding ground-truth segmentation masks.  Each image and mask should have the same name, but the mask should always be a .tif file (the image can have any extension).  Segmentation masks can be generated using QuPath or using the GelGenie extension (see [here](../qupath-gelgenie/README.md)) or any other method you prefer.  For segmentation mask formatting details, check out our dataset on Zenodo: [10.5281/zenodo.13218469](https://doi.org/10.5281/zenodo.13218469).  Further settings are available for more complex dataset setups (see below).

**Thesis development version:** for three-class segmentation, mask pixel values should represent background (0), bands (1) and wells (2).

Once the dataset is prepared, you will need to split it into a training, validation and test set partition.  You can do this manually, adapt our `data_handling/data_split.py` function or generate the split on the fly using the `split_training_dataset` flag (only available for a single input folder currently).  If you use our Zenodo datasets, the data has already been split for training.

### Setting up Training Config

There are a large number of different configuration options available for training.  The easiest way to supply these options and keep track of your selections is to use .toml configuration files.  An example config file used for the GelGenie paper is available at `training/config_files/final_paper_configs/base_unet_training_only.toml`.

**Thesis development version:** for three-class masks, set `classes` to 3 in the model settings; the number of classes used for data loading is then set automatically. Where class-specific loss weighting is used, a weight should be defined for each of the three classes. To train on TPU hardware, set `device` to `"tpu"` in the processing settings (requires PyTorch/XLA); if a TPU is not available, training falls back to GPU or CPU.

For a full explanation of all the config options, see the docstrings in `routine_training.py` or refer to the example above.  

### Running the Training Loop

The main class (`TrainingHandler`) that takes care of training is located at `training/core_training.py`.  The training loop is effectively organised by instantiating the handler and then running the `full_training()` method.  The code itself contains a full description of how supplied options are used and the training system is setup.  In brief:
- The dataset is loaded in and tested for compatibility with the system.
- The model in instantiated and the optimizer and scheduler are setup.  If training is being resumed, the specified model checkpoint is loaded.
- The `wandb` tracking/visualization system is setup to allow for real-time tracking (if enabled).  You will need to create your own account and setup the API key to use this system.  TODO: currently the settings are hard-coded for the Dunn group account.  Will need to make this user-adjustable.
- Once everything is ready, the training loop is started, which consists of iterating over the training and validation sets, calculating the loss and backpropagating the gradients.  Model checkpoints are saved throughout training, according to the settings supplied.  Automatic cleaning of old checkpoints is also possible (see the `model_cleanup_frequency` parameter for details).
- **Thesis development version:** the `wandb` project and entity are set to the account used for this work. Either set `wandb_track` to `false`, or update the project and entity names in `core_training.py`. Training was tested with `wandb` v0.26.1; if compatibility issues occur with a newer version, install the tested version using `pip install wandb==0.26.1`.
- 
Full usage of the training system can be evaluated from the `routine_training.py` script.  The training loop can also be run from the CLI using the `gelseg_train` command (see below).

## Evaluating a Model
During training, the model will be tested on the validation set several times.  However, to run ad-hoc inference on a pre-trained model you can use the evaluation functions directly in a separate script (or the CLI function - see below).  An example of how to use these functions is as follows:

```python
from os.path import join
from gelgenie.segmentation.evaluation.core_functions import segment_and_plot, segment_and_quantitate
from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty
from gelgenie.segmentation.evaluation import model_eval_load

# replace all file paths with your own
output_folder = '/Users/matt/Desktop/scratch_test'
model_folder = 'Automatic_Gel_Analyzer/segmentation_models/December 2023'
input_folder = ['Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/test_images']
mask_folder = ['Automatic_Gel_Analyzer/data/processed_gels/maximal_set/matthew_gels/test_masks'] # if no masks are available, use the segment_and_plot function instead of segment_and_quantitate

run_quant_analysis = True # runs a full quantitation analysis on the input if set to True
classical_analysis = False # set to true to also run watershed/multi-Otsu segmentation on the input
multi_augment = False  # set to true to perform test-time augmentation

model_and_epoch = [('unet_dec_21_finetune', '590')]

experiment_names, eval_epochs = zip(*model_and_epoch)

models = []

for experiment, eval_epoch in zip(experiment_names, eval_epochs):
    exp_folder = join(model_folder, experiment)
    model = model_eval_load(exp_folder, eval_epoch)
    models.append(model)

create_dir_if_empty(output_folder)

if run_quant_analysis: # runs evaluation and generates metrics
    segment_and_quantitate(models, list(experiment_names), input_folder, mask_folder, output_folder,
                           multi_augment=multi_augment, run_classical_techniques=classical_analysis)
else: # just runs model inference
    segment_and_plot(models, list(experiment_names), input_folder, output_folder, multi_augment=multi_augment,
                     run_classical_techniques=classical_analysis)
```

**Thesis development version:** the same functions (and `quick_seg`) support both two-class and three-class models. For three-class models, predicted wells are shown alongside bands in the output visualisations, and a well Dice score is reported when ground-truth masks are supplied. In addition to the original outputs, each model's output folder now also contains, for every image:
- `_raw_mask.tif`: the predicted segmentation mask (used as input to the post-segmentation pipelines)
- `_confidence_map.tif`: the per-pixel prediction confidence
- `_confidence_viridis.png` and `_confidence_overlay.png`: visualisations of the confidence map, on its own and overlaid on the gel
- `_annotated.png`: the segmentation result annotated with its evaluation metrics (when quantitative analysis is run)

Additional preprocessing options are available through the `minmax_norm`, `percentile_norm` and `invert_images` arguments.

## Post-Segmentation Analysis (Thesis Development Version)

Two post-segmentation analysis pipelines are provided in `gelgenie/segmentation/evaluation`:

- `quantitative_analysis_well_guided.py` uses predicted well and band positions to organise bands into lanes. It requires three-class masks (background = 0, bands = 1, wells = 2).
- `quantitative_analysis_well_independent.py` organises bands into lanes from band positions alone. It accepts both two-class and three-class masks (wells are ignored).

Both pipelines calculate migration distances and molecular size or mass estimates, and can be run from the `gelgenie/segmentation/evaluation` directory:

```
python quantitative_analysis_well_guided.py --masks_folder path/to/masks --output_folder path/to/results
python quantitative_analysis_well_independent.py --input_folder path/to/masks --output_folder path/to/results
```

Confidence maps are used automatically when available, provided their filenames match the masks with `_raw_mask.tif` replaced by `_confidence_map.tif`. Area and confidence filtering are applied by default and can be disabled with `--no_area_filter` and `--no_confidence_filter`. A different mask filename pattern can be supplied with `--mask_pattern`, although confidence maps are then not detected automatically.

By default, the lane with the most bands is proposed as the ladder and the user is prompted to confirm it and enter the ladder sizes. Prompts can be disabled with `--non_interactive`, in which case the ladder sizes must be supplied with `--ladder_sizes` (e.g. `--ladder_sizes "1000,750,500,250"`).

Each successfully analysed gel produces a visualisation, a text report and a migration-measurement CSV, together with run logs and a summary log.
## CLI Commands

These commands can be run from anywhere using a command line terminal.  They directly call the python functions in the `gelgenie` package and are automatically updated with the latest version of the code if installed using the instructions above.

- `gelseg_train`: Main function for training a segmentation model.  Full options and parameters available by running `gelseg_train --help`.
- `quick_seg`: Quick segmentation of a single image or a folder of images using pre-trained model(s).  Full options and parameters available by running `quick_seg --help`.
    - **Thesis development version:** the `--map_colour` option has been replaced by `--band_colour` and `--well_colour`, so bands and wells can be coloured separately.
- `export_model`: Main function for converting a PyTorch model checkpoint into either an onnx or torchscript format model for use by the GelGenie QuPath extension.  Full options and parameters available by running `export_model --help`.
    - **Thesis development version:** ONNX export requires additional packages, which can be installed using `pip install onnx onnxsim onnxscript`.
- `pull_model`: Helper function to extract model logs and checkpoints from the EDDIE server (assuming you have already setup ssh keypairing with your account).  Could potentially be translated for other servers too.  Full options and parameters available by running `pull_model --help`.
- `gen_eddie_qsub`: Helper function to generate a batch file for the EDDIE server (could also be translated for other servers).  Full options and parameters available by running `gen_eddie_qsub --help`.

## Developing new Models and Features

New contributions to improve or add additional functionality is welcome!  Some ideas on things to improve could include:

- Unit tests to ensure new changes do not break any existing functionality
- New architectures or loss functions to improve segmentation performance
- New data augmentation techniques to improve model generalization
- New evaluation metrics or visualizations to better understand model performance
- Expanded documentation for new developers
- Some way to export models to onnx with dynamic input sizes for OpenCV
