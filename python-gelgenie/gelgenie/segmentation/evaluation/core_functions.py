"""
 * Copyright 2024 University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

from gelgenie.classical_tools.watershed_segmentation import watershed_analysis, multiotsu_analysis
from gelgenie.segmentation.data_handling.dataloaders import ImageDataset, ImageMaskDataset
from gelgenie.segmentation.helper_functions.general_functions import create_dir_if_empty, index_converter
from gelgenie.segmentation.helper_functions.dice_score import multiclass_dice_coeff
from gelgenie.segmentation.evaluation.gel_analysis import analyze_gel_with_proper_well_centric_approach

import os
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy import ndimage as ndi
from skimage.color import label2rgb
from scipy.spatial.distance import directed_hausdorff
from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops_table
from matplotlib.patches import Patch
from tqdm import tqdm
import math
import imageio
import numpy as np
import itertools
from collections import defaultdict
import pandas as pd
from sklearn.metrics import confusion_matrix
from PIL import Image



# location of reference data, to be imported if required in other files
ref_data_folder = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)),
                               'data_analysis', 'ref_data')


def model_predict_and_process(model, image):
    """
    Runs the provided segmentation model and pre-processes it into an ordered mask for subsequent labelling.
    Computes a per-pixel confidence map (max softmax probability)
    :param model: Pytorch segmentation model
    :param image: Input image (torch tensor)
    :return:
    """
    with torch.no_grad():
        mask = model(image)
        num_classes = mask.shape[1] # Updated to fit multiclass segmentation 
        probs = F.softmax(mask, dim=1) # Get softmax probability across the classes dimension
        # Get the maximum value per pixel, no index 
        conf_map = probs.max(dim=1)[0].squeeze().cpu().numpy() # Numpy conversion for saving
        one_hot = F.one_hot(mask.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
        ordered_mask = one_hot.numpy().squeeze()
    return mask, ordered_mask, conf_map


def model_multi_augment_predict_and_process(model, image):
    """
    This function runs the provided model multiple times on augmented versions of the input, then combines the
    outputs into one averaged estimate.
    :param model: Pytorch segmentation model
    :param image: gel image in N,C,H,W format
    :return: direct pytorch mask and ordered numpy mask for easy use
    """
    mirror_axes = [0, 1]

    with torch.no_grad():
        mask = model(image)
        axes_combinations = [c for i in range(len(mirror_axes)) for c in
                             itertools.combinations([m + 2 for m in mirror_axes], i + 1)]
        for axes in axes_combinations:
            mask += torch.flip(model(torch.flip(image, axes)), axes)
        mask /= (len(axes_combinations) + 1)

    num_classes = mask.shape[1]  # For 3 class segmentation
    probs = F.softmax(mask, dim=1) # softmax with normalization for classes
    conf_map = probs.max(dim=1)[0].squeeze().cpu().numpy() 
    one_hot = F.one_hot(mask.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
    ordered_mask = one_hot.numpy().squeeze()

    return mask, ordered_mask, conf_map


def save_model_output(output_folder, model_name, image_name, labelled_image):
    """
    Saves labelled model output to file.
    :param output_folder: Output folder location
    :param model_name: Name of model
    :param image_name: Image name
    :param labelled_image: Image with RGB segmentation labels painted on top
    :return: N/A
    """
    imageio.v2.imwrite(os.path.join(output_folder, model_name, '%s.png' % image_name), (labelled_image * 255).astype(np.uint8))


def save_segmentation_map(output_folder, model_name, image_name, segmentation_map, confidence_map=None, band_colour=(163, 106, 13), well_colour= (0, 255,0)):

    if len(segmentation_map.shape) == 3:
        segmentation_map = segmentation_map.argmax(axis=0)

    # Saving the mask produced with classes (argmax)
    raw_mask_path = os.path.join(output_folder, model_name, f'{image_name}_raw_mask.tif')
    tiff.imwrite(raw_mask_path, segmentation_map.astype(np.uint8))

    # Save confidence map if provided
    if confidence_map is not None:
        conf_path = os.path.join(output_folder, model_name, f'{image_name}_confidence_map.tif')
        tiff.imwrite(conf_path, confidence_map.astype(np.float32)) # For continous numbers
    
        # Updated version per pixel 
        plt.figure(figsize=(8, 6))
        plt.imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Confidence')
        plt.title(f'Confidence Map: {image_name}', fontsize=14)
        plt.axis('off')
        conf_viridis_path = os.path.join(output_folder, model_name, f'{image_name}_confidence_viridis.png')
        plt.savefig(conf_viridis_path, dpi=300, bbox_inches='tight')
        plt.close()
    
        # To visualise the confidence map ( per well/band)
        conf_rgb = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) 

        # Loop through bands and wells
        for cls_val in [1, 2]:  # 1=bands, 2=wells
            mask_cls = (segmentation_map == cls_val)
            if not np.any(mask_cls): # safety check
                continue
            # Label connected components within this class
            labeled = label(mask_cls)

            # The mean confidence for each labelled region (well/band)
            props = regionprops_table(
                labeled,
                intensity_image=confidence_map,
                properties=("label", "mean_intensity")
            )

            # Colors based on mean confidence
            for region_label, mean_val in zip(props["label"], props["mean_intensity"]):
                if mean_val > 0.8:
                    color = (0, 255, 0)      # green = high confidence
                elif mean_val > 0.5:
                    color = (255, 255, 0)    # yellow = medium confidence
                else:
                    color = (255, 0, 0)      # red = low confidence
                conf_rgb[labeled == region_label] = color

        # Create figure with confidence overlay and legend
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(conf_rgb)
        ax.axis('off')
        ax.set_title(f'Confidence Map: {image_name}', fontsize=14, pad=10)

        # Create legend patches
        legend_elements = [
            Patch(facecolor='green', label='High confidence (>80%)'),
            Patch(facecolor='yellow', label='Medium confidence (50-80%)'),
            Patch(facecolor='red', label='Low confidence (<50%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10)
        
        plt.tight_layout() # spacing adjustment

        # Saving the figure
        conf_overlay_path = os.path.join(output_folder, model_name, f'{image_name}_confidence_overlay.png')
        plt.savefig(conf_overlay_path, dpi=300, bbox_inches='tight')
        plt.close(fig)        

    rgba_array = np.ones((segmentation_map.shape[0], segmentation_map.shape[1], 4), dtype=np.uint8)*255

    # negative pixels should have no alpha and no colour
    alpha_channel = np.where(segmentation_map > 0, 255, 0)
    rgba_array[:, :, 3] = alpha_channel

    # Set RGB channels - updated to use different colour for bands and well 
    rgba_array[:, :, :3] = np.where(
        (segmentation_map == 1)[..., None],  # Bands
        band_colour,
        np.where(
            (segmentation_map == 2)[..., None],  # Wells
            well_colour,
            0  # Background
        )
    )


    # Create PIL Image from RGBA array
    image = Image.fromarray(rgba_array, 'RGBA')

    image.save(os.path.join(output_folder, model_name, '%s_map_only.png' % image_name))

    return image

def save_annotated_output(output_folder, model_name, image_name, rgb_labels, metrics_dict):
    """
    Saves  every segmentation output (image) with a small text box showing the primary performance metrics.
    Only outputted for ML models (not classical methods).
    """
    # Pull latest metric values for this image (according to the model being tested, as multiple models can be run at once)
    band_dice      = metrics_dict["Band Dice Score"][image_name][-1]
    well_dice      = metrics_dict["Well Dice Score"][image_name][-1]
    fg_f1          = metrics_dict["Foreground F1"][image_name][-1]
    precision_fg   = metrics_dict["Precision"][image_name][-1]
    recall_fg      = metrics_dict["Recall"][image_name][-1]
    
    # Prepare tiny text block
    textstr = (
        f"Model: {model_name}\n"
        f"Band Dice: {band_dice:.2f}\n"
        f"Well Dice: {well_dice:.2f}\n"
        f"FG F1: {fg_f1:.2f}\n"
        f"Prec: {precision_fg:.2f}\n"
        f"Rec: {recall_fg:.2f}"
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb_labels) # Same image used afterwards
    ax.axis("off")
    
    # Style box is rounded rectangle, white, semi-transparent.
    props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="black")
    
    # Bottom-right corner 
    ax.text(
        0.98, 0.02, textstr,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props
    )
    
    save_path = os.path.join(output_folder, model_name, f"{image_name}_annotated.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    
def plot_model_comparison(model_outputs, model_names, image_name, raw_image, output_folder, images_per_row,
                          double_indexing, comments=None, title_length_cutoff=20):
    """
    Plots a comparison of input images and their segmentation results.
    :param model_outputs: Segmentation images for each input model
    :param model_names: Name of each model (in order)
    :param image_name: Name of test image
    :param raw_image: Raw un-segmented numpy image (for plotting)
    :param output_folder: Output folder to save results
    :param images_per_row: Number of images to plot per row in the output figure
    :param double_indexing: Whether or not double indexing is required for the plot axes
    :param comments: A string to add to the title of each model (if required)
    :return: N/A
    """
    # results preview

    rows = math.ceil((len(model_outputs) + 1) / images_per_row)
    fig, ax = plt.subplots(rows, images_per_row, figsize=(15, 15))

    for i in range(len(model_outputs) + 1, rows * images_per_row):
        ax[index_converter(i, images_per_row, double_indexing)].axis('off')  # turns off borders for unused panels

    zero_ax_index = index_converter(0, images_per_row, double_indexing)

    ax[zero_ax_index].imshow(raw_image, cmap='gray')
    ax[zero_ax_index].set_title('Reference Image')

    for index, (mask, name) in enumerate(zip(model_outputs, model_names)):
        plot_index = index_converter(index + 1, images_per_row, double_indexing)

        ax[plot_index].imshow(mask)
        title = name
        if comments:
            title += ' ' + comments[index]
        if len(title) > title_length_cutoff:
            title = title[:int(len(title) / 3)] + '\n' + title[int(len(title) / 3):int((2 * len(title)) / 3)] + '\n' + title[int((2 * len(title)) / 3):]
        elif len(name) > title_length_cutoff * 2:
            title = title[:int(len(title) / 2)] + '\n' + title[int(len(title) / 2):]
        else:
            title = title

        ax[plot_index].set_title(title, fontsize=13)

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.suptitle('Segmentation result for image %s' % image_name)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'method_comparison', '%s method comparison.png' % image_name), dpi=300)
    plt.close(fig)


def run_watershed(image, image_name, output_watershed):
    """
    Runs the multiotsu thresholding system on the input image and converts to torch format for loss computation
    :param image: Input image (numpy array)
    :param image_name: Image name (string)
    :param output_watershed: Watershed folder for saving intermediates directly
    :return: Torch mask for dice computation and numpy array of watershed labels
    """
    _, watershed_labels = watershed_analysis(image, image_name,
                                             intermediate_plot_folder=output_watershed,
                                             repetitions=1, background_jump=0.08,
                                             use_multiotsu=True)

    temp_array = watershed_labels.copy()
    temp_array[temp_array > 0] = 1
    # converts back to required format for dice score calculation
    torch_mask = F.one_hot(torch.tensor(temp_array).long().unsqueeze(0), 2).permute(0, 3, 1, 2).float()

    return torch_mask, watershed_labels


def run_multiotsu(image, image_name, output_otsu):
    """
    Runs the multiotsu thresholding system on the input image and converts to torch format for loss computation
    :param image: Input image (numpy array)
    :param image_name: Image name (string)
    :param output_otsu: Otsu folder for saving intermediates directly
    :return: Torch mask for dice computation and numpy array of otsu labels
    """
    otsu_labels = multiotsu_analysis(image, image_name, intermediate_plot_folder=output_otsu)

    otsu_labels[otsu_labels > 0] = 1

    # converts back to required format for dice score calculation
    torch_mask = F.one_hot(torch.tensor(otsu_labels).long().unsqueeze(0), 2).permute(0, 3, 1, 2).float()

    return torch_mask, otsu_labels


def read_nnunet_inference_from_file(nfile):
    # get image name, then convert into nnunet-compatible name
    # read image from file, convert to 1 channel segmentation format
    # pass on for dice score calculation and RGB labelling
    n_im = imageio.v2.imread(nfile)
    # Hardcoded for 3 class-segmentation
    torch_mask = F.one_hot(torch.tensor(n_im).long().unsqueeze(0), 3).permute(0, 3, 1, 2).float()

    return torch_mask, n_im


def segment_and_quantitate(models, model_names, input_folder, mask_folder, output_folder,
                           minmax_norm=False, percentile_norm=False, multi_augment=False, images_per_row=3,
                           run_classical_techniques=False, nnunet_models_and_folders=None,
                           band_colour=(163, 106, 13), well_colour=(0, 255, 0)):
    """

    Segments images in input_folder using the selected models and computes their Dice score versus the ground truth labels.
    :param models: Pre-loaded pytorch segmentation models
    :param model_names: Name for each model (list)
    :param input_folder: Input folder containing gel images
    :param mask_folder: Corresponding folder containing ground truth mask labels for loss computation
    :param output_folder: Output folder to save results
    :param minmax_norm: Set to true to min-max normalise images before segmentation
    :param percentile_norm: Set to true to percentile normalise images before segmentation
    :param multi_augment: Set to true to perform test-time augmentation
    :param images_per_row: Number of images to plot per row in the output comparison figure
    :param run_classical_techniques: Set to true to also run watershed and multiotsu segmentation apart from selected models
    :param nnunet_models_and_folders: List of tuples containing (model name, folder location) for pre-computed nnunet results on the same dataset
    :param map_pixel_colour: Colour to use for positive pixels in the output segmentation map (tuple, RGB)
    :return: N/A (all outputs saved to file)
    """
    dataset = ImageMaskDataset(input_folder, mask_folder, 1, padding=False, individual_padding=True,
                               minmax_norm=minmax_norm, percentile_norm=percentile_norm)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

    if run_classical_techniques:
        models.extend(['watershed', 'multiotsu'])
        model_names.extend(['watershed', 'multiotsu'])

    if nnunet_models_and_folders:
        nnunet_model_names, nnunet_eval_outputs = zip(*nnunet_models_and_folders)
        models.extend(nnunet_eval_outputs)
        model_names.extend(nnunet_model_names)
    else:
        nnunet_model_names = []

    for mname in model_names:
        create_dir_if_empty(os.path.join(output_folder, mname))

    create_dir_if_empty(os.path.join(output_folder, 'method_comparison'))
    create_dir_if_empty(os.path.join(output_folder, 'metrics'))
  


    double_indexing = True  # axes will have two indices rather than one
    if math.ceil((len(model_names) + 2) / images_per_row) == 1:  # axes will only have one index rather than 2
        double_indexing = False

    metrics_dict = {}
    # Metrics that go through the zip loop (10 items, applies to ALL models, including classical methods)
    metrics_for_zip = ['Foreground Dice Score', 'MultiClass Dice Score',
                      'True Negatives', 'False Positives', 'False Negatives', 'True Positives', 
                      'Precision', 'Recall', 'Foreground F1', 'Hausdorff Distance']

    # ALL metrics (for initialization and CSV output)
    all_metrics = metrics_for_zip + [
          'Band Dice Score', 'Well Dice Score',
          'Band TP', 'Band FP', 'Band FN', 'Band TN',
          'Band Precision', 'Band Recall', 'Band F1',
          'Well TP', 'Well FP', 'Well FN', 'Well TN',
          'Well Precision', 'Well Recall', 'Well F1']


    for metric in all_metrics:
        metrics_dict[metric] = defaultdict(list)

    # preparing model outputs, including separation of different bands and labelling
    for im_index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        orig_image_height, orig_image_width = int(batch['image_height'][0]), int(batch['image_width'][0])
        np_image = batch['image'].detach().squeeze().cpu().numpy()

        pad_h1 = (np_image.shape[0] - orig_image_height) // 2
        pad_w1 = (np_image.shape[1] - orig_image_width) // 2
        pad_h2 = (np_image.shape[0] - orig_image_height) - pad_h1
        pad_w2 = (np_image.shape[1] - orig_image_width) - pad_w1

        np_image = np_image[pad_h1:-pad_h2, pad_w1:-pad_w2]  # unpads the image for loss computation and display

        gt_mask = batch['mask'][:, pad_h1:-pad_h2, pad_w1:-pad_w2]
        image_name = batch['image_name'][0]

        all_model_outputs = []
        display_dice_scores = []

        num_classes = 3
        gt_one_hot = F.one_hot(gt_mask.long(), num_classes).permute(0, 3, 1, 2).float()

        for model, mname in zip(models, model_names):
            confidence_map = None # Initialise

            # classical methods
            if mname == 'watershed':
                torch_one_hot, mask = run_watershed(np_image, image_name, os.path.join(output_folder, mname))
            elif mname == 'multiotsu':
                torch_one_hot, mask = run_multiotsu(np_image, image_name, os.path.join(output_folder, mname))
            elif mname in nnunet_model_names:  # nnunet results are pre-computed
                torch_one_hot, mask = read_nnunet_inference_from_file(os.path.join(model, image_name + '.tif'))
            else:  # standard ML models
                if multi_augment:
                    torch_mask, mask, confidence_map = model_multi_augment_predict_and_process(model, batch['image'])
                else:
                    torch_mask, mask, confidence_map = model_predict_and_process(model, batch['image'])
                num_classes = torch_mask.shape[1] # For multi-class segmentation
                torch_one_hot = F.one_hot(torch_mask.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()

                torch_one_hot = torch_one_hot[:, :, pad_h1:-pad_h2, pad_w1:-pad_w2]  # unpads model outputs
                mask = mask[:, pad_h1:-pad_h2, pad_w1:-pad_w2]
                # Unpad confidence map
                if confidence_map is not None:
                    confidence_map = confidence_map[pad_h1:-pad_h2, pad_w1:-pad_w2]

            # dice score calculation
            dice_score = multiclass_dice_coeff(torch_one_hot[:, 1:, ...],
                                               gt_one_hot[:, 1:, ...],
                                               reduce_batch_first=False).cpu().numpy()

            dice_score_multi = multiclass_dice_coeff(torch_one_hot,
                                                     gt_one_hot,
                                                     reduce_batch_first=False).cpu().numpy()
            
            # Per-class dice scores (only for 3-class ML models)
            if mname not in ['watershed', 'multiotsu'] and mname not in nnunet_model_names:
                dice_band = multiclass_dice_coeff(torch_one_hot[:, 1:2, ...],
                                                  gt_one_hot[:, 1:2, ...],
                                                  reduce_batch_first=False).cpu().numpy()
                                                
                dice_well = multiclass_dice_coeff(torch_one_hot[:, 2:3, ...],
                                                  gt_one_hot[:, 2:3, ...],
                                                  reduce_batch_first=False).cpu().numpy()
                
                # Store per-class dice immediately
                metrics_dict['Band Dice Score'][image_name].append(dice_band)
                metrics_dict['Well Dice Score'][image_name].append(dice_well)
            display_dice_scores.append('Dice Score: %.3f' % dice_score)



            # confusion matrix calculation
            if mname == 'watershed':
                c_mask = mask.flatten()
                c_mask[c_mask > 0] = 1
            elif mname == 'multiotsu' or mname in nnunet_model_names:
                c_mask = mask.flatten()
            else:
                c_mask = mask.argmax(axis=0).flatten()

            # Standard metrics to complement Dice score
            # Convert multi-class to binary (foreground vs background)
            c_mask_binary = (c_mask > 0).astype(int)
            gt_mask_binary = (gt_mask.numpy().squeeze().flatten() > 0).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                gt_mask_binary, c_mask_binary, labels=[0, 1]
            ).ravel()

            # Recall: Of all GT positives, how many did we find?
            if tp + fn == 0:                 # no GT positives exist
                recall = 1.0 if fp == 0 else 0.0   # false positive present so penalise
            else:
                recall = tp / (tp + fn)

            # Precision: Of all predictions, how many were correct?
            if tp + fp == 0:                 # no predictions made
                precision = 1.0 if fn == 0 else 0.0   # missed everything so penalise
            else:
                precision = tp / (tp + fp)

            
            # Foreground F1 score
            if (precision + recall) == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            # Additional metrics output for multiclass prediction
            if mname not in ['watershed', 'multiotsu'] and mname not in nnunet_model_names:
                gt_labels = gt_mask.numpy().squeeze().astype(int).flatten()
                pred_labels = mask.argmax(axis=0).flatten()
    
                for cls_id, cls_name in [(1, "Band"), (2, "Well")]:
                    gt_binary   = (gt_labels == cls_id).astype(int)
                    pred_binary = (pred_labels == cls_id).astype(int)
                    tn_cls, fp_cls, fn_cls, tp_cls = confusion_matrix(
                        gt_binary, pred_binary, labels=[0, 1]
                    ).ravel()

                    # Recall
                    if tp_cls + fn_cls == 0:          # no GT positives for this class
                        recall_cls = 1.0 if fp_cls == 0 else 0.0
                    else:
                        recall_cls = tp_cls / (tp_cls + fn_cls)
        
                    # Precision
                    if tp_cls + fp_cls == 0:          # no predicted positives
                        precision_cls = 1.0 if fn_cls == 0 else 0.0
                    else:
                        precision_cls = tp_cls / (tp_cls + fp_cls)
        
                    # F1 score
                    if (precision_cls + recall_cls) == 0:
                        f1_cls = 0.0
                    else:
                        f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls)
        
                    # Store metrics
                    metrics_dict[f"{cls_name} TP"][image_name].append(tp_cls)
                    metrics_dict[f"{cls_name} FP"][image_name].append(fp_cls)
                    metrics_dict[f"{cls_name} FN"][image_name].append(fn_cls)
                    metrics_dict[f"{cls_name} TN"][image_name].append(tn_cls)
                    metrics_dict[f"{cls_name} Precision"][image_name].append(precision_cls)
                    metrics_dict[f"{cls_name} Recall"][image_name].append(recall_cls)
                    metrics_dict[f"{cls_name} F1"][image_name].append(f1_cls)


            # Hausdorff distance calculation here
            # Extract boundary points of segmentation maps
            ground_truth_boundary = np.argwhere(find_boundaries(gt_one_hot[:, 1:, ...].cpu().numpy().squeeze().astype(int), mode="outer"))
            prediction_boundary = np.argwhere(find_boundaries(torch_one_hot[:, 1:, ...].cpu().numpy().squeeze().astype(int), mode="outer"))

            # Compute distance - since it's directed, need to compute twice and take max
            d1 = directed_hausdorff(ground_truth_boundary, prediction_boundary)[0]
            d2 = directed_hausdorff(prediction_boundary, ground_truth_boundary)[0]
            hausdorff_distance = max(d1, d2)

            for metric, value in zip(metrics_for_zip, [dice_score, dice_score_multi, tn, fp, fn, tp, precision, recall, f1, hausdorff_distance]):
                metrics_dict[metric][image_name].append(value)

            # direct model plotting
            if mname == 'watershed':
                labels = mask
            elif mname == 'multiotsu' or mname in nnunet_model_names:
                labels, _ = ndi.label(mask)
            else:
                labels, _ = ndi.label(mask.argmax(axis=0))

            rgb_labels = label2rgb(labels, image=np_image)

            all_model_outputs.append(rgb_labels)
            save_model_output(output_folder, mname, image_name, rgb_labels)
            save_segmentation_map(output_folder, mname, image_name, mask, confidence_map=confidence_map, band_colour=band_colour, well_colour=well_colour)

            if mname not in ['watershed', 'multiotsu'] and mname not in nnunet_model_names:
                save_annotated_output(output_folder, mname, image_name, rgb_labels, metrics_dict)
       
        gt_labels, _ = ndi.label(gt_one_hot.numpy().squeeze().argmax(axis=0))
        gt_rgb_labels = label2rgb(gt_labels, image=np_image)

        # comparison plotting
        plot_model_comparison([gt_rgb_labels] + all_model_outputs, ['Ground Truth'] + model_names,
                              image_name, np_image, output_folder,
                              images_per_row, double_indexing, comments=[''] + display_dice_scores)

    # combines and saves final dice score data into a table
    for key in all_metrics:
        value = metrics_dict[key]
        if not value or all(len(v) == 0 for v in value.values()):
            continue # For metrics that are not included in the classical methods, skip that CSV
        pd_data = pd.DataFrame.from_dict(value, orient='index')
        pd_data.columns = model_names
        if len(pd_data) == 1: # this solves issues with computing mean when only one image is present
            pd_data = pd_data.map(lambda x: x.item() if isinstance(x, (np.ndarray,)) else x)
        pd_data.loc['mean'] = pd_data.mean()
        pd_data.to_csv(os.path.join(output_folder, 'metrics', '%s.csv' % key), mode='w', header=True, index=True, index_label='Image')


def segment_and_plot(models, model_names, input_folder, output_folder, minmax_norm=False, percentile_norm=False,
                     multi_augment=False, images_per_row=2, run_classical_techniques=False, nnunet_models_and_folders=None,
                     band_colour=(163, 106, 13), well_colour=(0, 255, 0), run_analysis=False, ladder_sizes_bp=None):
    """
    Segments images in input_folder using models and saves the output image and a quick comparison to the output folder.
    :param models: Pre-loaded pytorch segmentation models
    :param model_names: Name for each model (list)
    :param input_folder: Input folder containing gel images
    :param output_folder: Output folder to save results
    :param minmax_norm: Set to true to min-max normalise images before segmentation
    :param percentile_norm: Set to true to percentile normalise images before segmentation
    :param multi_augment: Set to true to perform test-time augmentation
    :param images_per_row: Number of images to plot per row in the output comparison figure
    :param run_classical_techniques: Set to true to also run watershed and multiotsu segmentation apart from selected models
    :param nnunet_models_and_folders: List of tuples containing (model name, folder location) for pre-computed nnunet results on the same dataset
    :param map_pixel_colour: Colour to use for positive pixels in the output segmentation map (tuple, RGB)
    :param run_analysis: Set to true to run distance measurement analysis on segmentation masks
    :param ladder_sizes_bp: Optional pre-specified ladder sizes (list of floats), otherwise prompts per image
    :return: N/A (all outputs saved to file)
    """

    dataset = ImageDataset(input_folder, 1, padding=False, individual_padding=True, minmax_norm=minmax_norm,
                           percentile_norm=percentile_norm)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True)

    if run_classical_techniques:
        models.extend(['watershed', 'multiotsu'])
        model_names.extend(['watershed', 'multiotsu'])

    if nnunet_models_and_folders:
        nnunet_model_names, nnunet_eval_outputs = zip(*nnunet_models_and_folders)
        models.extend(nnunet_eval_outputs)
        model_names.extend(nnunet_model_names)
    else:
        nnunet_model_names = []

    double_indexing = True  # axes will have two indices rather than one
    if math.ceil((len(model_names) + 1) / images_per_row) == 1:  # axes will only have one index rather than 2
        double_indexing = False

    for mname in model_names:
        create_dir_if_empty(os.path.join(output_folder, mname))

    create_dir_if_empty(os.path.join(output_folder, 'method_comparison'))

    # Track analysis results per model of success vs failed gel image post-segmentation analysis (distance and weight measurement) per imae
    analysis_results = {}
    if run_analysis:
        analysis_results = {mname: {'successful': 0, 'failed': 0, 'log_lines': []} for mname in model_names}

    # preparing model outputs, including separation of different bands and labelling
    for im_index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        orig_image_height, orig_image_width = int(batch['image_height'][0]), int(batch['image_width'][0])
        np_image = batch['image'].detach().squeeze().cpu().numpy()

        pad_h1 = (np_image.shape[0] - orig_image_height) // 2
        pad_w1 = (np_image.shape[1] - orig_image_width) // 2
        pad_h2 = (np_image.shape[0] - orig_image_height) - pad_h1
        pad_w2 = (np_image.shape[1] - orig_image_width) - pad_w1

        np_image = np_image[pad_h1:-pad_h2, pad_w1:-pad_w2] # unpads the image for loss computation and display

        image_name = batch['image_name'][0]
        all_model_outputs = []

        for model, mname in zip(models, model_names):
            confidence_map = None # Initialise 
            # classical methods
            if mname == 'watershed':
                _, mask = run_watershed(np_image, image_name, None)
            elif mname == 'multiotsu':
                _, mask = run_multiotsu(np_image, image_name, None)
            elif mname in nnunet_model_names:  # nnunet results are pre-computed
                _, mask = read_nnunet_inference_from_file(os.path.join(model, image_name + '.tif'))
            else:
                if multi_augment:
                    _, mask, confidence_map = model_multi_augment_predict_and_process(model, batch['image'])
                else:
                    _, mask, confidence_map = model_predict_and_process(model, batch['image'])
                mask = mask[:, pad_h1:-pad_h2, pad_w1:-pad_w2]
                if confidence_map is not None: 
                    #Same unpadding logic but for 1 channel
                    confidence_map = confidence_map[pad_h1:-pad_h2, pad_w1:-pad_w2]


            # direct model plotting
            if mname == 'watershed':
                labels = mask
            elif mname == 'multiotsu' or mname in nnunet_model_names:
                labels, _ = ndi.label(mask)
            else:
                labels, _ = ndi.label(mask.argmax(axis=0))

            rgb_labels = label2rgb(labels, image=np_image)
            all_model_outputs.append(rgb_labels)
            save_model_output(output_folder, mname, image_name, rgb_labels)
            save_segmentation_map(output_folder, mname, image_name, mask,confidence_map=confidence_map, band_colour=(163, 106, 13), well_colour=(0, 255, 0))

            if run_analysis:
                # Get path to the raw mask
                raw_mask_path = os.path.join(output_folder, mname, f'{image_name}_raw_mask.tif')

                # Check if confidence map exists ( Theoretical string path and actual path)
                conf_path = os.path.join(output_folder, mname, f'{image_name}_confidence_map.tif')
                confidence_path = conf_path if os.path.exists(conf_path) else None 
                
                # Set up analysis output paths 
                analysis_plot_path = os.path.join(output_folder, mname, f'{image_name}_analysis.png')
                analysis_report_path = os.path.join(output_folder, mname, f'{image_name}_report.txt')
                analysis_csv_path = os.path.join(output_folder, mname, f'{image_name}_distances.csv')
                
                print(f"\n>>> Running analysis on {image_name} (model: {mname})...")
                
                try:
                    # Runring analysis
                    results = analyze_gel_with_proper_well_centric_approach(
                        segmap_path=raw_mask_path,
                        confidence_path=confidence_path,
                        ladder_lane_id=None,  # Auto-select
                        ladder_sizes_bp=ladder_sizes_bp,  # Will prompt if None
                        renumber_lanes=True,
                        show_plot=False,  # Need to update to remove 
                        save_plot_path=analysis_plot_path,
                        save_report_path=analysis_report_path
                    )
                    
                    # Save CSV if results available
                    if results and results.get('distances'):
                        df = pd.DataFrame(results['distances'])
                        df['image_name'] = image_name
                        df['model_name'] = mname
                        df.to_csv(analysis_csv_path, index=False)
                        analysis_results[mname]['successful'] += 1
                        analysis_results[mname]['log_lines'].append(f"SUCCESS: {image_name}")
                        print(f"Analysis complete: {len(results['distances'])} distance measurements saved")
                    else:
                        analysis_results[mname]['failed'] += 1
                        analysis_results[mname]['log_lines'].append(f"FAILED: {image_name} - No distances found")
                        print(f"Analysis completed but no distances found")
                    
                except Exception as e:
                    analysis_results[mname]['failed'] += 1
                    analysis_results[mname]['log_lines'].append(f"FAILED: {image_name} - {str(e)}")
                    print(f" Analysis failed for {image_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()

        plot_model_comparison(all_model_outputs, model_names, image_name, np_image, output_folder,
                              images_per_row, double_indexing)

    # Write analysis summary logs at the end
    if run_analysis:
        from datetime import datetime
        for mname in model_names:
            results = analysis_results[mname]
            total_analyzed = results['successful'] + results['failed']
            
            if total_analyzed > 0:  # Only if analysis was run for this model
                log_path = os.path.join(output_folder, mname, 'analysis_log.txt')
                with open(log_path, 'w') as f:
                    f.write(f"Gel Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("="*60 + "\n\n")
                    f.write(f"Model: {mname}\n")
                    f.write(f"Total images: {total_analyzed}\n")
                    f.write(f"Successful: {results['successful']}\n")
                    f.write(f"Failed: {results['failed']}\n\n")
                    f.write("Details:\n")
                    f.write("-" * 20 + "\n")
                    for line in results['log_lines']:
                        f.write(line + "\n")
                print(f"\n Analysis summary saved: {log_path}")
