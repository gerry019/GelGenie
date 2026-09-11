import numpy as np
import argparse
import glob # For file matching patterns
import pandas as pd
import os
from datetime import datetime
from skimage.io import imread
from skimage.measure import regionprops, label
from skimage.morphology import convex_hull_image
from matplotlib import patches # For drawing shapes
import matplotlib
import sys

if 'COLAB_RELEASE_TAG' in os.environ:
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.ion()

def console(msg=""):
    """Print directly to the real terminal, bypassing any stdout redirection."""
    if 'COLAB_RELEASE_TAG' in os.environ:
        print(msg, flush=True)
    else:
        print(msg, file=sys.__stdout__, flush=True)

def log(msg=""):
    """Print to both the log file (stdout) and the real terminal."""
    print(msg)      # goes to the log file
    console(msg)    # goes to the real terminal


# Well-centric analysis
class WellCentricLaneAnalyzer:
    def __init__(self, segmap_path, confidence_path=None, use_area_filter=True, use_confidence_filter=True):
        """
        segmap format: 2 = wells, 1 = bands 0 = background
        """
        log(f"Loading segmentation map: {segmap_path}")
        self.segmap = imread(segmap_path) # Numpy array

        # Load confidence map if provided for reporting in csv file
        self.confidence_map = None
        self.use_area_filter = use_area_filter
        self.use_confidence_filter = use_confidence_filter
        if confidence_path and os.path.exists(confidence_path):
            self.confidence_map = imread(confidence_path).astype(np.float32)  # To be used for calculations
            log(f"Loaded confidence map: {confidence_path}")
        else:
            log("No confidence map provided")

       # storage
        self.wells = [] # From regionprops
        self.bands = []
        self.bands_mask = None  # binary band mask, persisted so repair_split_bands() can relabel against it
        self.extended_lanes = {}   # {lane_id: {..., 'bbox': (top,left,bottom,right), 'wells':[], 'bands':[]}} (dictionary)
        self.complete_lanes = {} # With bands
        self.incomplete_lanes = {} # Without bands
        self.distances = [] # Well to band

        # params
        self.gel_height = self.segmap.shape[0]  # Used for lane bottom boundary when no well below

        # ladder
        self.ladder_calibrations = {}   # {ladder_lane_id: {curve, coeffs, migs_min, migs_max, n, known_sizes}}
        self.lane_to_ladder = {}        # {lane_id: ladder_lane_id} assignment
        self.band_size_bp_by_id = {}   # id(band) and size_bp (from calibration)
        self.outside_ladder = []        # list of (lane_id, band) with NaN size

        # filter info
        self.well_filter_info = {}
        self.band_filter_info = {}
        self.removed_low_confidence_wells = []
        self.removed_low_confidence_bands = []
        self.removed_small_wells = []
        self.removed_small_bands = []


        log(f"Loaded segmentation map: {self.segmap.shape}") # Size
        log(f"Unique labels: {np.unique(self.segmap)}") # Labels

    def extract_wells_and_bands(self):
        log("\n=== Step 1: Extracting Wells and Bands and applying filtering===")
        log(f"   Area filtering: {'ON' if self.use_area_filter else 'OFF'}")

        if self.confidence_map is None:
            log("   Confidence filtering: NOT APPLIED (no confidence map)")
        else:
            log(f"   Confidence filtering: {'ON' if self.use_confidence_filter else 'OFF'}")

        # Wells
        wells_mask = (self.segmap == 2) # Binary mask
        if np.any(wells_mask): # Check if any exist
            wells_labeled = label(wells_mask) # label each connected componenet
            all_wells = regionprops(wells_labeled) # Get region properties
            wells_sorted = sorted(all_wells, key=lambda w: w.centroid[1])  # left to right
            self.all_wells_unfiltered = wells_sorted

            total_image_area = self.segmap.shape[0] * self.segmap.shape[1]

            # Filter 1: remove low-confidence wells (mean confidence < 0.8)
            confidence_threshold = 0.8
            if self.confidence_map is not None and self.use_confidence_filter:
                confidence_filtered_wells = []
                removed_low_confidence_wells = []
                for well in wells_sorted:
                    coords = well.coords
                    mean_conf = float(np.mean(self.confidence_map[coords[:, 0], coords[:, 1]])) # Average for every coordinte of that closed component
                    if mean_conf >= confidence_threshold:
                        confidence_filtered_wells.append(well)
                    else:
                        removed_low_confidence_wells.append(well)

                self.removed_low_confidence_wells = removed_low_confidence_wells

                if removed_low_confidence_wells:
                    log(f"   Filtered out {len(removed_low_confidence_wells)} low-confidence wells (< {confidence_threshold})")
            else:
                confidence_filtered_wells = wells_sorted

            # Filter 2: remove wells < 0.00875% of total image area
            min_well_area = 0.0000875 * total_image_area

            if self.use_area_filter:
                self.removed_small_wells = [w for w in confidence_filtered_wells if w.area < min_well_area]
                self.wells = [w for w in confidence_filtered_wells if w.area >= min_well_area]
            else:
                self.removed_small_wells = []
                self.wells = confidence_filtered_wells

            self.well_filter_info = { # Filter information
                'total_image_area': float(total_image_area),
                'threshold': float(min_well_area),
                'threshold_percent': 0.00875,
                'filtered_wells': [idx for idx, w in enumerate(confidence_filtered_wells, start=1) if w.area < min_well_area]
            }

            removed_wells = len(confidence_filtered_wells) - len(self.wells)
            if removed_wells > 0:
                log(f"   Filtered out {removed_wells} small wells (< 0.00875% of image area)")

            filtering_applied = self.use_area_filter or (self.use_confidence_filter and self.confidence_map is not None)

            if filtering_applied:
                log(f"   Found {len(self.wells)} wells (after filtering)")
            else:
                log(f"   Found {len(self.wells)} wells")
        # Bands
        bands_mask = (self.segmap == 1)
        if np.any(bands_mask):
            bands_labeled = label(bands_mask)
            all_bands = regionprops(bands_labeled)
            self.all_bands_unfiltered = all_bands

            # Filter 1: remove low-confidence bands (mean confidence < 0.8)
            confidence_threshold = 0.8
            if self.confidence_map is not None and self.use_confidence_filter:
                confidence_filtered_bands = []
                removed_low_confidence = []
                for band in all_bands:
                    coords = band.coords
                    mean_conf = float(np.mean(self.confidence_map[coords[:, 0], coords[:, 1]]))
                    if mean_conf >= confidence_threshold:
                        confidence_filtered_bands.append(band)
                    else:
                        removed_low_confidence.append(band)

                self.removed_low_confidence_bands = removed_low_confidence

                if removed_low_confidence:
                    log(f"   Filtered out {len(removed_low_confidence)} low-confidence bands (< {confidence_threshold})")
            else:
                confidence_filtered_bands = all_bands

            # Filter 2: remove bands < 0.0075% of total image area
            min_band_area = 0.000075 * total_image_area

            if self.use_area_filter:
                self.removed_small_bands = [b for b in confidence_filtered_bands if b.area < min_band_area]
                area_filtered_bands = [b for b in confidence_filtered_bands if b.area >= min_band_area]
            else:
                self.removed_small_bands = []
                area_filtered_bands = confidence_filtered_bands

            self.band_filter_info = {
                'total_image_area': float(total_image_area),
                'threshold': float(min_band_area),
                'threshold_percent': 0.0075,
                'filtered_bands': [b.label for b in confidence_filtered_bands if b.area < min_band_area]
            }

            removed_small = len(confidence_filtered_bands) - len(area_filtered_bands)
            if removed_small > 0:
                log(f"   Filtered out {removed_small} small bands (< 0.0075% of image area)")

            self.bands = area_filtered_bands
            if filtering_applied:
                log(f"   Found {len(self.bands)} bands (after filtering)")
            else:
                log(f"   Found {len(self.bands)} bands")

            filtered_mask = np.zeros_like(bands_mask, dtype=bool)
            for band in self.bands:
                filtered_mask[bands_labeled == band.label] = True
            self.bands_mask = filtered_mask

            filtered_labels = label(self.bands_mask)
            self.bands = list(regionprops(filtered_labels)) # Needed for more  processing

            for band in self.bands:
                band.id = band.label

        else:
            log("   No bands found (label=1)")
            self.bands_mask = np.zeros_like(self.segmap, dtype=bool)

        return True

    def cluster_wells_into_lane_groups(self):
        log("\n=== Step 2: Clustering wells into lanes (One well per lane) ===")

        self.well_groups = []
        for well_idx, well in enumerate(self.wells):
            cx = float(well.centroid[1])
            cy = float(well.centroid[0])
            width = well.bbox[3] - well.bbox[1]

            self.well_groups.append({  # Well/lane information
                'lane_id': well_idx,
                'wells': [well],
                'well_indices': [well_idx],
                'center_x': cx,
                'center_y': cy,
                'width': width,
                'well_count': 1 
            })

        log(f"   Created {len(self.well_groups)} lane groups (1 well per lane).")
        return True


    def create_extended_lanes_from_wells(self):
        log("\n=== Step 3: Creating Extended Lane Boundaries ===")

        self.extended_lanes = {}
        for group in self.well_groups:
            lid = group['lane_id']
            gw = group['wells'] # Only one well per group

            # Horizontal boundaries from well's own edges
            left = float(min(w.bbox[1] for w in gw))
            right = float(max(w.bbox[3] for w in gw))

            # Top = bottom edge of the well
            top = float(min(w.bbox[2] for w in gw))

            # Bottom = top of the next well below that horizontally overlaps, or image bottom
            wells_below = [w.bbox[0] for w in self.wells
                           if w not in gw
                           and w.bbox[0] > top
                           and (w.bbox[1] < right and w.bbox[3] > left)]

            bottom = float(min(wells_below)) if wells_below else float(self.gel_height)

            self.extended_lanes[lid] = {
                'lane_id': lid,
                'wells': gw,
                'center_x': group['center_x'],
                'center_y': group['center_y'],
                'bbox': (top, left, bottom, right),
                'bands': []
            }

        log(f"   Created {len(self.extended_lanes)} extended lanes.")
        return True


    def assign_bands_to_extended_lanes(self):
        log("\n=== Step 4: Assigning Bands to Extended Lanes ===")
        assigned = set() # Prevents bands being assigned to multiple lanes
        for lane_id, lane in self.extended_lanes.items():
            bbox = lane['bbox']  # (top, left, bottom, right)
            lane_bands = []
            for band in self.bands:
                if id(band) in assigned: continue
                y, x = band.centroid
                if (bbox[1] <= x <= bbox[3]) and (bbox[0] <= y <= bbox[2]): # If within boundary, assign
                    lane_bands.append(band)
                    assigned.add(id(band))
            lane['bands'] = lane_bands
            print(f"   Lane {lane_id}: {len(lane['wells'])} wells → {len(lane_bands)} bands")

        # classify lane id in lane dictionary
        self.complete_lanes = {lid: ln for lid, ln in self.extended_lanes.items() if ln['bands']}
        self.incomplete_lanes = {lid: ln for lid, ln in self.extended_lanes.items() if not ln['bands']}
        unassigned = len(self.bands) - len(assigned)
        if unassigned > 0:
            log(f"   Unassigned bands: {unassigned} (outside all lane boundaries)")
        return True

    def repair_split_bands(self):
        """
        Same split-band repair logic as the DBSCAN pipeline: within each lane,
        find pairs of bands that are actually one physical band split into two
        mask fragments (side-by-side, similar y), and merge them via convex hull.
        """
        log("\n=== Step 4b: Repairing Split Bands ===")

        def find_split_bands(bands):
            """
            Check all pairs of bands in a lane for split band condition.
            ratio = y_range / x_range < 0.5 means bands are side by side = split band.
            Returns list of index pairs that are part of a split pair.
            """
            split_pairs = []
            for i in range(len(bands)):
                for j in range(i + 1, len(bands)):
                    y_i = bands[i].centroid[0]
                    y_j = bands[j].centroid[0]
                    x_i = bands[i].centroid[1]
                    x_j = bands[j].centroid[1]
                    y_range = abs(y_i - y_j)
                    x_range = abs(x_i - x_j) if abs(x_i - x_j) > 0.001 else 0.001
                    ratio = y_range / x_range
                    if ratio < 0.5:
                        split_pairs.append((i, j))
            return split_pairs

        bands_labeled = label(self.bands_mask)
        self.merged_pixels = np.zeros_like(self.bands_mask, dtype=bool)

        for lane_id, lane in sorted(self.extended_lanes.items()):
            bands = lane['bands']
            if len(bands) < 2:
                continue

            split_pairs = find_split_bands(bands)
            if not split_pairs:
                continue

            log(f"Lane {lane_id}: {len(split_pairs)} split band pair(s) detected")

            # Create a mask containing only the current lane's bands
            lane_mask = np.zeros_like(self.bands_mask, dtype=bool)
            for band in bands:
                lane_mask[bands_labeled == band.label] = True

            for i, j in split_pairs:
                mask_i = bands_labeled == bands[i].label
                mask_j = bands_labeled == bands[j].label
                merged_mask = convex_hull_image(mask_i | mask_j)
                lane_mask[merged_mask] = True
                self.bands_mask[merged_mask] = True
                self.merged_pixels[merged_mask] = True

            # Relabel only this lane's local mask (not globally, to avoid disturbing other lanes' labels)
            lane_labels = label(lane_mask)
            repaired_bands = list(regionprops(lane_labels))
            lane['bands'] = repaired_bands

        # Keep self.bands in sync with any merges done above (flattened across lanes,
        # excluding any bands that were outside all lane boundaries to begin with)
        repaired_flat = [
            band
            for lane in self.extended_lanes.values()
            for band in lane['bands']
        ]
        self.bands = repaired_flat

        # complete_lanes / incomplete_lanes reference the same lane dicts, so
        # their 'bands' entries are automatically up to date (shared objects)

        return True

    def renumber_lanes_left_to_right(self):
        if not self.extended_lanes:
            return
        # sort by left x of bbox. To include all lanes (complete + incomplete)
        sorted_items = sorted(self.extended_lanes.items(), key=lambda kv: kv[1]['bbox'][1])
        old_to_new = {old_id: new_id+1 for new_id, (old_id, _) in enumerate(sorted_items)}

        self.extended_lanes = {old_to_new[k]: v for k, v in self.extended_lanes.items()}
        self.complete_lanes = {old_to_new[k]: v for k, v in self.complete_lanes.items()}
        self.incomplete_lanes = {old_to_new[k]: v for k, v in self.incomplete_lanes.items()}

        for lid, lane in self.extended_lanes.items():
            lane['lane_id'] = lid

        log("\n   All lanes renumbered left to right (starting from 1).")

    # Ladder sizes to  linear calibration, to use the well centroud and not the beginning of the lane
    def _lane_origin_y(self, lane):
        """Per-lane origin for migration: well centroid (y)."""
        return float(lane['wells'][0].centroid[0])

    def interpolate_size_local(self, m, ladder_migs_sorted, log_sizes_sorted):

        n = len(ladder_migs_sorted)

        # Outside the calibrated range entirely - do not extrapolate
        if m < ladder_migs_sorted[0] or m > ladder_migs_sorted[-1]:
            return np.nan

        # Find the two bracketing ladder rungs either side of m
        idx = np.searchsorted(ladder_migs_sorted, m, side='right') - 1
        idx = min(max(idx, 0), n - 2)  # clamp so idx/idx+1 stay valid, incl. exact top/bottom rung

        mig_low, mig_high = ladder_migs_sorted[idx], ladder_migs_sorted[idx + 1]
        log_low, log_high = log_sizes_sorted[idx], log_sizes_sorted[idx + 1]

        # How far along between the two rungs (0 = at mig_low, 1 = at mig_high)
        fraction = (m - mig_low) / (mig_high - mig_low)

        # Interpolate in log space, then convert back to bp
        log_size_est = log_low + fraction * (log_high - log_low)
        return float(10 ** log_size_est)

    def calibrate(self, ladder_sizes_bp=None, interactive=True):
            log(f"\n=== Step 5: Ladder Calibration (local two-point interpolation) ===")
            if not self.complete_lanes:
                log("No complete lanes available for calibration.")
                self.ladder_calibrations = {}
                self.lane_to_ladder = {}
                return False

            # Auto-select ladder: lane with most bands (and ask user to verify)
            auto_ladder_id = max(self.complete_lanes.keys(), key=lambda k: len(self.complete_lanes[k]['bands']))
            print(f"\nAuto-selected ladder: Lane {auto_ladder_id}")

            if not interactive:
                if ladder_sizes_bp is None:
                    log("Non-interactive mode requires ladder_sizes_bp to be provided. Skipping calibration.")
                    self.ladder_calibrations = {}
                    self.lane_to_ladder = {}
                    return False
                ladder_ids = [auto_ladder_id]
                log(f"Non-interactive mode: using auto-selected Lane {auto_ladder_id} with provided sizes.")
            else:
                console(f"Use Lane {auto_ladder_id} as the ladder? (Y/N): ")
                answer = input().strip().lower()

                if answer in ("y", "yes", ""):
                    ladder_ids = [auto_ladder_id]
                else:
                    console("Enter ladder lane number(s), e.g. 1 or 1,8: ")
                    raw = input().strip()
                    ladder_ids = [int(x.strip()) for x in raw.split(",") if x.strip()]

            ladder_calibrations = {} # For multiple ladders

            for ladder_id in ladder_ids: # through each ladder
                if ladder_id not in self.complete_lanes:
                    log(f"Lane {ladder_id} is not a complete lane, skipping.")
                    continue

                ladder_lane = self.complete_lanes[ladder_id]
                ladder_bands_sorted = sorted(ladder_lane['bands'], key=lambda b: b.centroid[0])  # top to bottom
                n = len(ladder_bands_sorted)
                if n < 2:
                    log(f"Lane {ladder_id}: too few bands, skipping.")
                    continue

                # Euclidean migration from well centroid (origin)
                y0 = self._lane_origin_y(ladder_lane)
                x0 = float(ladder_lane['wells'][0].centroid[1])
                ladder_migs = np.array([
                    float(np.sqrt((b.centroid[0] - y0) ** 2 + (b.centroid[1] - x0) ** 2))
                    for b in ladder_bands_sorted
                ], dtype=float)


                # Get sizes, with retry if the count doesn't match
                sizes = None
                if ladder_sizes_bp is not None:
                    candidate_sizes = np.array(ladder_sizes_bp, dtype=float)
                    if len(candidate_sizes) == n:
                        sizes = candidate_sizes
                    else:
                        log(f"Provided {len(candidate_sizes)} sizes, but Lane {ladder_id} has {n} bands.")

                while sizes is None:
                    if not interactive:
                        log(f"Non-interactive mode: could not match provided sizes to {n} bands in Lane {ladder_id}. Skipping.")
                        self.ladder_calibrations = {}
                        self.lane_to_ladder = {}
                        return False

                    console(f"Enter {n} sizes in bp for Lane {ladder_id} (top to bottom): ")
                    raw = input().strip()
                    try:
                        candidate_sizes = np.array(
                            [float(x) for x in raw.replace(';', ',').split(',') if x.strip() != ""],
                            dtype=float
                        )
                    except Exception:
                        print("Could not parse sizes. Please try again.")
                        continue

                    if len(candidate_sizes) != n:
                        print(f"Provided {len(candidate_sizes)} sizes but {n} bands in Lane {ladder_id}. Please re-enter.")
                        continue

                    sizes = candidate_sizes

                if not np.all(np.diff(sizes) < 0): # checks for negative difference
                    print("Note: ladder sizes are not strictly decreasing top to bottom. Proceeding anyway.")

                log_sizes = np.log10(sizes)

                # Sort by migration distance ascending so bracketing works correctly
                sort_idx = np.argsort(ladder_migs)
                migs_sorted = ladder_migs[sort_idx]
                log_sizes_sorted = log_sizes[sort_idx]

                ladder_calibrations[ladder_id] = {
                    'migs_sorted': migs_sorted,
                    'log_sizes_sorted': log_sizes_sorted,
                    'migs_min': float(migs_sorted.min()),
                    'migs_max': float(migs_sorted.max()),
                    'n': n,
                    'known_sizes': {id(b): float(s) for b, s in zip(ladder_bands_sorted, sizes)},
                }

                log(f"   Local two-point log-linear interpolation calibration complete for Ladder Lane {ladder_id}. Sizes attached for {n} ladder bands.")
            if not ladder_calibrations:
                log("No usable ladder calibration.")
                self.ladder_calibrations = {}
                self.lane_to_ladder = {}
                return False

            lane_to_ladder = {}

            if len(ladder_calibrations) == 1:
                only_lid = next(iter(ladder_calibrations))
                for lid in self.extended_lanes:
                    lane_to_ladder[lid] = only_lid
                log(f"\nAll lanes will use Ladder Lane {only_lid}.")
            else:
                log("\nMultiple ladders selected — assign which sample lanes use each ladder.")

                for lid in ladder_calibrations:
                    lane_to_ladder[lid] = lid

                for lid in ladder_calibrations:
                    console(f"Sample lanes using Ladder Lane {lid} (e.g. 2,3,4 or 2-5): ")
                    raw = input().strip()

                    for part in raw.split(","):
                        part = part.strip()
                        if not part:
                            continue
                        if "-" in part:
                            start, end = map(int, part.split("-"))
                            for num in range(start, end + 1):
                                lane_to_ladder[num] = lid
                        else:
                            lane_to_ladder[int(part)] = lid

                unassigned = [lid for lid in self.extended_lanes if lid not in lane_to_ladder]
                if unassigned:
                    log(
                        "Note: lanes "
                        f"{unassigned} were not assigned "
                        "to any ladder and will not receive size estimates."
                    )

            self.ladder_calibrations = ladder_calibrations
            self.lane_to_ladder = lane_to_ladder

            return True

    def assign_band_sizes(self):
        """Estimate the molecular weight (bp) of every band, using local two-point
            log-linear interpolation between the two ladder rungs bracketing its
            migration distance, based on whichever ladder lane it was assigned to.
            Migration distance is measured as the Euclidean distance from the band's
            centroid to its own lane's well centroid. Bands whose migration falls
            outside the calibrated ladder's range are left without a size estimate."""

        log("\n=== Step 6: Calculating Migration Distances and Estimating Band Sizes ===")
        # Assign known sizes to ladder bands
        band_size_bp_by_id = {}
        for cal in self.ladder_calibrations.values():
            band_size_bp_by_id.update(cal['known_sizes'])

        # Clear outside list
        outside_ladder = []

        # For all other lanes, using whichever ladder that lane was assigned to
        for lane_id, lane in self.extended_lanes.items():
            if lane_id not in self.lane_to_ladder:
                continue
            cal = self.ladder_calibrations[self.lane_to_ladder[lane_id]]
            y0_lane = self._lane_origin_y(lane) # the well centroid

            for band in lane['bands']:
                bid = id(band)
                if bid in band_size_bp_by_id: # only get IDS of those non-ladder bands
                    continue

                # Euclidean migration from this lane's well, then interpolate size 
                x0_lane = float(lane['wells'][0].centroid[1])
                m = float(np.sqrt((band.centroid[0] - y0_lane) ** 2 + (band.centroid[1] - x0_lane) ** 2))
                size_est = self.interpolate_size_local(m, cal['migs_sorted'], cal['log_sizes_sorted'])
                if np.isnan(size_est):
                    outside_ladder.append((lane_id, band))
                band_size_bp_by_id[bid] = size_est

        self.band_size_bp_by_id = band_size_bp_by_id
        self.outside_ladder = outside_ladder

        log(f" Sizes attached for {len(self.band_size_bp_by_id)} bands.")

        if self.outside_ladder:
            print("\n Bands outside ladder range (no size assigned):")
            for lane_id, band in self.outside_ladder:
                by, bx = band.centroid
                print(f"   Lane {lane_id}: band centroid at x={bx:.1f}, y={by:.1f}")

        return True

    # Distances + print px & bp
    def calculate_distances_for_complete_lanes(self):
        log("\n=== Step 6b: Building Well-to-Band Distance Report ===")
        if not self.complete_lanes:
            log(" No complete lanes for distance calculations")
            return False

        self.distances = []
        for lane_id, lane in sorted(self.complete_lanes.items()):
            wells_in_lane = lane['wells']
            bands_in_lane = lane['bands']
            log(f"\nLane {lane_id}: {len(wells_in_lane)} wells x {len(bands_in_lane)} bands")

            for w_idx, well in enumerate(wells_in_lane, start=1):
                bands_sorted = sorted(bands_in_lane, key=lambda b: b.centroid[0])  # top to bottom
                for b_idx, band in enumerate(bands_sorted, start=1):
                    wy, wx = well.centroid
                    by, bx = band.centroid
                    # downward only
                    if by <= wy:
                        continue
                    euclidean_dist = float(np.sqrt((by - wy) ** 2 + (bx - wx) ** 2))  # migration we print
                    size_bp = self.band_size_bp_by_id.get(id(band))
                    if size_bp is not None and not np.isnan(size_bp):
                        print(f"   Well {w_idx} → Band {b_idx}: {euclidean_dist:.1f}px, {int(round(size_bp))}bp")
                    else:
                        print(f"   Well {w_idx} → Band {b_idx}: {euclidean_dist:.1f}px")

                    self.distances.append({
                        'lane_id': lane_id,
                        'well_idx': w_idx,
                        'band_idx': b_idx,
                        'well_centroid_y': float(wy),    # Separate key for y coordinate
                        'well_centroid_x': float(wx),    # Separate key for x coordinate
                        'band_centroid_y': float(by),
                        'band_centroid_x': float(bx),
                        'migration_px_euclidean': euclidean_dist,
                        'size_bp': size_bp,
                        'calibration_ladder_lane': self.lane_to_ladder.get(lane_id),
                    })
        return True


    # Simple visualization
    def create_visualization(self, save_path=None):
        """Create visualization with option to save instead of show if a path is provided"""
        log("\n === Step 7: Creating Visualization ===")
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan', 'lime', 'yellow']

        # Plot 1: original seg map
        ax1 = axes[0]
        # using rainbow for colour mapping
        ax1.imshow(self.segmap, cmap='nipy_spectral')

        # Colour filtered components
        for band in self.removed_low_confidence_bands + self.removed_small_bands:
            coords = band.coords
            ax1.scatter(coords[:, 1], coords[:, 0], c='red', s=1)

        for well in self.removed_low_confidence_wells + self.removed_small_wells:
            coords = well.coords
            ax1.scatter(coords[:, 1], coords[:, 0], c='orange', s=1)

        # Colour key outside the image
        ax1.scatter([], [], c='red', label='Filtered band')
        ax1.scatter([], [], c='orange', label='Filtered well')
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        ax1.set_title('Segmentation (1=bands, 2=wells)', pad=20)
        ax1.axis('off')

        # Plot 2: lanes (properly numbered 1, 2, 3...)
        ax2 = axes[1]
        # Greyscale with lighter background to overlay on it
        ax2.imshow(self.segmap, cmap='gray', alpha=0.6)
        # Loop through lanes in sorted order
        # - lane_id = the dictionary key (int, the ID of the lane)
        # - lane = the dictionary value (with bbox, wells, bands, etc.)
        # - i = a counter from enumerate, used here to cycle through colors
        for i, (lane_id, lane) in enumerate(sorted(self.extended_lanes.items())):
            # Colours get repeated if more lanes are present
            color = colors[i % len(colors)]
            top, left, bottom, right = lane['bbox'] # unpacking
            # Starting corner (as per imshow origin), widt and height
            rect = patches.Rectangle((left, top), right-left, bottom-top,  # to overlay on it, for lanes
                                     # border and fill colour and transparency to see behind
                                    linewidth=3, edgecolor=color, facecolor=color, alpha=0.1)
            ax2.add_patch(rect)
            # Black colour text of lane ID and the addition of a box around that text for visibility
            # rounded corners ans space between edges and text and leaves default font size
            ax2.text((left + right) / 2, top - 10, f'{lane_id}', color='black', fontsize=10, fontweight='bold', ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9), clip_on=False)
        for ladder_id in self.ladder_calibrations:
            if ladder_id not in self.extended_lanes:
                continue
            # Get the lader lane bounding box and adds another box on it
            lbbox = self.extended_lanes[ladder_id]['bbox']
            top, left, bottom, right = lbbox  # unpacking the lbbox
            rect = patches.Rectangle((left, top), right-left, bottom-top,
                                    linewidth=4, edgecolor='yellow', facecolor='none', alpha=0.9)
            ax2.add_patch(rect)
            ax2.text(left + 10, top + 40, f'Ladder (Lane {ladder_id})',
                    bbox=dict(boxstyle="round,pad=0.35", facecolor='yellow', alpha=0.9),
                    fontsize=12, fontweight='bold', color='black')
        ax2.set_title('Extended Lanes (numbered 1, 2, 3...)', pad=20)
        ax2.axis('off')

        # Plot 3: migration lines
        ax3 = axes[2]
        ax3.imshow(self.segmap, cmap='gray', alpha=0.6)
        for lane_id, lane in sorted(self.complete_lanes.items()):
            # To get the actual lane colour, even with incomplete lanes, while adjusting for python based indexing
            color = colors[(lane_id-1) % len(colors)]
            for well in lane['wells']:
                # Get the well and band centroid coordinates and continue if band is below
                wy, wx = well.centroid
                for band in lane['bands']:
                    by, bx = band.centroid
                    if by <= wy:
                        continue
                    ax3.plot([wx, bx], [wy, by], '-', color=color, linewidth=2, alpha=0.8)
        ax3.set_title('Downward Migrations', pad=20)
        ax3.axis('off')

        # Fixes spacing between plots
        plt.tight_layout()
    
        # Save visualization
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            log(f" Visualization saved: {save_path}")

        fig.canvas.draw()

        if 'COLAB_RELEASE_TAG' not in os.environ:
            fig.canvas.flush_events()
            plt.show(block=False)
            plt.pause(0.5)

        return fig


    def save_detailed_report(self, save_path):
        """Generate and save a detailed text report"""
        log("Step 8: Saving Detailed Report")
        report_lines = []
        report_lines.append("="*80)
        report_lines.append(" Well Centric Gel analysisT")
        report_lines.append("="*80)
        report_lines.append(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Segmentation file: {getattr(self, 'segmap_path', 'Unknown')}")
        report_lines.append("")

        # Summary statistics
        report_lines.append("Detection summary:")
        report_lines.append(f"  • Wells detected: {len(self.wells)}")
        report_lines.append(f"  • Bands detected: {len(self.bands)}")
        report_lines.append(f"  • Extended lanes created: {len(self.extended_lanes)}")
        report_lines.append(f"  • Complete lanes (with bands): {len(self.complete_lanes)}")
        report_lines.append(f"  • Incomplete lanes (no bands): {len(self.incomplete_lanes)}")

        if self.incomplete_lanes:
            # converts to string, joins them and prints them out sorted
            incomplete_ids = ", ".join(map(str, sorted(self.incomplete_lanes.keys())))
            report_lines.append(f"    - Incomplete lane IDs: {incomplete_ids}")
        report_lines.append("")

        # Ladder information
        if self.ladder_calibrations:
            report_lines.append("Ladder information:")
            report_lines.append(f"  • Calibration method: local two-point log-linear interpolation")
            for lid, cal in self.ladder_calibrations.items():
                report_lines.append(f"  • Ladder Lane {lid}: {cal['n']} bands")
            if self.outside_ladder:
                # Print the count of any bands outside the range
                report_lines.append(f"  • Bands outside ladder range: {len(self.outside_ladder)}")
            report_lines.append("")

        # Lane details
        report_lines.append("Lane details:")
        for lane_id in sorted(self.extended_lanes.keys()):
            lane = self.extended_lanes[lane_id]
            wells_count = len(lane['wells'])
            bands_count = len(lane['bands'])
            bbox = lane['bbox']
            report_lines.append(f"  Lane {lane_id}:")
            report_lines.append(f"    - Wells: {wells_count}, Bands: {bands_count}")
            report_lines.append(f"    - Center: ({lane['center_x']:.1f}, {lane['center_y']:.1f})")
            report_lines.append(f"    - Bounding box: top={bbox[0]:.1f}, left={bbox[1]:.1f}, bottom={bbox[2]:.1f}, right={bbox[3]:.1f}")
        report_lines.append("")

        # Distance measurements
        if self.distances:
            report_lines.append("Migration measurements:")
            report_lines.append(f"  • Total distance pairs: {len(self.distances)}")

            # Group by lane for reporting
            lane_groups = {}
            for dist in self.distances:
                lid = dist['lane_id']
                if lid not in lane_groups:
                    lane_groups[lid] = []
                lane_groups[lid].append(dist)

            for lane_id in sorted(lane_groups.keys()):
                distances_in_lane = lane_groups[lane_id]
                report_lines.append(f"  Lane {lane_id} ({len(distances_in_lane)} measurements):")

                for dist in distances_in_lane:
                    w_idx, b_idx = dist['well_idx'], dist['band_idx']
                    eucladean_dist = dist['migration_px_euclidean']
                    size_bp = dist.get('size_bp')

                    if size_bp is not None and not np.isnan(size_bp):
                        size_str = f", {int(round(size_bp))}bp"
                    else:
                        size_str = ""

                    report_lines.append(f"    Well {w_idx} → Band {b_idx}: {eucladean_dist:.1f}px{size_str}")
            report_lines.append("")

        report_lines.append("="*80)

        report_text = "\n".join(report_lines)

        with open(save_path, 'w',encoding='utf-8') as f:
            f.write(report_text)
        log(f" Detailed report saved: {save_path}")


    # Report
    def generate_report(self):
        console("Finalizing results")
        print("\n" + "="*80)
        print(" Well-Centric gel analysis report")
        print("="*80)
        print(f"   Wells: {len(self.wells)} | Bands: {len(self.bands)}")
        print(f"   Extended lanes: {len(self.extended_lanes)}")
        print(f"   Complete lanes: {len(self.complete_lanes)} | Incomplete lanes: {len(self.incomplete_lanes)}")
        if self.incomplete_lanes:
            print("   Incomplete lane IDs:", ", ".join(map(str, sorted(self.incomplete_lanes.keys()))))
        if self.outside_ladder:
            print(f"   Bands outside ladder range: {len(self.outside_ladder)}")
        print(f"   Distance pairs stored: {len(self.distances)}")
        return {
            'wells': self.wells,
            'bands': self.bands,
            'extended_lanes': self.extended_lanes,
            'complete_lanes': self.complete_lanes,
            'incomplete_lanes': self.incomplete_lanes,
            'distances': self.distances,
            'ladder_lane_ids': list(self.ladder_calibrations.keys()),
            'outside_ladder_count': len(self.outside_ladder),
        }

def analyze_gel_with_proper_well_centric_approach(
    segmap_path,
    confidence_path=None,
    ladder_sizes_bp=None,
    renumber_lanes=True,
    show_plot=True,
    save_plot_path=None,
    save_report_path=None,
    interactive=True,
    use_area_filter=True,
    use_confidence_filter=True
):
    analyzer = WellCentricLaneAnalyzer(segmap_path, confidence_path=confidence_path, use_area_filter=use_area_filter, use_confidence_filter=use_confidence_filter)
    analyzer.segmap_path = segmap_path  # Store for reporting
    
    try:
        if not analyzer.extract_wells_and_bands(): return None
        if not analyzer.cluster_wells_into_lane_groups(): return None
        if not analyzer.create_extended_lanes_from_wells(): return None
        if not analyzer.assign_bands_to_extended_lanes(): return None
        if not analyzer.repair_split_bands(): return None

        if renumber_lanes:
            analyzer.renumber_lanes_left_to_right()

        # Show numbered lanes before asking which lane is the ladder
        fig = None
        if show_plot or save_plot_path:
            fig = analyzer.create_visualization(save_path=save_plot_path)

            if 'COLAB_RELEASE_TAG' in os.environ and interactive and fig is not None:
                from IPython.display import display
                display(fig)

        calibrated = analyzer.calibrate(
            ladder_sizes_bp=ladder_sizes_bp,
            interactive=interactive
        )

        # Close this gel's figure after ladder selection
        if fig is not None:
            plt.close(fig)

        if not calibrated:
            log("No usable ladder calibration - skipping this gel.")
            return 'Skipped'

        analyzer.assign_band_sizes()

        if not analyzer.calculate_distances_for_complete_lanes():
            return None

        return analyzer.generate_report()
        
    except Exception as e:
        log(f"Error during analysis: {e}")
        import traceback; traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Well-centric gel lane analysis pipeline')
    parser.add_argument('--masks_folder', required=True,
                        help='Folder containing mask files')
    parser.add_argument('--output_folder', required=True,
                        help='Output folder for analysis results')
    parser.add_argument('--mask_pattern', default='*.tif',
                        help='Pattern to match mask files (default: *.tif)')
    parser.add_argument('--show_plots', action='store_true',
                        help='Show plots in addition to saving them')
    parser.add_argument('--non_interactive', action='store_true',
                        help='Auto-select the ladder lane with the most bands and use --ladder_sizes with no prompts')
    parser.add_argument('--ladder_sizes',
                        help='Comma-separated ladder sizes, required when --non_interactive is set (e.g. "1000,750,500,250")')
    parser.add_argument('--no_area_filter', action='store_true',
                    help='Disable small-component area filtering')
    parser.add_argument('--no_confidence_filter', action='store_true',
                    help='Disable confidence-based filtering')

    args = parser.parse_args()

    # Set up paths
    masks_folder = args.masks_folder
    output_folder = args.output_folder

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Parse ladder sizes if provided
    ladder_sizes_bp = None
    if args.ladder_sizes:
        try:
            ladder_sizes_bp = [float(x.strip()) for x in args.ladder_sizes.split(',')]
            print(f"Using provided ladder sizes: {ladder_sizes_bp}")
        except Exception:
            print("Could not parse ladder sizes, will prompt during analysis")
    # Find all mask files matching the pattern
    mask_files = glob.glob(os.path.join(masks_folder, args.mask_pattern))

    if not mask_files:
        print(f"No mask files matching '{args.mask_pattern}' found in: {masks_folder}")
        exit(1)

    print(f"Found {len(mask_files)} masks to analyze")
    print(f"Output will be saved to: {output_folder}")

    # Process each mask
    successful = 0
    failed = 0
    log_lines = []

    in_colab = 'COLAB_RELEASE_TAG' in os.environ
    redirect_stdout = not (in_colab and not args.non_interactive)

    for i, mask_file in enumerate(mask_files, 1):
            image_name = os.path.splitext(os.path.basename(mask_file))[0]
            console(f"\n[{i}/{len(mask_files)}] Processing: {image_name}")

            # Set up output paths
            plot_path = os.path.join(output_folder, f"{image_name}_analysis.png")
            report_path = os.path.join(output_folder, f"{image_name}_report.txt")
            csv_path = os.path.join(output_folder, f"{image_name}_distances.csv")
            gel_log_path = os.path.join(output_folder, f"{image_name}_run.log")

            try:
                if redirect_stdout:
                    log_file = open(gel_log_path, 'w', encoding='utf-8')
                    sys.stdout = log_file
                else:
                    log_file = None

                results = analyze_gel_with_proper_well_centric_approach(
                    segmap_path=mask_file,
                    ladder_sizes_bp=ladder_sizes_bp,
                    renumber_lanes=True,
                    show_plot=args.show_plots,
                    save_plot_path=plot_path,
                    save_report_path=report_path,
                    interactive=not args.non_interactive,
                    use_area_filter=not args.no_area_filter,
                    use_confidence_filter=not args.no_confidence_filter
                )

                if redirect_stdout:
                    sys.stdout = sys.__stdout__
                    log_file.close()

                if results == 'Skipped':
                    log_lines.append(f"Skipped: {image_name} - no usable ladder calibration")
                    console("   Skipped: no usable ladder calibration")
                elif results:
                    if results.get('distances'):
                        df = pd.DataFrame(results['distances'])
                        df['image_name'] = image_name
                        df['size_bp'] = df['size_bp'].fillna('Outside ladder range')
                        df.to_csv(csv_path, index=False)

                    successful += 1
                    log_lines.append(f"Success: {image_name}")
                    console(f"   Done: {image_name}")
                else:
                    failed += 1
                    log_lines.append(f"Failed: {image_name} - Analysis returned None")
                    console(f"   Failed: {image_name} - Analysis returned None")

            except Exception as e:
                if redirect_stdout:
                    sys.stdout = sys.__stdout__
                    log_file.close()
                failed += 1
                log_lines.append(f"Failed: {image_name} - {str(e)}")
                console(f"   Failed: {image_name} - {str(e)}")

    # Write log file
    log_path = os.path.join(output_folder, "analysis_log.txt")
    with open(log_path, 'w') as f:
        f.write(f"Gel Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total files: {len(mask_files)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")
        f.write("Details:\n")
        f.write("-" * 20 + "\n")
        for line in log_lines:
            f.write(line + "\n")

    # Final summary
    print(f"\n{'='*60}")
    print("Batch analysis complete")
    print(f"{'='*60}")
    print(f"Processed: {successful}/{len(mask_files)} successfully")
    print(f"Results saved to: {output_folder}")
    print(f"Log saved to: {log_path}")
    if failed > 0:
        log(f" {failed} analyses failed - check log for details")     
