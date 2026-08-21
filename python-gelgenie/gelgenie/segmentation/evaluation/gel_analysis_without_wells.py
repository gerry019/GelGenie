import argparse
import glob
import numpy as np
import os
import pandas as pd
from datetime import datetime
from skimage.io import imread
from skimage.measure import regionprops, label
from sklearn.cluster import DBSCAN
from collections import defaultdict
from skimage.morphology import convex_hull_image
import matplotlib
import sys

def console(msg=""):
    """Print directly to the real terminal, bypassing any stdout redirection."""
    print(msg, file=sys.__stdout__, flush=True)


if 'google.colab' not in sys.modules:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()


class DBSCANLaneAnalyzer:
    def __init__(self, segmap_path, confidence_path=None):
        """
        segmap format: 1 = bands, 0 = background
        """
        print(f"Loading segmentation map: {segmap_path}")
        self.segmap_path = segmap_path
        self.gel_name = os.path.basename(segmap_path).removesuffix("_raw_mask.tif")
        self.segmap = imread(segmap_path) # Numpy array

        # Load confidence map if provided for reporting in csv file
        self.confidence_map = None
        if confidence_path and os.path.isfile(confidence_path):
            self.confidence_map = imread(confidence_path).astype(np.float32)  # To be used for calculations

            if self.confidence_map.shape != self.segmap.shape: # Size map check
                raise ValueError(
                    f"Shape mismatch: mask={self.segmap.shape}, "
                    f"confidence={self.confidence_map.shape}"
                )
            print(f"Loaded confidence map: {confidence_path}")
        else:
            print("No confidence map provided")

        self.use_confidence_map = self.confidence_map is not None  # auto-detected from input

        # storage / checkpoints (populated as pipeline steps run)
        self.original_bands = None       # all raw regionprops, before any filtering
        self.original_bands_mask = None  # binary mask, before any filtering
        self.original_bands_labeled = None  # labeled array matching original_bands' .label IDs
        self.removed_bands = None        # bands dropped by confidence filtering
        self.filtered_bands = None       # surviving bands after confidence + relabel
        self.filtered_bands_mask = None  # binary mask after confidence filtering
        self.bands_mask = None           # final repaired mask (post split-band repair)
        self.lane_clusters = None        # {cluster_id: [bands]} after DBSCAN + repair
        self.all_lane_axes = None        # final lane dict after estimate_lanes()
        self.repaired_bands = None       # flat list of bands after split-band repair
        self.gel_height = self.segmap.shape[0]
        self.img_width = self.segmap.shape[1]
        self.migration_origin_y = 0.0  # Takes top of image
        self.ladder_calibrations = None  # {lane_id: {curve, coeffs, migs_min, migs_max, n, known_sizes}}
        self.lane_to_ladder = None        # {lane_id: ladder_lane_id} assignment

        print(f"Loaded segmentation map: {self.segmap.shape}") # Size
        print(f"Unique labels: {np.unique(self.segmap)}") # Labels

    def extract_bands(self):
        console("Step 1: Extracting Bands")
        print("\n=== Step 1: Extracting Bands ===")

        # Extract bands from segmap using connected component labeling and gets its properties
        bands_mask = (self.segmap == 1)
        bands_labeled = label(bands_mask)
        all_bands = regionprops(bands_labeled)

        self.original_bands_mask = bands_mask
        self.original_bands_labeled = bands_labeled
        self.original_bands = all_bands

        print(f"   Found {len(all_bands)} raw band detections")
        return True
    
    def filter_bands(self):
        console("Step 2: Filtering Bands")
        print("\n=== Step 2: Filtering Bands ===")

        # Filter 1: remove low-confidence bands (mean confidence < 0.8)
        confidence_threshold = 0.8
        if self.use_confidence_map:
            confidence_filtered_bands = []
            removed_low_confidence = []
            for band in self.original_bands: # Looks up confidence-map values and gets the mean per band
                coords = band.coords
                mean_conf = float(
                    np.mean(self.confidence_map[coords[:, 0], coords[:, 1]])
                )

                if mean_conf >= confidence_threshold:
                    confidence_filtered_bands.append(band)
                else:
                    removed_low_confidence.append(band)

            if removed_low_confidence:
                print(f"   Filtered out {len(removed_low_confidence)} low-confidence bands (< {confidence_threshold})")

            self.removed_bands = removed_low_confidence
        else:
            # Using manually cleaned masks
            confidence_filtered_bands = list(self.original_bands)
            self.removed_bands = []

        # Filter 2: remove bands < 0.0075% of total image area
        total_image_area = self.segmap.shape[0] * self.segmap.shape[1]
        min_band_area = 0.000075 * total_image_area

        area_filtered_bands = [b for b in confidence_filtered_bands if b.area >= min_band_area]

        removed_small = len(confidence_filtered_bands) - len(area_filtered_bands)
        if removed_small > 0:
            print(f"   Filtered out {removed_small} small bands (< 0.0075% of image area)")

        print(f"   Found {len(area_filtered_bands)} bands (after filtering)")

        # Create a mask containing only bands that passed filtering
        filtered_mask = np.zeros_like(self.original_bands_mask, dtype=bool)

        for band in area_filtered_bands:
            filtered_mask[self.original_bands_labeled == band.label] = True

        # Use the filtered mask for any additional processing
        self.filtered_bands_mask = filtered_mask.copy()
        self.bands_mask = filtered_mask.copy()  # carried forward as the working mask
        # Relabel after removing low-confidence/small bands
        bands_labeled = label(self.filtered_bands_mask)
        # rebuild RegionProps so labels match bands_labeled
        self.filtered_bands = list(regionprops(bands_labeled))

        if not self.filtered_bands:
            console("No bands detected after filtering.")
            print("No bands detected after filtering.")
            return False

        return True
    # Step 3
    def cluster_bands(self):
        console("Step 3: Clustering Bands")
        print("\n=== Step 3: Clustering Bands ===")

        # Extract x centroid of each band
        x_centroids = np.array([band.centroid[1] for band in self.filtered_bands])
        X = x_centroids.reshape(-1, 1) # 2d array for DBSCAN

        # Median band width as base for eps
        # Bands in same lane should have x-centroids within roughly one band-width of each other
        band_widths = np.array([band.bbox[3] - band.bbox[1] for band in self.filtered_bands])
        eps_val = float(np.median(band_widths)) * 0.60 # This can be changed  per image as required
        print(f"Median band width: {np.median(band_widths):.1f}px, eps={eps_val:.1f}px")

        # DBSCAN clustering on 1D x-centroids
        # min_samples=1 because single band lanes are biologically valid
        dbscan = DBSCAN(eps=eps_val, min_samples=1)
        labels = dbscan.fit_predict(X)
        n_clusters = len(set(labels))  # No noise removal as each band is valid
        print(f"DBSCAN: {n_clusters} clusters")

        # Relabel clusters so IDs match left-to-right lane order
        label_mean_x = {
            lab: float(np.mean(x_centroids[labels == lab]))
            for lab in np.unique(labels)
        }

        sorted_labels = sorted(label_mean_x, key=label_mean_x.get)

        relabel_map = {
            old: new
            for new, old in enumerate(sorted_labels, start=1)
        }

        labels = np.array([relabel_map[lab] for lab in labels])

        # Group bands by cluster
        cluster_bands = defaultdict(list)

        for band, label_id in zip(self.filtered_bands, labels):
            cluster_bands[label_id].append(band)

        self.lane_clusters = dict(cluster_bands)
        self.eps_val = eps_val
        self.n_clusters = n_clusters

        return True
    # Step 4
    def repair_split_bands(self):
        console("Step 4: Repairing Split Bands")
        print("\n=== Step 4: Repairing Split Bands ===")

        def find_split_bands(bands):
            """
            Check all pairs of bands in a cluster for split band condition.
            ratio = y_range / x_range < 0.5 means bands are side by side = split band.
            Returns list of index  pairs that are part of a split pair.
            """
            split_pairs = []
            for i in range(len(bands)):
                for j in range(i+1, len(bands)):
                    y_i = bands[i].centroid[0]
                    y_j = bands[j].centroid[0]
                    x_i = bands[i].centroid[1]
                    x_j = bands[j].centroid[1]
                    y_range = abs(y_i - y_j)
                    x_range = abs(x_i - x_j) if abs(x_i - x_j) > 0.001 else 0.001 # avoid division by zero
                    ratio = y_range / x_range
                    if ratio < 0.5:
                        split_pairs.append((i, j))

            return split_pairs

        bands_labeled = label(self.bands_mask)  # relabeled to match current self.bands_mask
        self.merged_pixels = np.zeros_like(self.bands_mask, dtype=bool)

        for cluster_id, bands in sorted(self.lane_clusters.items()):
            if len(bands) < 2:
                continue

            # Find split band pairs using ratio check
            split_pairs = find_split_bands(bands)
            if split_pairs:
                print(f"Lane {cluster_id}: {len(split_pairs)} split band pair(s) detected")

                # Merge each split pair using a convex hull (taken from postprocessing segmentation code)
                # Create a mask containing only the current cluster
                cluster_mask = np.zeros_like(self.bands_mask, dtype=bool)
                for band in bands:
                    cluster_mask[bands_labeled == band.label] = True

                for i, j in split_pairs: # Creates a bianry mask for each fragment
                    mask_i = bands_labeled == bands[i].label
                    mask_j = bands_labeled == bands[j].label
                    # Create a single connected region enclosing both split-band fragments
                    merged_mask = convex_hull_image(mask_i | mask_j)
                    # Update both the cluster mask and the global mask
                    cluster_mask[merged_mask] = True
                    self.bands_mask[merged_mask] = True
                    self.merged_pixels[merged_mask] = True 

                # Relabel only this cluster
                cluster_labels = label(cluster_mask)
                bands = list(regionprops(cluster_labels))
                self.lane_clusters[cluster_id] = bands

        self.repaired_bands = [
            band
            for bands_in_cluster in self.lane_clusters.values()
            for band in bands_in_cluster
        ]

        return True

        # Step 5
    def estimate_lanes(self):
        console("Step 5: Estimating Lanes")
        print("\n=== Step 5: Estimating Lanes ===")

        # Can be adapted accordingly
        MIN_COVERAGE = 0.15      # bands must span at least 15% of gel height
        SLOPE_FACTOR = 3.0       # Flags slopes > 3x median normal slope
        K = 2                    # borrow from 2 nearest reliable lanes

        # Fit line through each cluster's (x,y) centroids
        # The fit x = m*y + c because lanes run vertically (y is independent variable)
        multi_band_clusters = {} # cluster with multiple bands
        single_band_clusters = {}
        weak_clusters = {} # not enough bands or not enough coverage
        candidate_slopes = []

        gel_height = self.gel_height
        migration_origin_y = self.migration_origin_y

        for cluster_id, bands in self.lane_clusters.items(): # Extract centroids
            ys = np.array([b.centroid[0] for b in bands])
            xs = np.array([b.centroid[1] for b in bands])

            if len(bands) == 1: # Store info
                single_band_clusters[cluster_id] = {
                    'n_bands': 1,
                    'mean_x': float(xs[0]),
                    'bands': bands
                }
                print(f"Cluster {cluster_id}: 1 band and so it will borrow slope from neighbours")

            else:
                # Apply small band filter
                # A band is small if its width < 50% of largest band width in cluster
                if bands: # Could be adapted accordingly
                    max_cluster_width = float(np.max([b.bbox[3] - b.bbox[1] for b in bands]))
                    bands_for_slope = [b for b in bands
                                    if (b.bbox[3] - b.bbox[1]) >= 0.5 * max_cluster_width]
                else:
                    bands_for_slope = []

                if len(bands_for_slope) >= 2:
                    ys_fit = np.array([b.centroid[0] for b in bands_for_slope])
                    xs_fit = np.array([b.centroid[1] for b in bands_for_slope])

                    coeffs = np.polyfit(ys_fit, xs_fit, 1) # for it to be a straight line
                    slope = float(coeffs[0])
                    intercept = float(coeffs[1])

                    y_span = float(ys_fit.max() - ys_fit.min()) # Vertical range to support this fit
                    coverage = y_span / gel_height
                    candidate_slopes.append(abs(slope))

                    multi_band_clusters[cluster_id] = {
                        'slope': slope,
                        'intercept': intercept,
                        'coverage': coverage,
                        'n_bands': len(bands),
                        'mean_x': float(np.mean(xs)),
                        'bands': bands,
                        'bands_for_slope': bands_for_slope
                    }

                # For 1 remaining band or no bands after filtering, so borrows slope
                elif len(bands_for_slope) == 1:
                    weak_clusters[cluster_id] = {
                        'n_bands': len(bands),
                        'mean_x': float(np.mean(xs)),
                        'bands': bands
                    }
                    print(f"Cluster {cluster_id}: 1 band after filtering and so will borrow from neighbours")

                else:
                    weak_clusters[cluster_id] = {
                        'n_bands': len(bands),
                        'mean_x': float(np.mean(xs)),
                        'bands': bands
                    }
                    print(f"Cluster {cluster_id}: all split bands and so will borrow from neighbours")

        # To use for a later print out for slopes that are highly different
        if candidate_slopes:
            median_slope = float(np.median(candidate_slopes))
            slope_limit = SLOPE_FACTOR * median_slope
        else:
            median_slope = 0.0
            slope_limit = 0.01

        print(f"\nMedian absolute slope: {median_slope:.4f}")
        print(f"Slope limit: {slope_limit:.4f}")

        for cluster_id, lane in list(multi_band_clusters.items()): #Checks for slop and output warning
            coverage_ok = lane['coverage'] >= MIN_COVERAGE
            slope_ok = abs(lane['slope']) <= slope_limit if slope_limit > 0 else True

            if not coverage_ok:
                weak_clusters[cluster_id] = lane
                del multi_band_clusters[cluster_id]
                print(
                    f"Cluster {cluster_id}: weak slope, will borrow "
                    f"(coverage={lane['coverage']:.3f}, slope={lane['slope']:.4f})"
                )

            elif not slope_ok:
                print(
                    f"Cluster {cluster_id}: unusual slope but keeping because coverage is good "
                    f"(coverage={lane['coverage']:.3f}, slope={lane['slope']:.4f})"
                )

        print(f"\nReliable multi-band clusters: {len(multi_band_clusters)}")
        print(f"Weak clusters: {len(weak_clusters)}")
        print(f"Single-band clusters: {len(single_band_clusters)}")

        # Add original single-band clusters to weak clusters so they also borrow
        for cluster_id, cluster in single_band_clusters.items():
            weak_clusters[cluster_id] = cluster

        # Borrow slopes for weak clusters rom 2 nearest multi-band clusters
        for cluster_id, cluster in weak_clusters.items():
            if multi_band_clusters: # sorts by x
                sorted_by_distance = sorted(multi_band_clusters.values(),
                                        key=lambda c: abs(c['mean_x'] - cluster['mean_x'])) # To get the distances
                k_nearest = sorted_by_distance[:K] # take the closest k
                borrowed_slope = float(np.mean([c['slope'] for c in k_nearest])) # Averages the slope of the closest reliable lanes
                anchor_bands = cluster.get('bands_for_slope', cluster['bands']) # Which bands to  use for slope

                y0 = float(np.mean([b.centroid[0] for b in anchor_bands])) # Average centroid for that cluster
                x0 = float(np.mean([b.centroid[1] for b in anchor_bands]))

                intercept = x0 - borrowed_slope * y0 # get the intercept based on the borrowed slope
                cluster['slope'] = borrowed_slope # store the results
                cluster['intercept'] = intercept
                print(f"Cluster {cluster_id}: borrowed slope={borrowed_slope:.4f} from {len(k_nearest)} nearest reliable neighbours")
            else: # If no multicluster, fall back to  vertical
                y0 = float(cluster['bands'][0].centroid[0])
                x0 = float(cluster['bands'][0].centroid[1])
                cluster['slope'] = 0.0
                cluster['intercept'] = x0 # Vertical lane
                print(f"Cluster {cluster_id}: no reliable multi-band clusters, assuming vertical")

        # Merge all clusters
        all_lane_axes = {**multi_band_clusters, **weak_clusters}

        # Sort lanes by mean x position left to right (by their horizontal positon)
        sorted_lanes = sorted(all_lane_axes.items(), key=lambda kv: kv[1]['mean_x'])

        MARGIN = 20
        # To get 5 coordinates,quarters, top, middle, bottom (this is for the lanes)
        check_ys = [0, gel_height // 4, gel_height // 2, (3 * gel_height) // 4, gel_height]

        # Calculate initial widths using largest band width + margin
        for cluster_id, lane in all_lane_axes.items():
            band_widths_cluster = [b.bbox[3] - b.bbox[1] for b in lane['bands']] # Width
            lane['max_width'] = float(np.max(band_widths_cluster)) + MARGIN
            lane['median_width'] = float(np.median(band_widths_cluster)) + MARGIN # Used later
            lane['lane_width'] = lane['max_width']

        # Trim overlaps equally from both sides
        print("\n Trim remaining overlaps")
        for i in range(len(sorted_lanes) - 1):
            id_a, lane_a = sorted_lanes[i]
            id_b, lane_b = sorted_lanes[i + 1]

            # Check overlap at multiple y positions and find worst case
            max_overlap = 0
            for y in check_ys:
                center_a = lane_a['slope'] * y + lane_a['intercept']
                center_b = lane_b['slope'] * y + lane_b['intercept']
                right_a = center_a + lane_a['lane_width'] / 2
                left_b = center_b - lane_b['lane_width'] / 2
                overlap = right_a - left_b
                if overlap > max_overlap:
                    max_overlap = overlap

            if max_overlap > 0:
                # Trim half the overlap from each lane
                trim = max_overlap
                lane_a['lane_width'] -= trim
                lane_b['lane_width'] -= trim
                print(f"Cluster {id_a} and {id_b}: trimmed {trim:.1f}px from each side (overlap was {max_overlap:.1f}px)")

        # Renumber lanes left to right (final lane numbers, replacing DBSCAN cluster IDs)
        sorted_lane_ids = [k for k, _ in sorted_lanes] # To get the IDS
        lane_number = {lid: i + 1 for i, lid in enumerate(sorted_lane_ids)} # Change the number afetr sorting

        print("\nLane numbering:")
        for cluster_id in sorted_lane_ids:
            print(f"Cluster {cluster_id} is now Lane {lane_number[cluster_id]}")

        self.all_lane_axes = {lane_number[cid]: lane for cid, lane in all_lane_axes.items()}

        return True

    #Step 6
    def visualise(self, save_path=None):
        console("Step 6: Creating Visualization")
        print("\n=== Step 6: Creating Visualization ===")
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.imshow(self.bands_mask, cmap='gray')


        gel_height = self.bands_mask.shape[0]
        img_width = self.bands_mask.shape[1]

        cmap = plt.cm.get_cmap('tab20', 20)
        y_vals = np.linspace(0, gel_height, 200)

        for lane_number, lane in self.all_lane_axes.items():
            color = cmap(lane_number % 20)
            half_width = lane['lane_width'] / 2

            # Draw band centroids as dots
            for band in lane['bands']:
                y, x = band.centroid
                ax.scatter(x, y, color=color, s=40, zorder=3) # draw a point in the centroid

            # Draw lane axis line from top to bottom of gel
            x_centers = lane['slope'] * y_vals + lane['intercept']
            ax.plot(x_centers, y_vals, color=color, linewidth=1.5, alpha=0.9, zorder=2)

            # Draw lane ROI strip edges following the axis
            x_left = np.clip(x_centers - half_width, 0, img_width)
            x_right = np.clip(x_centers + half_width, 0, img_width)

            ax.plot(x_left, y_vals, '--', color=color, linewidth=1, alpha=0.6, zorder=2)
            ax.plot(x_right, y_vals, '--', color=color, linewidth=1, alpha=0.6, zorder=2)
            ax.fill_betweenx(y_vals, x_left, x_right, alpha=0.1, color=color, zorder=1)

            # Label cluster ID at top of lane
            ax.text(x_centers[0], y_vals[0] + 15, str(lane_number),
                    fontsize=6, color='white', ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

        ax.set_title(f'Lane ROI strips (DBSCAN eps={self.eps_val:.1f}px): {len(self.all_lane_axes)} lanes')
        ax.axis('off')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=200)
            print(f"\nSaved {save_path}!")

        plt.show(block=False)
        plt.pause(0.1)

        return fig
    def euclidean_migration_distance(self, band, lane):
        """
        Calculate the Euclidean migration distance from the centre of the lane
        at the loading-well level to the centroid of a detected band.
        """

        band_y, band_x = band.centroid # Get every band centroid to measure

        # Centre of this lane at the migration origin
        origin_x = lane["slope"] * self.migration_origin_y + lane["intercept"]

        dx = band_x - origin_x # How far is x from the slope
        dy = band_y - self.migration_origin_y # vertical migration

        return np.sqrt(dx**2 + dy**2)


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
    
    # Step 7
    def calibrate(self, ladder_sizes_bp=None, interactive=True):
        console("Step 7: Ladder Calibration (local two-point interpolation)")
        print(f"\n=== Step 7: Ladder Calibration (local two-point interpolation) ===")
        # Auto-select ladder: lane with most bands (and ask user to verify)
        auto_ladder_id = max(self.all_lane_axes.keys(), key=lambda k: len(self.all_lane_axes[k]['bands']))
        print(f"\nAuto-selected ladder: Lane {auto_ladder_id}")

        if not interactive:
            if ladder_sizes_bp is None:
                console("Non-interactive mode requires ladder_sizes_bp to be provided. Skipping calibration.")
                print("Non-interactive mode requires ladder_sizes_bp to be provided. Skipping calibration.")
                self.ladder_calibrations = {}
                self.lane_to_ladder = {}
                return False
            ladder_ids = [auto_ladder_id]
            console(f"Non-interactive mode: using auto-selected Lane {auto_ladder_id} with provided sizes.")
            print(f"Non-interactive mode: using auto-selected Lane {auto_ladder_id} with provided sizes.")
        else:
            console("Use this ladder? (Y/N): ")
            answer = input().strip().lower()

            if answer in ("y", "yes", ""):
                ladder_ids = [auto_ladder_id]
            else:
                console("Enter ladder lane number(s), e.g. 1 or 1,8: ")
                raw = input().strip()
                ladder_ids = [int(x.strip()) for x in raw.split(",") if x.strip()]

        ladder_calibrations = {} # For multiple ladders

        for ladder_id in ladder_ids: # through each ladder
            ladder_lane = self.all_lane_axes[ladder_id]
            ladder_bands_sorted = sorted(ladder_lane['bands'], key=lambda b: b.centroid[0])
            n = len(ladder_bands_sorted)
            if n < 2:
                console(f"Lane {ladder_id}: too few bands, skipping.")
                print(f"Lane {ladder_id}: too few bands, skipping.")
                continue

            # Migration from middle of lane
            ladder_migs = np.array([
                self.euclidean_migration_distance(band=b, lane=ladder_lane)
                for b in ladder_bands_sorted
            ], dtype=float)

            sizes = None
            if ladder_sizes_bp is not None:
                candidate_sizes = np.array(ladder_sizes_bp, dtype=float)
                if len(candidate_sizes) == n:
                    sizes = candidate_sizes
                else:
                    print(f"Provided {len(candidate_sizes)} sizes, but Lane {ladder_id} has {n} bands.")

            while sizes is None:
                if not interactive:
                    console(f"Non-interactive mode: could not match provided sizes to {n} bands in Lane {ladder_id}. Skipping.")
                    print(f"Non-interactive mode: could not match provided sizes to {n} bands in Lane {ladder_id}. Skipping.")
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

            if not np.all(np.diff(sizes) < 0):
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

            print(f"   Local two-point log-linear interpolation calibration complete for Ladder Lane {ladder_id}. Sizes attached for {n} ladder bands.")

        if not ladder_calibrations:
            console("No usable ladder calibration.")
            print("No usable ladder calibration.")
            self.ladder_calibrations = {}
            self.lane_to_ladder = {}
            return False

        lane_to_ladder = {}

        if len(ladder_calibrations) == 1:
            only_lid = next(iter(ladder_calibrations))
            for lid in self.all_lane_axes:
                lane_to_ladder[lid] = only_lid
            print(f"\nAll lanes will use Ladder Lane {only_lid}.")
        else:
            print("\nMultiple ladders selected — assign which sample lanes use each ladder.")

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

            unassigned = [lid for lid in self.all_lane_axes if lid not in lane_to_ladder]
            if unassigned:
                print(
                    "Note: lanes "
                    f"{unassigned} were not assigned "
                    "to any ladder and will not receive size estimates."
                )

        self.ladder_calibrations = ladder_calibrations
        self.lane_to_ladder = lane_to_ladder

        return True

        # Step 8
    def measure(self):
        console("Step 8: Calculating Distances")
        print(f"\n=== Step 8: Calculating Distances ===")
        # Assign known sizes to ladder bands
        band_size_bp_by_id = {}
        for cal in self.ladder_calibrations.values():
            band_size_bp_by_id.update(cal['known_sizes'])

        # Clear outside list
        outside_ladder = []

        # For all other lanes (same as set_ladder_sizes_and_calibrate loop),
        # using whichever ladder that lane was assigned to
        for lane_id, lane in self.all_lane_axes.items():
            if lane_id not in self.lane_to_ladder:
                continue
            cal = self.ladder_calibrations[self.lane_to_ladder[lane_id]]

            for band in lane['bands']:
                bid = id(band)
                if bid in band_size_bp_by_id:
                    continue

                m = self.euclidean_migration_distance(band, lane)
                size_est = self.interpolate_size_local(m, cal['migs_sorted'], cal['log_sizes_sorted'])
                if np.isnan(size_est):
                    outside_ladder.append((lane_id, band))
                band_size_bp_by_id[bid] = size_est

        print(f"   Sizes attached for {len(band_size_bp_by_id)} bands.")

        if outside_ladder:
            print("\n   Bands outside their assigned ladder range (no size assigned):")
            for lane_id, band in outside_ladder:
                by, bx = band.centroid
                print(f"      Lane {lane_id}: band centroid at x={bx:.1f}, y={by:.1f}")

        # Distances
        distances = []
        for lane_id in sorted(self.all_lane_axes.keys()):
            lane = self.all_lane_axes[lane_id]
            bands_in_lane = lane['bands']
            print(f"\n   Lane {lane_id}: {len(bands_in_lane)} bands")

            bands_sorted = sorted(bands_in_lane, key=lambda b: b.centroid[0])  # top to bottom
            for b_idx, band in enumerate(bands_sorted, start=1):
                by, bx = band.centroid
                migration_px = self.euclidean_migration_distance(band, lane)  # Eucladian distance from the middle of the lane
                size_bp = band_size_bp_by_id.get(id(band))


                if size_bp is not None and not np.isnan(size_bp):
                    print(f"      Band {b_idx}: {migration_px:.1f}px, {int(round(size_bp))}bp")
                else:
                    print(f"      Band {b_idx}: {migration_px:.1f}px, outside ladder range")

                distances.append({
                    'lane_number': lane_id,
                    'band_idx': b_idx,
                    'band_centroid_y': float(by),
                    'band_centroid_x': float(bx),
                    'migration_px_euclidean': migration_px,
                    'size_bp': size_bp,
                    'calibration_ladder_lane': self.lane_to_ladder.get(lane_id),
                })

        self.distances = distances
        self.outside_ladder = outside_ladder
        self.band_size_bp_by_id = band_size_bp_by_id

        return True

    # Step 9
    def report(self, output_folder):
        console("Step 9: Saving Report and Distances")
        # CSV
        df = pd.DataFrame(self.distances)
        csv_path = os.path.join(output_folder, f"{self.gel_name}_distances.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n   Saved distances: {csv_path}")

        # Text report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append(" DBSCAN Lane gel Analysis Report")
        report_lines.append("="*80)
        report_lines.append(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Segmentation file: {self.segmap_path}")
        report_lines.append("")

        # Summary
        report_lines.append("Detection summary:")
        report_lines.append(f"  • Bands detected: {len(self.repaired_bands)}")
        report_lines.append(f"  • Lanes detected: {len(self.all_lane_axes)}")
        report_lines.append(f"  • Complete lanes (with bands): {len([l for l in self.all_lane_axes.values() if l['bands']])}")
        report_lines.append(f"  • Empty lanes (no bands): {len([l for l in self.all_lane_axes.values() if not l['bands']])}")
        report_lines.append("")

        # Ladder info (now supports one or more ladder lanes)
        report_lines.append("Ladder information:")
        report_lines.append(f"  • Calibration method: local two-point log-linear interpolation")
        for lid, cal in self.ladder_calibrations.items():
            report_lines.append(f"  • Ladder Lane {lid}: {cal['n']} bands, calibration=local two-point log-linear interpolation")
        if self.outside_ladder:
            report_lines.append(f"  • Bands outside assigned ladder range: {len(self.outside_ladder)}")
            for lane_id, band in self.outside_ladder:
                by, bx = band.centroid
                report_lines.append(f"    - Lane {lane_id}: band centroid at x={bx:.1f}, y={by:.1f}")
        report_lines.append("")

        # Lane details 
        report_lines.append("Lane details:")
        for lane_id in sorted(self.all_lane_axes.keys()):
            lane = self.all_lane_axes[lane_id]
            report_lines.append(f"  Lane {lane_id} ({'ladder' if lane_id in self.ladder_calibrations else 'sample'}):")
            report_lines.append(f"    - Bands: {len(lane['bands'])}")
            report_lines.append(f"    - Slope: {lane['slope']:.4f}, Intercept: {lane['intercept']:.1f}")
            report_lines.append(f"    - Lane width: {lane['lane_width']:.1f}px")
            report_lines.append(f"    - Mean x: {lane['mean_x']:.1f}")
        report_lines.append("")

        # Migration measurements
        report_lines.append("Migration measurements:")
        report_lines.append(f"  • Total bands measured: {len(self.distances)}")

        lane_groups = {}
        for dist in self.distances:
            ln = dist['lane_number']
            if ln not in lane_groups:
                lane_groups[ln] = []
            lane_groups[ln].append(dist)

        for ln in sorted(lane_groups.keys()):
            dists_in_lane = lane_groups[ln]
            report_lines.append(f"  Lane {ln} ({len(dists_in_lane)} bands):")
            for dist in dists_in_lane:
                b_idx = dist['band_idx']
                migration_px = dist['migration_px_euclidean']
                size_bp = dist.get('size_bp')

                if size_bp is not None and not np.isnan(size_bp):
                    size_str = f", {int(round(size_bp))}bp"
                else:
                    size_str = ", outside ladder range"

                report_lines.append(f"    Band {b_idx}: {migration_px:.1f}px{size_str}")
        report_lines.append("")
        report_lines.append("="*80)

        report_path = os.path.join(output_folder, f"{self.gel_name}_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"   Saved report: {report_path}")

        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DBSCAN-based gel lane analysis pipeline')
    parser.add_argument("--input_folder", required=True, help="Folder containing mask files")
    parser.add_argument("--output_folder", required=True, help="Output folder for analysis results")
    parser.add_argument("--mask_pattern", default="*_raw_mask.tif", help="Pattern to match mask files (default: *_raw_mask.tif)")
    parser.add_argument("--non_interactive", action="store_true", help="Auto-select the ladder lane with the most bands and use --ladder_sizes with no prompts")
    parser.add_argument("--ladder_sizes", help='Comma-separated ladder sizes, required when --non_interactive is set (e.g. "1000,750,500,250")')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    mask_paths = sorted(glob.glob(os.path.join(args.input_folder, args.mask_pattern)))

    # Parse ladder sizes if provided
    ladder_sizes_bp = None
    if args.ladder_sizes:
        try:
            ladder_sizes_bp = [float(x.strip()) for x in args.ladder_sizes.split(',')]
            print(f"Using provided ladder sizes: {ladder_sizes_bp}")
        except Exception:
            print("Could not parse ladder sizes, will prompt during analysis")

    successful = 0
    failed = 0
    log_lines = []

    confidence_path_by_segmap = {} # Loops through the raw mask files and locates matching confidence maps, if available

    for mask_path in mask_paths:
        confidence_path = mask_path.replace(
            "_raw_mask.tif",
            "_confidence_map.tif"
        )
        confidence_path_by_segmap[mask_path] = (
            confidence_path if os.path.isfile(confidence_path) else None
        )

    for segmap_path in mask_paths: #  Processes each mask
        gel_name = os.path.basename(segmap_path).removesuffix("_raw_mask.tif") #Removes suffix
        console(f"\nProcessing: {gel_name}")

        gel_log_path = os.path.join(args.output_folder, f"{gel_name}_run.log")

        try:
            confidence_path = confidence_path_by_segmap[segmap_path]
            log_file = open(gel_log_path, 'w', encoding='utf-8')
            sys.stdout = log_file

            analyzer = DBSCANLaneAnalyzer(segmap_path, confidence_path=confidence_path)
            analyzer.extract_bands()

            if not analyzer.filter_bands():
                log_lines.append(f"Skipped: {gel_name} - no bands detected after filtering")
                sys.stdout = sys.__stdout__
                log_file.close()
                console(f"   Skipped: {gel_name} - no bands detected after filtering")
                continue

            analyzer.cluster_bands()
            analyzer.repair_split_bands()
            analyzer.estimate_lanes()
            fig = analyzer.visualise(save_path=os.path.join(args.output_folder, f"{gel_name}_lanesnew.png"))
            calibrated = analyzer.calibrate(
                ladder_sizes_bp=ladder_sizes_bp,
                interactive=not args.non_interactive
            )

            plt.close(fig)

            if not calibrated:
                log_lines.append(f"Skipped: {gel_name} - no usable ladder calibration")
                sys.stdout = sys.__stdout__
                log_file.close()
                console(f"   Skipped: {gel_name} - no usable ladder calibration")
                continue

            analyzer.measure()
            analyzer.report(output_folder=args.output_folder)

            successful += 1
            log_lines.append(f"Success: {gel_name}")
            sys.stdout = sys.__stdout__
            log_file.close()
            console(f"   Done: {gel_name}")

        except Exception as e:
            sys.stdout = sys.__stdout__
            log_file.close()
            failed += 1
            log_lines.append(f"Failed: {gel_name} - {str(e)}")
            console(f"   Failed: {gel_name} - {str(e)}")
            continue
        print()  # blank line between gels for readability

    # Write log file
    log_path = os.path.join(args.output_folder, "analysis_log.txt")
    with open(log_path, 'w') as f:
        f.write(f"Gel Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total files: {len(mask_paths)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")
        f.write("Details:\n")
        f.write("-" * 20 + "\n")
        for line in log_lines:
            f.write(line + "\n")    

    print(f"\nProcessed: {successful}/{len(mask_paths)} successfully")
    if failed > 0:
        print(f" {failed} gel(s) failed - see below")
        for line in log_lines:
            if line.startswith("FAILED"):
                print(f"   {line}")