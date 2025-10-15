import numpy as np
import argparse
import glob # For file matching
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import regionprops, label
from scipy.spatial.distance import euclidean
from matplotlib import patches # For drawing shapes


# Core clustering (gelgenie code)
def find_nearest_points(all_centroids, median_width):
    """Find nearest neighbors for clustering)."""
    nearest_centroid_indices = []
    for self_index, sel_centroid in enumerate(all_centroids):
        distance_y = np.array([a[0] - sel_centroid[0] for a in all_centroids])  # vertical
        distance_x = np.array([a[1] - sel_centroid[1] for a in all_centroids])  # horizontal
        # indices (only) of wells within half well-width horizontally 
        valid_indices = np.where(np.abs(distance_x) < (median_width / 2))[0]
        distances = np.sqrt(distance_y[valid_indices] ** 2 + distance_x[valid_indices] ** 2) # Eucladian (better for any misaligmnet)
        if len(valid_indices) == 1:
            nearest_point = self_index # itself
        else:
            # Sort by the eucladian distance
            nearest_point = valid_indices[np.argsort(distances)[1]] # finds the second one, which is index 1 adter sorting as 0 would be itself
        nearest_centroid_indices.append(nearest_point)
    return nearest_centroid_indices

def separate_into_columns(nearest_centroid_list):
    """Group centroids into columns via 'friend-of-nearest-friend' linking."""
    group_indices = []
    for i, cent_friend in enumerate(nearest_centroid_list):
        a_group = [i] # new group with current centroid
        while cent_friend not in a_group:
            a_group.append(cent_friend)
            cent_friend = nearest_centroid_list[cent_friend]
        group_indices.append(a_group)
    # de-duplicate overlapping groups
    idx = 0
    while idx < len(group_indices):
        current_array = group_indices[idx]
        for j in range(idx + 1, len(group_indices)):
            if any(item in current_array for item in group_indices[j]): # checks for overlap
                current_array = list(set(current_array) | set(group_indices[j]))
                group_indices[idx] = current_array
                group_indices.pop(j)
                idx -= 1
                break
        idx += 1
    return group_indices


# Well-centric analysis
class WellCentricLaneAnalyzer:
    def __init__(self, segmap_path, confidence_path=None):
        """
        segmap format: 2 = wells, 1 = bands 0 = background
        """
        print(f"Loading segmentation map: {segmap_path}")
        self.segmap = imread(segmap_path)

        # Load confidence map if provided
        self.confidence_map = None
        if confidence_path and os.path.exists(confidence_path):
            self.confidence_map = imread(confidence_path).astype(np.float32)
            print(f"Loaded confidence map: {confidence_path}")
        else:
            print("No confidence map provided")

        # storage
        self.wells = []
        self.bands = []
        self.well_groups = []
        self.extended_lanes = {}   # {lane_id: {..., 'bbox': (top,left,bottom,right), 'wells':[], 'bands':[]}} (dictionary)
        self.complete_lanes = {}
        self.incomplete_lanes = {}
        self.distances = []

        # params - reflect well width approach
        self.median_well_width = None
        self.horizontal_threshold = None
        self.gel_height = self.segmap.shape[0]

        # ladder
        self.ladder_lane_id = None
        self.band_size_bp_by_id = {}   # id(band) and size_bp (from interpolation)
        self.outside_ladder = []        # list of (lane_id, band) with NaN size

        print(f"Loaded segmentation map: {self.segmap.shape}") # Size
        print(f"Unique labels: {np.unique(self.segmap)}") # Labels

    # Step 1: Extract well and band regions
    def extract_wells_and_bands(self):
        print("\n=== STEP 1: Extracting Wells and Bands ===")
        wells_mask = (self.segmap == 2) # Binary mask
        if np.any(wells_mask): # Check if any exist
            wells_labeled = label(wells_mask)
            self.wells = regionprops(wells_labeled) # Get region properties
            print(f"Found {len(self.wells)} wells")
        else:
            print("No wells found (label=2)")
            return False

        bands_mask = (self.segmap == 1) # Bands binary mask
        if np.any(bands_mask): # Checks if any exist
            bands_labeled = label(bands_mask)
            self.bands = regionprops(bands_labeled) # Get region properties
            print(f"Found {len(self.bands)} bands")
        else:
            print("No bands found (label=1)")
        return True

    # Step 2: Analyze well width
    def calculate_well_parameters(self):
        print("\n === STEP 2: Analyzing Well Layout ===")
        if len(self.wells) == 1: # For only one well scenario
            print(" Only one well detected — treating as single lane.")
            c = self.wells[0].centroid
            self.median_well_width = self.segmap.shape[1]  # Treat the whole width as filler to avoid errors downstream
            self.horizontal_threshold = self.median_well_width / 2
            self.well_groups = [{
                'group_id': 0,
                'wells': [self.wells[0]],
                'well_indices': [0],
                'center_x': c[1], 'center_y': c[0],
                'width': self.median_well_width,
                'well_count': 1
            }]
            return True

        if len(self.wells) < 1:
            print(" No wells detected")
            return False

        # Use well widths between wells to check for any other
        well_widths = [w.bbox[3] - w.bbox[1] for w in self.wells]  # width of each well
        self.median_well_width = np.median(well_widths)
        self.horizontal_threshold = self.median_well_width / 2  # using well width for threshold
        print(f"   • Median well width: {self.median_well_width:.1f} px")
        print(f"   • Horizontal threshold: {self.horizontal_threshold:.1f} px")
        return True

    # Step 3: Cluster wells into lane groups
    def cluster_wells_into_lane_groups(self):
        print("\n === STEP 3: Grouping Wells Into Lane Clusters ===")
        # Clusterning and columns based on wells
        well_centroids = [w.centroid for w in self.wells]
        nearest_indices = find_nearest_points(well_centroids, self.median_well_width)
        well_groups = separate_into_columns(nearest_indices)

        self.well_groups = []
        for group_idx, well_indices in enumerate(well_groups):
            if not well_indices: continue
            group_wells = [self.wells[i] for i in well_indices]
            centroids_array = np.array([w.centroid for w in group_wells])
            cx = float(np.mean(centroids_array[:, 1])) # center x coordinate
            cy = float(np.mean(centroids_array[:, 0])) # center y coordinate
            min_x, max_x = np.min(centroids_array[:, 1]), np.max(centroids_array[:, 1]) # For the x coordinates
            width = max(max_x - min_x, self.horizontal_threshold) # To ensure minimum width
            self.well_groups.append({
                'group_id': group_idx,
                'wells': group_wells,
                'well_indices': well_indices,
                'center_x': cx, 'center_y': cy,
                'width': width, 'well_count': len(group_wells)
            })

        print(f"   Created {len(self.well_groups)} well groups.")
        return True

    # Step 4: Build extended lane boxes from wells
    def create_extended_lanes_from_wells(self):
        print("\n === STEP 4: Creating Extended Lane Boundaries ===")
        self.extended_lanes = {}
        for group in self.well_groups:
            gid = group['group_id']
            gw = group['wells']
            cx = group['center_x']
            lane_half_width = max(group['width'] / 2, self.horizontal_threshold) # width
            top = float(min(w.bbox[0] for w in gw))
            bottom = float(self.gel_height) # not good for when there are two runs in one image
            left_actual = float(min(w.bbox[1] for w in gw))
            right_actual = float(max(w.bbox[3] for w in gw))
            left = min(left_actual, cx - lane_half_width) # Most left point or center point minus half the group span (whichever is smaller)
            right = max(right_actual, cx + lane_half_width)  # Most right point or center point plus half the group span (whichever is larger)
            self.extended_lanes[gid] = {
                'group_id': gid, 'wells': gw,
                'center_x': cx, 'center_y': group['center_y'],
                'bbox': (top, left, bottom, right),
                'width': right - left, 'height': bottom - top,
                'bands': []
            }
        print(f"  Created {len(self.extended_lanes)} extended lanes.")
        return True

    # Step 5: Assign bands to lanes by bounding box
    def assign_bands_to_extended_lanes(self):
        print("\n === STEP 5: Assigning Bands to Extended Lanes ===")
        assigned = set()
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

        # classify
        self.complete_lanes = {lid: ln for lid, ln in self.extended_lanes.items() if ln['bands']}
        self.incomplete_lanes = {lid: ln for lid, ln in self.extended_lanes.items() if not ln['bands']}
        unassigned = len(self.bands) - len(assigned)
        if unassigned > 0:
            print(f"   Unassigned bands: {unassigned} (outside all lane boundaries)")
        return True

    # ---- Step 5.5: Renumber ALL lanes left→right starting from 1 ----
    def renumber_lanes_left_to_right(self):
        if not self.extended_lanes:
            return
        # sort by left x of bbox - include ALL lanes (complete + incomplete)
        sorted_items = sorted(self.extended_lanes.items(), key=lambda kv: kv[1]['bbox'][1])
        old_to_new = {old_id: new_id+1 for new_id, (old_id, _) in enumerate(sorted_items)}  # Start from 1

        def remap_lanes_dict(d): # Used to update the existing dictionaries
            return {old_to_new[k]: v for k, v in d.items()} 

        self.extended_lanes = remap_lanes_dict(self.extended_lanes)
        self.complete_lanes = remap_lanes_dict(self.complete_lanes)
        self.incomplete_lanes = remap_lanes_dict(self.incomplete_lanes)

        # update group_id and keep order-consistent
        for lid, lane in self.extended_lanes.items():
            lane['group_id'] = lid

        print("\n   All lanes renumbered left→right (starting from 1).")

    # Step 6: Auto-pick ladder AFTER renumbering from the complete lanes
    def auto_select_ladder_lane(self):
        if self.complete_lanes and self.ladder_lane_id is None:
            self.ladder_lane_id = max(self.complete_lanes.keys(), key=lambda k: len(self.complete_lanes[k]['bands'])) # with the most bands ( can add more)
            print(f"   (Auto) selected ladder lane: {self.ladder_lane_id}")

    # Ladder sizes → linear interpolation
    def _lane_origin_y(self, lane):
        """Per-lane origin for migration: mean of well centroids (y)."""
        return float(np.mean([w.centroid[0] for w in lane['wells']])) if lane['wells'] else 0.0

    # Step 7- ladder migration and interpolation from pixels to bp
    def set_ladder_sizes_and_interpolate(self, ladder_sizes_bp=None):
        """
        Linear interpolation from ladder migration (px), size (bp)
        - If ladder_sizes_bp is None, prompts the user for N sizes (top to bottom).
        - Stores results in self.band_size_bp_by_id[id(band)] = size_bp
        - Bands outside ladder range get NaN size (no extrapolation)
        """
        if self.ladder_lane_id is None or self.ladder_lane_id not in self.complete_lanes:
            print("No ladder lane available for interpolation.")
            return False

        ladder_lane = self.complete_lanes[self.ladder_lane_id]
        ladder_bands_sorted = sorted(ladder_lane['bands'], key=lambda b: b.centroid[0])  # top to bottom
        n = len(ladder_bands_sorted)
        if n < 2:
            print(" Ladder lane has fewer than 2 bands; cannot interpolate.")
            return False

        # ladder migrations from origin
        y0 = self._lane_origin_y(ladder_lane)
        ladder_migs = np.array([float(b.centroid[0] - y0) for b in ladder_bands_sorted], dtype=float)

        # get/ask sizes
        if ladder_sizes_bp is None:
            print(f"\nLadder lane {self.ladder_lane_id} has {n} bands (top to bottom).")
            raw = input(f"Enter {n} ladder sizes in bp (comma-separated, top to bottom): ").strip()
            try:
                # Takes input, checks for ;, removeswhite spaces and converts to float
                ladder_sizes_bp = [float(x) for x in raw.replace(';', ',').split(',') if x.strip() != ""]
            except Exception:
                print("Could not parse sizes.")
                return False

        sizes = np.array(ladder_sizes_bp, dtype=float)
        if len(sizes) != n:
            print(f" Provided {len(sizes)} sizes, but detected {n} ladder bands.")
            return False

        if not np.all(np.diff(sizes) < 0): # checks for negative difference
            print("Note: ladder sizes are not strictly decreasing top to bottom. Proceeding anyway.")

        # Assign sizes to ladder bands
        self.band_size_bp_by_id = {id(b): float(s) for b, s in zip(ladder_bands_sorted, sizes)}

        # Always use NaN for bands outside ladder range (no extrapolation)
        left_val = np.nan
        right_val = np.nan

        # Clear outside list
        self.outside_ladder = []

        # For the other lanes
        for lane_id, lane in self.extended_lanes.items():
            y0_lane = self._lane_origin_y(lane)
            for band in lane['bands']:
                bid = id(band)
                if bid in self.band_size_bp_by_id:
                    continue
                m = float(band.centroid[0] - y0_lane)
                # Interpolation and top and bottom outside of ladder bands
                size_est = float(np.interp(m, ladder_migs, sizes, left=left_val, right=right_val)) 
                self.band_size_bp_by_id[bid] = size_est # To save the above result
                if np.isnan(size_est):
                    self.outside_ladder.append((lane_id, band))

        print(f" Linear interpolation complete. Sizes attached for {len(self.band_size_bp_by_id)} bands.")
        if self.outside_ladder:
            print("\n Bands outside ladder range (no size assigned):")
            for lane_id, band in self.outside_ladder:
                by, bx = band.centroid
                print(f"   Lane {lane_id}: band centroid at x={bx:.1f}, y={by:.1f}")
        return True

    # Step 8: Distances + print px & bp, maybe combine with previous one  
    def calculate_distances_for_complete_lanes(self):
        print("\n === STEP 7: Calculating Distances ===")
        if not self.complete_lanes:
            print(" No complete lanes for distance calculations")
            return False

        self.distances = []
        for lane_id, lane in sorted(self.complete_lanes.items()):
            wells_in_lane = lane['wells']
            bands_in_lane = lane['bands']
            print(f"\nLane {lane_id}: {len(wells_in_lane)} wells × {len(bands_in_lane)} bands")
            
            for w_idx, well in enumerate(wells_in_lane, start=1):
                # Initialise metrics
                well_conf_mean = None
                well_conf_std = None
                if self.confidence_map is not None:
                    # Get pixels belonging to this specific well
                    well_coords = well.coords  # Array of (y, x) coordinates
                    # pixel for each y,x position
                    well_confidences = self.confidence_map[well_coords[:, 0], well_coords[:, 1]]
                    if len(well_confidences) > 0: # Ensures well is not empty
                        # Mean and std of the wells which may be combines
                        well_conf_mean = float(np.mean(well_confidences))
                        well_conf_std = float(np.std(well_confidences))

                bands_sorted = sorted(bands_in_lane, key=lambda b: b.centroid[0])  # top to bottom
                for b_idx, band in enumerate(bands_sorted, start=1):
                    wy, wx = well.centroid
                    by, bx = band.centroid
                    # downward only
                    if by <= wy:
                        continue
                    vertical_dist = float(by - wy)  # migration we print
                    size_bp = self.band_size_bp_by_id.get(id(band))

                    # For bands confidence
                    band_conf_mean = None
                    band_conf_std = None
                    if self.confidence_map is not None:
                        # Get pixels belonging to this specific band
                        band_coords = band.coords
                        band_confidences = self.confidence_map[band_coords[:, 0], band_coords[:, 1]]
                        if len(band_confidences) > 0:
                            band_conf_mean = float(np.mean(band_confidences))
                            band_conf_std = float(np.std(band_confidences))
                    if size_bp is not None and not np.isnan(size_bp):
                        conf_str = f", conf: W={well_conf_mean:.3f} B={band_conf_mean:.3f}" if well_conf_mean else ""
                        print(f"   Well {w_idx} → Band {b_idx}: {vertical_dist:.1f}px, {int(round(size_bp))}bp{conf_str}")
                    else:
                        conf_str = f", conf: W={well_conf_mean:.3f} B={band_conf_mean:.3f}" if well_conf_mean else ""
                        print(f"   Well {w_idx} → Band {b_idx}: {vertical_dist:.1f}px{conf_str}")

                    self.distances.append({
                        'lane_id': lane_id,
                        'well_idx': w_idx,
                        'band_idx': b_idx,
                        'well_centroid_y': float(wy),    # Separate key for y coordinate
                        'well_centroid_x': float(wx),    # Separate key for x coordinate
                        'band_centroid_y': float(by),
                        'band_centroid_x': float(bx),
                        'vertical': vertical_dist,
                        'size_bp': size_bp,
                        'well_confidence_mean': well_conf_mean,
                        'well_confidence_std': well_conf_std,
                        'band_confidence_mean': band_conf_mean,
                        'band_confidence_std': band_conf_std
                    })
        return True

    # Step 9: Simple visualization
    def create_visualization(self, save_path=None):
        """Create visualization with option to save instead of show if a path is provided"""
        print("\n === STEP 8: Creating Visualization ===")
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan', 'lime', 'yellow']

        # Plot 1: original seg map
        ax1 = axes[0]
        # using rainbow for colour mapping
        ax1.imshow(self.segmap, cmap='nipy_spectral')
        ax1.set_title('Segmentation (1=bands, 2=wells)')
        ax1.axis('off') # Does not show axes

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
            ax2.text(left+5, top+15, f'Lane {lane_id}', color='k',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
        if self.ladder_lane_id and self.ladder_lane_id in self.extended_lanes:
            # Get the lader lane bounding box and adds another box on it
            lbbox = self.extended_lanes[self.ladder_lane_id]['bbox']
            top, left, bottom, right = lbbox  # unpacking the lbbox
            rect = patches.Rectangle((left, top), right-left, bottom-top,
                                    linewidth=4, edgecolor='yellow', facecolor='none', alpha=0.9)
            ax2.add_patch(rect)
            ax2.text(left + 10, top + 40, f'Ladder (Lane {self.ladder_lane_id})',
                    bbox=dict(boxstyle="round,pad=0.35", facecolor='yellow', alpha=0.9),
                    fontsize=12, fontweight='bold', color='black')
        ax2.set_title('Extended Lanes (numbered 1, 2, 3...)')
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
        ax3.set_title('Downward Migrations')
        ax3.axis('off')

        # Fixes spacing between plots
        plt.tight_layout()
    
        # Save or show
        if save_path:
            # High resolution and trim extra white spaces
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
            print(f" Visualization saved: {save_path}")
        else:
            plt.show()

    def save_detailed_report(self, save_path):
        """Generate and save a detailed text report"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append(" WELL-CENTRIC GEL ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Segmentation file: {getattr(self, 'segmap_path', 'Unknown')}")
        report_lines.append("")

        # Summary statistics
        report_lines.append("DETECTION SUMMARY:")
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
        if self.ladder_lane_id:
            report_lines.append("LADDER INFORMATION:")
            report_lines.append(f"  • Ladder lane: {self.ladder_lane_id}")
            ladder_bands = len(self.complete_lanes[self.ladder_lane_id]['bands'])
            report_lines.append(f"  • Ladder bands: {ladder_bands}")
            if self.outside_ladder:
                # Print the count of any bands outside the range
                report_lines.append(f"  • Bands outside ladder range: {len(self.outside_ladder)}")
            report_lines.append("")

        # Lane details
        report_lines.append("LANE DETAILS:")
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
            report_lines.append("MIGRATION MEASUREMENTS:")
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
                    w_idx, b_idx = dist['well_idx'] + 1, dist['band_idx'] + 1
                    vertical_dist = dist['vertical']
                    size_bp = dist.get('size_bp')

                    # Confidence values
                    well_conf = dist.get('well_confidence_mean')
                    band_conf = dist.get('band_confidence_mean')

                    if size_bp is not None and not np.isnan(size_bp):
                        size_str = f", {int(round(size_bp))}bp"
                    else:
                        size_str = ""
                    
                    # Output condifence
                    if well_conf is not None and band_conf is not None:
                        conf_str = f", conf: W={well_conf:.3f} B={band_conf:.3f}"
                    else:
                        conf_str = ""

                    report_lines.append(f"    Well {w_idx} → Band {b_idx}: {vertical_dist:.1f}px{size_str}{conf_str}")
            report_lines.append("")

        report_lines.append("="*80)

        report_text = "\n".join(report_lines)

        with open(save_path, 'w',encoding='utf-8') as f:
            f.write(report_text)
        print(f" Detailed report saved: {save_path}")

    # ---- Report ----
    def generate_report(self):
        print("\n" + "="*80)
        print(" WELL-CENTRIC GEL ANALYSIS REPORT")
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
            'ladder_lane_id': self.ladder_lane_id,
            'outside_ladder_count': len(self.outside_ladder),
        }


def analyze_gel_with_proper_well_centric_approach(
    segmap_path,
    confidence_path=None, 
    ladder_lane_id=None,
    ladder_sizes_bp=None,
    renumber_lanes=True,
    show_plot=True,
    save_plot_path=None,
    save_report_path=None
):
    analyzer = WellCentricLaneAnalyzer(segmap_path, confidence_path=confidence_path)
    analyzer.segmap_path = segmap_path  # Store for reporting
    
    try:
        if not analyzer.extract_wells_and_bands(): return None
        if not analyzer.calculate_well_parameters(): return None
        if not analyzer.cluster_wells_into_lane_groups(): return None
        if not analyzer.create_extended_lanes_from_wells(): return None
        if not analyzer.assign_bands_to_extended_lanes(): return None

        if renumber_lanes:
            analyzer.renumber_lanes_left_to_right()

        analyzer.auto_select_ladder_lane()

        if (ladder_lane_id is not None) and (ladder_lane_id in analyzer.complete_lanes):
            analyzer.ladder_lane_id = ladder_lane_id
            print(f"   (Override) ladder lane set to: {analyzer.ladder_lane_id}")

        analyzer.set_ladder_sizes_and_interpolate(ladder_sizes_bp=ladder_sizes_bp)

        if not analyzer.calculate_distances_for_complete_lanes(): return None

        # Handle visualization
        if show_plot or save_plot_path:
            analyzer.create_visualization(save_path=save_plot_path)

        # Handle report saving
        if save_report_path:
            analyzer.save_detailed_report(save_path=save_report_path)

        return analyzer.generate_report()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback; traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch gel analysis pipeline')
    parser.add_argument('--masks_folder', required=True, 
                        help='Folder containing mask files')
    parser.add_argument('--output_folder', required=True,
                        help='Output folder for analysis results')
    parser.add_argument('--mask_pattern', default='*.tif',
                        help='Pattern to match mask files (default: *.tif)')
    parser.add_argument('--ladder_sizes', 
                        help='Comma-separated ladder sizes (e.g. "1000,750,500,250")')
    parser.add_argument('--show_plots', action='store_true',
                        help='Show plots in addition to saving them')

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
        except:
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
    for i, mask_file in enumerate(mask_files, 1):
        # Get image name by removing file extension
        image_name = os.path.splitext(os.path.basename(mask_file))[0]
        print(f"\n[{i}/{len(mask_files)}] Processing: {image_name}")

        # Set up output paths
        plot_path = os.path.join(output_folder, f"{image_name}_analysis.png")
        report_path = os.path.join(output_folder, f"{image_name}_report.txt")
        csv_path = os.path.join(output_folder, f"{image_name}_distances.csv")

        try:
            # Analyze with per-image ladder prompting
            results = analyze_gel_with_proper_well_centric_approach(
                segmap_path=mask_file,
                ladder_lane_id=None,
                ladder_sizes_bp=ladder_sizes_bp,
                renumber_lanes=True,
                show_plot=args.show_plots,
                save_plot_path=plot_path,
                save_report_path=report_path
            )

            if results:
                # Save CSV
                if results.get('distances'):
                    df = pd.DataFrame(results['distances'])
                    df['image_name'] = image_name
                    df.to_csv(csv_path, index=False)

                successful += 1
                log_lines.append(f"SUCCESS: {image_name}")
                print(f"   Saved: plot, report, distances")
            else:
                failed += 1
                log_lines.append(f"FAILED: {image_name} - Analysis returned None")

        except Exception as e:
            failed += 1
            log_lines.append(f"FAILED: {image_name} - {str(e)}")
            print(f"   Error: {str(e)}")

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
    print("BATCH ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {successful}/{len(mask_files)} successfully")
    print(f"Results saved to: {output_folder}")
    print(f"Log saved to: {log_path}")
    if failed > 0:
        print(f" {failed} analyses failed - check log for details")