/**
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
 */

package qupath.ext.gelgenie.tools;

import org.locationtech.jts.geom.Geometry;
import qupath.ext.gelgenie.GelGenieClasses;
import qupath.ext.gelgenie.GelMeasurements;
import qupath.ext.gelgenie.ui.GelGeniePrefs;
import qupath.lib.geom.Point2;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.GeometryTools;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.scripting.QP;

import java.util.*;

public class BandSorter {

    /**
     * Minimum band area as a fraction of the whole image area; bands below this are treated as too
     * small and flagged as {@link GelGenieClasses#FILTERED_BAND} rather than assigned to a lane.
     * Matches the 0.0075% threshold in the Python {@code DBSCANLaneAnalyzer.filter_bands}.
     */
    private static final double MIN_BAND_AREA_FRACTION = 0.000075;

    /**
     * Labels a collection of annotations (wells and/or bands).
     *
     * <p>Bands that are too small are first flagged (see {@link #filterSmallBands}) and set aside. If
     * any wells remain, lanes are anchored to wells like the Python well-centric analyzer: each well
     * defines a lane column extending down to the next overlapping well (or the image bottom), and
     * bands whose centroid falls inside join that lane. Bands with no well mate (stragglers) are
     * clustered into their own lanes by DBSCAN. Finally all lanes are renumbered left-to-right, so
     * wells become {@code W1, W2, ...} and bands {@code L<lane>-<band>} top-to-bottom, and a fitted
     * axis line is drawn per lane.
     *
     * <p>If no wells are present, everything goes through the DBSCAN lane finder, mirroring the Python
     * {@code DBSCANLaneAnalyzer}. Split-band repair is a separate, user-triggered step
     * ({@link #RepairSplitBands(Collection)}).
     */
    public static void LabelBands(Collection<PathObject> annotations){
        List<PathObject> wells = new ArrayList<>();
        List<PathObject> candidateBands = new ArrayList<>();
        for (PathObject annot : annotations) {
            if (GelGenieClasses.WELL.matches(annot)) {
                wells.add(annot);
            } else if (GelGenieClasses.isBandOrFiltered(annot)) {
                candidateBands.add(annot);
            }
        }

        List<PathObject> bands = filterSmallBands(candidateBands);

        // key decision tree start - wells or no wells
        if (wells.isEmpty()) {
            labelBandsByDbscan(bands);
        } else {
            labelBandsByWellLanes(wells, bands);
        }
    }

    /**
     * Post-processing step (triggered by the user, separate from labelling): within each already
     * labelled lane, merge horizontally-split fragments of a band into one annotation via the convex
     * hull of their union, mirroring the Python {@code repair_split_bands}. Lanes are read back from
     * the existing {@code LaneID} labels, so this only makes sense after {@link #LabelBands}. Bands are
     * re-labelled and the lane axes redrawn afterwards.
     */
    public static void RepairSplitBands(Collection<PathObject> annotations) {
        List<Lane> lanes = lanesFromLabels(annotations);

        List<PathObject> mergedToAdd = new ArrayList<>();
        List<PathObject> fragmentsToRemove = new ArrayList<>();
        List<Lane> repaired = new ArrayList<>();
        for (Lane lane : lanes) {
            List<PathObject> mergedBands = repairSplitBands(lane.bands, mergedToAdd, fragmentsToRemove);
            repaired.add(new Lane(lane.well, mergedBands));
        }

        if (!fragmentsToRemove.isEmpty()) {
            QP.removeObjects(fragmentsToRemove, false);
        }
        if (!mergedToAdd.isEmpty()) {
            QP.addObjects(mergedToAdd);
        }

        finalizeLanes(repaired);   // re-name bands and redraw lane axes to reflect the merges
    }

    /** Script-friendly split-band repair over every annotation in the current image. */
    public static void RepairSplitBands() {
        RepairSplitBands(QP.getAnnotationObjects());
    }

    /** Reconstructs lanes from the {@code LaneID} measurements already written by {@link #LabelBands}. */
    private static List<Lane> lanesFromLabels(Collection<PathObject> annotations) {
        Map<Integer, PathObject> wellByLane = new HashMap<>();
        Map<Integer, List<PathObject>> bandsByLane = new HashMap<>();
        for (PathObject annot : annotations) {
            Integer lane = laneIdOf(annot);
            if (lane == null) {
                continue;
            }
            if (GelGenieClasses.WELL.matches(annot)) {
                wellByLane.put(lane, annot);
            } else if (GelGenieClasses.GEL_BAND.matches(annot)) {
                bandsByLane.computeIfAbsent(lane, k -> new ArrayList<>()).add(annot);
            }
        }

        Set<Integer> laneIds = new TreeSet<>();
        laneIds.addAll(wellByLane.keySet());
        laneIds.addAll(bandsByLane.keySet());

        List<Lane> lanes = new ArrayList<>();
        for (Integer lane : laneIds) {
            lanes.add(new Lane(wellByLane.get(lane), bandsByLane.getOrDefault(lane, new ArrayList<>())));
        }
        return lanes;
    }

    /** The object's {@code LaneID} as an int, or {@code null} if it carries no lane label. */
    private static Integer laneIdOf(PathObject pathObject) {
        Number laneId = pathObject.getMeasurements().get(GelMeasurements.LANE_ID);
        return laneId == null ? null : laneId.intValue();
    }

    /**
     * Splits candidate bands into those large enough to keep and those too small. Small bands are
     * re-classed to {@link GelGenieClasses#FILTERED_BAND} and unnamed (kept for inspection, excluded
     * from lanes); large bands are (re)set to {@link GelGenieClasses#GEL_BAND} so a band that grew, or
     * that was previously filtered, is reconsidered on every run.
     *
     * @return the bands that passed the size filter
     */
    private static List<PathObject> filterSmallBands(List<PathObject> candidateBands) {
        List<PathObject> kept = new ArrayList<>();
        double imageArea = imageArea();
        double minArea = Double.isNaN(imageArea) ? Double.NaN : MIN_BAND_AREA_FRACTION * imageArea;

        for (PathObject band : candidateBands) {
            // If the image size is unknown we cannot threshold, so keep every band.
            if (!Double.isNaN(minArea) && band.getROI().getArea() < minArea) {
                band.setPathClass(GelGenieClasses.FILTERED_BAND);
                band.setName(null);
            } else {
                band.setPathClass(GelGenieClasses.GEL_BAND.getPathClass());
                kept.add(band);
            }
        }
        return kept;
    }

    /**
     * A single lane: an optional anchoring well plus the bands assigned to it. Straggler lanes (from
     * DBSCAN) have no well. {@link #sortX()} gives the horizontal position used to order lanes
     * left-to-right during the final renumbering.
     */
    private static final class Lane {
        final PathObject well;          // nullable - straggler lanes have no well
        final List<PathObject> bands;

        Lane(PathObject well, List<PathObject> bands) {
            this.well = well;
            this.bands = bands;
        }

        /** Horizontal anchor: the well centroid if present, otherwise the mean band centroid. */
        double sortX() {
            if (well != null) {
                return well.getROI().getCentroidX();
            }
            double sum = 0.0;
            for (PathObject band : bands) {
                sum += band.getROI().getCentroidX();
            }
            return bands.isEmpty() ? 0.0 : sum / bands.size();
        }
    }

    /**
     * Well-centric labelling (mirrors the Python {@code WellCentricLaneAnalyzer}): each well anchors
     * one lane and claims the bands whose centroid falls inside its lane box. Bands left without a
     * well mate are clustered into straggler lanes by DBSCAN. All lanes are then handed to
     * {@link #finalizeLanes} for split-band repair and a single left-to-right renumbering.
     */
    private static void labelBandsByWellLanes(List<PathObject> wells, List<PathObject> bands) {
        double imageHeight = getImageHeight();

        // Wells anchor lanes, ordered left-to-right.
        wells.sort(new CentroidCompareX());

        Set<PathObject> assignedBands = new HashSet<>();
        List<Lane> lanes = new ArrayList<>();
        for (PathObject well : wells) {
            double left = well.getROI().getBoundsX();
            double right = left + well.getROI().getBoundsWidth();
            // Lane starts at the bottom edge of the well.
            double laneTop = well.getROI().getBoundsY() + well.getROI().getBoundsHeight();

            // Lane bottom = top of the next well below that overlaps horizontally, else image bottom.
            double laneBottom = imageHeight;
            for (PathObject other : wells) {
                if (other == well) {
                    continue;
                }
                double otherLeft = other.getROI().getBoundsX();
                double otherRight = otherLeft + other.getROI().getBoundsWidth();
                double otherTop = other.getROI().getBoundsY();
                boolean horizontallyOverlaps = otherLeft < right && otherRight > left;
                if (otherTop > laneTop && horizontallyOverlaps) {
                    laneBottom = Math.min(laneBottom, otherTop);
                }
            }

            // Claim bands whose centroid falls inside the lane box.
            List<PathObject> laneBands = new ArrayList<>();
            for (PathObject band : bands) {
                if (assignedBands.contains(band)) {
                    continue;
                }
                double xCent = band.getROI().getCentroidX();
                double yCent = band.getROI().getCentroidY();
                if (xCent >= left && xCent <= right && yCent >= laneTop && yCent <= laneBottom) {
                    laneBands.add(band);
                    assignedBands.add(band);
                }
            }
            lanes.add(new Lane(well, laneBands));
        }

        // Bands with no well mate become straggler lanes via DBSCAN. eps is derived from the full band
        // population for a stable length scale even when only a few stragglers remain.
        List<PathObject> stragglers = new ArrayList<>();
        for (PathObject band : bands) {
            if (!assignedBands.contains(band)) {
                stragglers.add(band);
            }
        }
        for (List<PathObject> group : clusterBandsByX(stragglers, epsFor(bands))) {
            lanes.add(new Lane(null, group));
        }

        finalizeLanes(lanes);
    }

    /**
     * DBSCAN lane finder used when no wells are present at all, mirroring the Python
     * {@code DBSCANLaneAnalyzer.cluster_bands}. Every lane is a straggler lane (no well).
     */
    private static void labelBandsByDbscan(List<PathObject> bands) {
        List<Lane> lanes = new ArrayList<>();
        for (List<PathObject> group : clusterBandsByX(bands, epsFor(bands))) {
            lanes.add(new Lane(null, group));
        }
        finalizeLanes(lanes);
    }

    /**
     * Orders every lane left-to-right, renumbers them {@code 1..N}, applies the labels (wells become
     * {@code W<lane>}, bands {@code L<lane>-<band>} top-to-bottom), fits a lane axis {@code x = m·y + c}
     * per lane, and refreshes the lane-line overlay. This is the single place lane numbers are
     * assigned, so well and straggler lanes share one consistent scheme.
     *
     * <p>Split-band repair is deliberately <b>not</b> done here — it is a separate post-processing step
     * ({@link #RepairSplitBands(Collection)}) the user triggers explicitly.
     */
    private static void finalizeLanes(List<Lane> lanes) {
        lanes.sort(Comparator.comparingDouble(Lane::sortX));

        // Pass 1: label each lane and fit its axis where there are enough points.
        List<LaneAxis> axes = new ArrayList<>();
        int laneId = 1;
        for (Lane lane : lanes) {
            if (lane.well != null) {
                lane.well.setName(String.format("W%d", laneId));
                lane.well.getMeasurementList().put(GelMeasurements.LANE_ID, laneId);
                lane.well.getMeasurementList().put(GelMeasurements.WELL_ID, 1);
            }
            nameLaneBands(lane.bands, laneId);
            axes.add(fitLaneAxis(lane, laneId));
            laneId++;
        }

        // Pass 2: lanes too sparse to fit a slope borrow one from their nearest fitted neighbours.
        borrowMissingSlopes(axes);

        List<PathObject> laneLines = new ArrayList<>();
        for (LaneAxis axis : axes) {
            PathObject line = axis.buildLine();
            if (line != null) {
                laneLines.add(line);
            }
        }
        refreshLaneConnectors(laneLines);
    }

    /**
     * A fitted lane axis {@code x = slope·y + intercept} plus the vertical extent to draw it over.
     * Mirrors the Python {@code estimate_lanes}: lanes run vertically, so x is fit as a function of y.
     */
    private static final class LaneAxis {
        final int laneId;
        final int bandCount;
        final double laneX;         // horizontal position, used to find nearest neighbours
        final double meanX;
        final double meanY;
        final double yTop;
        final double yBottom;
        final ImagePlane plane;
        final boolean hasWell;      // if true, the axis is anchored to start at the well centroid
        final double wellX;
        final double wellY;
        double slope;
        double intercept;
        boolean fitted;             // true if slope came from a real least-squares fit

        LaneAxis(int laneId, int bandCount, double laneX, double meanX, double meanY,
                 double yTop, double yBottom, ImagePlane plane,
                 boolean hasWell, double wellX, double wellY) {
            this.laneId = laneId;
            this.bandCount = bandCount;
            this.laneX = laneX;
            this.meanX = meanX;
            this.meanY = meanY;
            this.yTop = yTop;
            this.yBottom = yBottom;
            this.plane = plane;
            this.hasWell = hasWell;
            this.wellX = wellX;
            this.wellY = wellY;
        }

        /**
         * Builds the lane-axis polyline, or {@code null} for a lane with no bands to draw through.
         * When the lane has a well, the axis always starts exactly at the well centroid (the fitted
         * slope is anchored through it); otherwise it starts at the top of the topmost band.
         */
        PathObject buildLine() {
            if (bandCount == 0 || plane == null) {
                return null;
            }
            double topX;
            double topY;
            double effIntercept;
            if (hasWell) {
                topX = wellX;
                topY = wellY;
                effIntercept = wellX - slope * wellY;   // fitted slope, anchored through the well
            } else {
                topY = yTop;
                effIntercept = intercept;
                topX = slope * topY + effIntercept;
            }
            List<Point2> points = new ArrayList<>();
            points.add(new Point2(topX, topY));
            points.add(new Point2(slope * yBottom + effIntercept, yBottom));
            return polylineConnector(points, plane, laneId);
        }
    }

    /**
     * Least-squares fit of {@code x = slope·y + intercept} through a lane's members (well centroid, if
     * any, plus band centroids). If there are fewer than two points, or they share a y (a not-yet
     * repaired split band), the slope is left unfitted for {@link #borrowMissingSlopes} to fill in.
     */
    private static LaneAxis fitLaneAxis(Lane lane, int laneId) {
        List<Point2> pts = new ArrayList<>();
        ImagePlane plane = null;
        double yTop = Double.POSITIVE_INFINITY;
        double yBottom = Double.NEGATIVE_INFINITY;

        List<PathObject> members = new ArrayList<>();
        if (lane.well != null) {
            members.add(lane.well);
        }
        members.addAll(lane.bands);
        for (PathObject m : members) {
            ROI roi = m.getROI();
            pts.add(new Point2(roi.getCentroidX(), roi.getCentroidY()));
            yTop = Math.min(yTop, roi.getBoundsY());
            yBottom = Math.max(yBottom, roi.getBoundsY() + roi.getBoundsHeight());
            if (plane == null) {
                plane = roi.getImagePlane();
            }
        }

        double meanX = 0.0;
        double meanY = 0.0;
        for (Point2 p : pts) {
            meanX += p.getX();
            meanY += p.getY();
        }
        if (!pts.isEmpty()) {
            meanX /= pts.size();
            meanY /= pts.size();
        }

        boolean hasWell = lane.well != null;
        double wellX = hasWell ? lane.well.getROI().getCentroidX() : 0.0;
        double wellY = hasWell ? lane.well.getROI().getCentroidY() : 0.0;

        LaneAxis axis = new LaneAxis(laneId, lane.bands.size(), lane.sortX(), meanX, meanY,
                yTop, yBottom, plane, hasWell, wellX, wellY);

        double syy = 0.0;
        double sxy = 0.0;
        for (Point2 p : pts) {
            double dy = p.getY() - meanY;
            syy += dy * dy;
            sxy += dy * (p.getX() - meanX);
        }
        if (pts.size() >= 2 && syy > 1e-9) {
            axis.slope = sxy / syy;
            axis.intercept = meanX - axis.slope * meanY;
            axis.fitted = true;
        } else {
            axis.slope = 0.0;                 // provisional: vertical, refined by slope borrowing
            axis.intercept = meanX;
            axis.fitted = false;
        }
        return axis;
    }

    /**
     * Fills in slopes for lanes that could not be fitted (e.g. single-band lanes) by borrowing from
     * the two nearest fitted lanes by horizontal position, mirroring the Python single-band slope
     * borrow. Falls back to a vertical axis if no fitted lane exists.
     */
    private static void borrowMissingSlopes(List<LaneAxis> axes) {
        List<LaneAxis> fitted = new ArrayList<>();
        for (LaneAxis a : axes) {
            if (a.fitted) {
                fitted.add(a);
            }
        }
        for (LaneAxis a : axes) {
            if (a.fitted) {
                continue;
            }
            fitted.sort(Comparator.comparingDouble(f -> Math.abs(f.laneX - a.laneX)));
            int k = Math.min(2, fitted.size());
            if (k > 0) {
                double slope = 0.0;
                for (int i = 0; i < k; i++) {
                    slope += fitted.get(i).slope;
                }
                a.slope = slope / k;
                a.intercept = a.meanX - a.slope * a.meanY;
            }
        }
    }

    /**
     * Merges horizontally-split fragments of a band within one lane, mirroring the Python
     * {@code repair_split_bands}: any pair whose centroids are more side-by-side than stacked
     * ({@code |Δy| / |Δx| < 0.5}) is treated as one band split in two. Connected fragments (via a
     * union-find over those pairs) are merged into a single band using the convex hull of their union.
     *
     * @param bands             the lane's bands
     * @param mergedToAdd       merged band annotations are appended here (to add to the hierarchy)
     * @param fragmentsToRemove the consumed fragment annotations are appended here (to remove)
     * @return the lane's bands after merging (merged bands replace their fragments)
     */
    private static List<PathObject> repairSplitBands(List<PathObject> bands,
                                                     List<PathObject> mergedToAdd,
                                                     List<PathObject> fragmentsToRemove) {
        int n = bands.size();
        if (n < 2) {
            return new ArrayList<>(bands);
        }

        // Union-find over split-pairs so a chain of >2 fragments collapses into one group.
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (isSplitPair(bands.get(i), bands.get(j))) {
                    union(parent, i, j);
                }
            }
        }

        Map<Integer, List<PathObject>> groups = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) {
            groups.computeIfAbsent(find(parent, i), k -> new ArrayList<>()).add(bands.get(i));
        }

        List<PathObject> result = new ArrayList<>();
        for (List<PathObject> group : groups.values()) {
            if (group.size() == 1) {
                result.add(group.get(0));
            } else {
                PathObject merged = mergeFragments(group);
                mergedToAdd.add(merged);
                fragmentsToRemove.addAll(group);
                result.add(merged);
            }
        }
        return result;
    }

    /** True if two bands look like fragments of one horizontally-split band (side-by-side centroids). */
    private static boolean isSplitPair(PathObject a, PathObject b) {
        double yRange = Math.abs(a.getROI().getCentroidY() - b.getROI().getCentroidY());
        double xRange = Math.abs(a.getROI().getCentroidX() - b.getROI().getCentroidX());
        if (xRange < 0.001) {
            xRange = 0.001;   // avoid division by zero
        }
        return (yRange / xRange) < 0.5;
    }

    /** Merges band fragments into one gel-band annotation via the convex hull of their union. */
    private static PathObject mergeFragments(List<PathObject> fragments) {
        Geometry geom = fragments.get(0).getROI().getGeometry();
        for (int i = 1; i < fragments.size(); i++) {
            geom = geom.union(fragments.get(i).getROI().getGeometry());
        }
        geom = geom.convexHull();

        ImagePlane plane = fragments.get(0).getROI().getImagePlane();
        ROI mergedRoi = GeometryTools.geometryToROI(geom, plane);
        return PathObjects.createAnnotationObject(mergedRoi, GelGenieClasses.GEL_BAND.getPathClass());
    }

    private static int find(int[] parent, int i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];   // path halving
            i = parent[i];
        }
        return i;
    }

    private static void union(int[] parent, int a, int b) {
        parent[find(parent, a)] = find(parent, b);
    }

    /**
     * Clusters bands into lanes purely on their x-centroid ({@code min_samples = 1} DBSCAN). In 1-D
     * with {@code min_samples = 1}, DBSCAN reduces exactly to single-linkage chaining: sort by
     * x-centroid and start a new lane wherever the gap to the previous band exceeds {@code eps}.
     *
     * @return lanes as lists of bands, ordered left-to-right; empty if {@code bands} is empty
     */
    private static List<List<PathObject>> clusterBandsByX(List<PathObject> bands, double eps) {
        List<List<PathObject>> lanes = new ArrayList<>();
        if (bands.isEmpty()) {
            return lanes;
        }
        List<PathObject> sorted = new ArrayList<>(bands);
        sorted.sort(new CentroidCompareX());

        List<PathObject> current = new ArrayList<>();
        double prevX = 0.0;
        for (PathObject band : sorted) {
            double x = band.getROI().getCentroidX();
            if (!current.isEmpty() && (x - prevX) > eps) {
                lanes.add(current);
                current = new ArrayList<>();
            }
            current.add(band);
            prevX = x;
        }
        if (!current.isEmpty()) {
            lanes.add(current);
        }
        return lanes;
    }

    /** Sorts a lane's bands top-to-bottom and names them {@code L<lane>-<index>} with measurements. */
    private static void nameLaneBands(List<PathObject> laneBands, int laneId) {
        laneBands.sort(Comparator.comparing((PathObject p) -> p.getROI().getCentroidY()));
        int bandIdCounter = 1;
        for (PathObject band : laneBands) {
            band.setName(String.format("L%d-%d", laneId, bandIdCounter));
            band.getMeasurementList().put(GelMeasurements.LANE_ID, laneId);
            band.getMeasurementList().put(GelMeasurements.BAND_ID, bandIdCounter);
            bandIdCounter++;
        }
    }

    /** DBSCAN eps: the tunable factor (a preference) times the median band width. */
    private static double epsFor(List<PathObject> bands) {
        return GelGeniePrefs.dbscanEpsFactor().getValue() * medianBandWidth(bands);
    }

    /** Median of the bands' bounding-box widths; the base length scale for the DBSCAN eps. */
    private static double medianBandWidth(List<PathObject> bands) {
        double[] widths = bands.stream()
                .mapToDouble(b -> b.getROI().getBoundsWidth())
                .sorted()
                .toArray();
        int n = widths.length;
        if (n == 0) {
            return 0.0;
        }
        return (n % 2 == 1) ? widths[n / 2] : 0.5 * (widths[n / 2 - 1] + widths[n / 2]);
    }

    /** Current image area (width × height) in pixels, or {@link Double#NaN} if it cannot be determined. */
    private static double imageArea() {
        var imageData = QP.getCurrentImageData();
        if (imageData != null && imageData.getServer() != null) {
            var server = imageData.getServer();
            return (double) server.getWidth() * server.getHeight();
        }
        return Double.NaN;
    }

    /** Wraps an ordered set of points into a named lane-line polyline annotation. */
    private static PathObject polylineConnector(List<Point2> points, ImagePlane plane, int laneId) {
        ROI line = ROIs.createPolylineROI(points, plane);
        PathObject connector = PathObjects.createAnnotationObject(line, GelGenieClasses.LANE_CONNECTOR);
        connector.setName(String.format("L%d lane", laneId));
        return connector;
    }

    /**
     * Removes any lane connectors left over from a previous run and adds the freshly built ones, so
     * the overlay always reflects the current well→band assignment.
     */
    private static void refreshLaneConnectors(List<PathObject> connectors) {
        List<PathObject> stale = new ArrayList<>();
        for (PathObject annot : QP.getAnnotationObjects()) {
            if (GelGenieClasses.LANE_CONNECTOR.equals(annot.getPathClass())) {
                stale.add(annot);
            }
        }
        if (!stale.isEmpty()) {
            QP.removeObjects(stale, false);
        }
        if (!connectors.isEmpty()) {
            QP.addObjects(connectors);
        }
    }

    /** Returns the current image height, or {@link Double#MAX_VALUE} if it cannot be determined. */
    private static double getImageHeight() {
        var imageData = QP.getCurrentImageData();
        if (imageData != null && imageData.getServer() != null) {
            return imageData.getServer().getHeight();
        }
        return Double.MAX_VALUE;
    }

    /**
     * Script-friendly system where labelling is applied to all gel bands and wells in the current image.
     */
    public static void LabelBands(){
        Collection<PathObject> actionableAnnotations = new ArrayList<>();
        for (PathObject annot : QP.getAnnotationObjects()) {
            if (GelGenieClasses.isBandOrFiltered(annot) || GelGenieClasses.WELL.matches(annot)) {
                actionableAnnotations.add(annot);
            }
        }
        LabelBands(actionableAnnotations);
    }
}
