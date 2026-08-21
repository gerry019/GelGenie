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

package qupath.ext.gelgenie;

/**
 * Single source of truth for the measurement-list keys GelGenie writes onto its annotations.
 *
 * <p>These keys (e.g. {@code "LaneID"}) are set by {@link qupath.ext.gelgenie.tools.BandSorter} and
 * read back by the data table and comparators. They are <b>persisted</b> in QuPath's measurement
 * lists, so the string values must stay stable across releases. Previously they were hand-typed as
 * bare literals across several files; centralising them here keeps them consistent and makes adding
 * new measurement keys a single-file edit.
 */
public final class GelMeasurements {

    private GelMeasurements() {} // constants holder - not instantiable

    /** Measurement key for the lane a band/well belongs to. */
    public static final String LANE_ID = "LaneID";

    /** Measurement key for a band's position within its lane. */
    public static final String BAND_ID = "BandID";

    /** Measurement key for a well's position within its lane. */
    public static final String WELL_ID = "WellID";
}
