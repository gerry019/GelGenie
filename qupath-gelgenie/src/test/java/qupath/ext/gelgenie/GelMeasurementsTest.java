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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Guards the {@link GelMeasurements} keys against accidental change. These strings are persisted in
 * QuPath measurement lists, so their values must stay stable across releases or previously-saved
 * projects would lose their lane/band/well associations.
 */
public class GelMeasurementsTest {

    @Test
    public void keysAreStable() {
        assertEquals("LaneID", GelMeasurements.LANE_ID);
        assertEquals("BandID", GelMeasurements.BAND_ID);
        assertEquals("WellID", GelMeasurements.WELL_ID);
    }
}
