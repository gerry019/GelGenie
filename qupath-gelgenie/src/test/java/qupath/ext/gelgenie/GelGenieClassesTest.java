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
import qupath.lib.common.ColorTools;
import qupath.lib.objects.classes.PathClass;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Smoke test for {@link GelGenieClasses}, the single source of truth for GelGenie's segmentation
 * classes. This is the first test in the extension's suite; it guards the class scheme (names,
 * indices, colours) against accidental change and pins the {@code ModelRunner} colour fix
 * (Gel Band must resolve to {@code 10709517}, never the old stray {@code 8000}).
 */
public class GelGenieClassesTest {

    @Test
    public void classIndicesMatchDeclarationOrder() {
        assertEquals(0, GelGenieClasses.BACKGROUND.getClassIndex());
        assertEquals(1, GelGenieClasses.GEL_BAND.getClassIndex());
        assertEquals(2, GelGenieClasses.WELL.getClassIndex());

        // values() is relied upon (e.g. by classNames) to be in class-index order
        GelGenieClasses[] values = GelGenieClasses.values();
        for (int i = 0; i < values.length; i++)
            assertEquals(i, values[i].getClassIndex(), "value at position " + i);
    }

    @Test
    public void namesAreStable() {
        assertEquals("Background", GelGenieClasses.BACKGROUND.getName());
        assertEquals("Gel Band", GelGenieClasses.GEL_BAND.getName());
        assertEquals("Well", GelGenieClasses.WELL.getName());
    }

    @Test
    public void coloursAreStable() {
        assertEquals(ColorTools.WHITE, GelGenieClasses.BACKGROUND.getColor());
        assertEquals(10709517, GelGenieClasses.GEL_BAND.getColor());
        assertEquals(3394611, GelGenieClasses.WELL.getColor());
    }

    @Test
    public void getPathClassIsCanonical() {
        PathClass gelBand = GelGenieClasses.GEL_BAND.getPathClass();
        assertNotNull(gelBand);
        assertEquals("Gel Band", gelBand.getName());
        // pins the colour fix: Gel Band must be 10709517, never the old stray 8000
        assertEquals(10709517, gelBand.getColor().intValue());

        PathClass well = GelGenieClasses.WELL.getPathClass();
        assertEquals("Well", well.getName());
        assertEquals(3394611, well.getColor().intValue());
    }

    @Test
    public void classNamesReturnsOrderedPrefix() {
        assertEquals(List.of("Background", "Gel Band"), GelGenieClasses.classNames(2));
        assertEquals(List.of("Background", "Gel Band", "Well"), GelGenieClasses.classNames(3));
        // capped at the number of defined classes
        assertEquals(List.of("Background", "Gel Band", "Well"), GelGenieClasses.classNames(5));
    }

    @Test
    public void matchesIsNullSafe() {
        assertFalse(GelGenieClasses.GEL_BAND.matches(null));
    }

    @Test
    public void globalBackgroundConstant() {
        assertNotNull(GelGenieClasses.GLOBAL_BACKGROUND);
        assertEquals("Global Background", GelGenieClasses.GLOBAL_BACKGROUND.getName());
        assertEquals(906200, GelGenieClasses.GLOBAL_BACKGROUND.getColor().intValue());
    }
}