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

import qupath.lib.common.ColorTools;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;

import java.util.ArrayList;
import java.util.List;

/**
 * Single source of truth for GelGenie's segmentation classes.
 *
 * <p>The model produces a per-pixel class index: {@code 0 = Background}, {@code 1 = Gel Band},
 * {@code 2 = Well}. These names, class indices and annotation colours were previously duplicated as
 * bare string literals and magic colour ints across several files; centralising them here keeps the
 * scheme consistent and makes future changes to the class set a single-file edit.
 *
 * <p>The enum values are declared in class-index order, so {@code values()[i].getClassIndex() == i}.
 *
 * <p>{@link #GLOBAL_BACKGROUND} is deliberately <b>not</b> an enum member: it is a user-drawn region
 * used for background correction, not a model-output segmentation class, so it must not appear in the
 * ordered class list.
 */
public enum GelGenieClasses {

    BACKGROUND(0, "Background", ColorTools.WHITE),
    GEL_BAND(1, "Gel Band", 10709517),
    WELL(2, "Well", 3394611);

    /** Class name of the user-drawn global background-correction region (not a segmentation class). */
    public static final PathClass GLOBAL_BACKGROUND = PathClass.fromString("Global Background", 906200);

    private final int classIndex;
    private final String className;
    private final int color;

    GelGenieClasses(int classIndex, String className, int color) {
        this.classIndex = classIndex;
        this.className = className;
        this.color = color;
    }

    /** The per-pixel class index produced by the model for this class. */
    public int getClassIndex() {
        return classIndex;
    }

    /** The QuPath {@link PathClass} name for this class. */
    public String getName() {
        return className;
    }

    /** The packed-RGB annotation colour for this class. */
    public int getColor() {
        return color;
    }

    /** The canonical {@link PathClass} (cached singleton) for this class, with its standard colour. */
    public PathClass getPathClass() {
        return PathClass.fromString(className, color);
    }

    /** {@code true} if {@code pathObject} carries exactly this class. Null-safe. */
    public boolean matches(PathObject pathObject) {
        if (pathObject == null)
            return false;
        PathClass pathClass = pathObject.getPathClass();
        return pathClass != null && className.equals(pathClass.getName());
    }

    /**
     * Ordered class names for the first {@code numClasses} classes, e.g. for building a DJL
     * translator's class list. {@code numClasses} is capped at the number of defined classes.
     *
     * @param numClasses number of leading classes to return (2 → [Background, Gel Band];
     *                   3 → [Background, Gel Band, Well])
     * @return the class names in class-index order
     */
    public static List<String> classNames(int numClasses) {
        int n = Math.min(numClasses, values().length);
        List<String> names = new ArrayList<>(n);
        for (int i = 0; i < n; i++)
            names.add(values()[i].className);
        return names;
    }
}