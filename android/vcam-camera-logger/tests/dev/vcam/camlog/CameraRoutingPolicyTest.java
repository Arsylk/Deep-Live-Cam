package dev.vcam.camlog;

import java.util.Arrays;

public final class CameraRoutingPolicyTest {

    private static final String[] CANDIDATES = {"101", "100", "120"};

    public static void main(String[] args) {
        replacesOnlyDefaultFront();
        matchesLivePhoneTopology();
        redirectsSecondaryFrontWithoutHidingIt();
        redirectsEveryDiscoveredFrontToProcessed();
        neverAddsAnExtraCameraToTheList();
        retainsCachedAliasesAcrossProviderRestart();
        keepsAliasOnIdempotentSecondPass();
        aliasFallsBackWhenProcessedCameraIsAbsent();
        preservesSecondaryFrontAndAuxiliaryCameras();
        movesExistingProcessedIdIntoFrontPosition();
        honorsProcessedCameraPriorityWithoutHidingOthers();
        detectsAlreadyRoutedListBeforeSecondaryFront();
        secondRoutingPassIsANoOp();
        fallsBackWhenProcessedCameraIsAbsent();
        fallsBackWhenPhysicalFrontIsAbsent();
        System.out.println("CameraRoutingPolicyTest: all tests passed");
    }

    private static void replacesOnlyDefaultFront() {
        CameraRoutingPolicy.Route route = CameraRoutingPolicy.create(
                new String[] {"0", "1", "120"}, "1", CANDIDATES);
        assertArrayEquals(new String[] {"0", "120"}, route.exposedIds());
        assertEquals("120", route.cameraIdForOpen("1"));
        assertEquals("0", route.cameraIdForOpen("0"));
        assertEquals("120", route.cameraIdForOpen("120"));
    }

    private static void matchesLivePhoneTopology() {
        CameraRoutingPolicy.Route route = CameraRoutingPolicy.create(
                new String[] {"0", "1", "2", "3", "4", "5", "6", "7", "120"},
                "1", CANDIDATES);
        assertArrayEquals(
                new String[] {"0", "120", "2", "3", "4", "5", "6", "7"},
                route.exposedIds());
        assertEquals("120", route.cameraIdForOpen("1"));
        assertEquals("7", route.cameraIdForOpen("7"));
    }

    private static void redirectsSecondaryFrontWithoutHidingIt() {
        CameraRoutingPolicy.Route route = CameraRoutingPolicy.createIdempotentWithAliases(
                new String[] {"0", "1", "2", "3", "4", "5", "6", "7", "120"},
                "1", new String[] {"7"}, CANDIDATES);
        assertArrayEquals(
                new String[] {"0", "120", "2", "3", "4", "5", "6", "7"},
                route.exposedIds());
        assertArrayEquals(new String[] {"1", "7"},
                route.redirectedPhysicalFrontIds());
        assertEquals("120", route.cameraIdForOpen("1"));
        assertEquals("120", route.cameraIdForOpen("7"));
        assertTrue(route.isActive());
    }

    /**
     * The global module discovers every LENS_FACING_FRONT physical id and
     * passes them all as redirect aliases. The primary (first discovered)
     * front's slot is taken by the processed camera; the others stay visible
     * but resolve to the processed camera on open, and the back/aux cameras
     * are untouched.
     */
    private static void redirectsEveryDiscoveredFrontToProcessed() {
        // ids: 0=back, 1=front-main, 2=aux back, 3=front-wide, 120=processed.
        String[] fronts = {"1", "3"};
        CameraRoutingPolicy.Route route = CameraRoutingPolicy.createIdempotentWithAliases(
                new String[] {"0", "1", "2", "3", "120"},
                "1", fronts, CANDIDATES);
        // Primary front slot replaced; secondary front still enumerated.
        assertArrayEquals(new String[] {"0", "120", "2", "3"}, route.exposedIds());
        assertArrayEquals(new String[] {"1", "3"},
                route.redirectedPhysicalFrontIds());
        // Both fronts route to the processed camera.
        assertEquals("120", route.cameraIdForOpen("1"));
        assertEquals("120", route.cameraIdForOpen("3"));
        // Back and aux are left alone.
        assertEquals("0", route.cameraIdForOpen("0"));
        assertEquals("2", route.cameraIdForOpen("2"));
        assertTrue(route.isActive());
    }

    /**
     * Invariant: the processed camera is only ever moved into the front slot,
     * never appended. The exposed list must contain the processed id exactly
     * once and be exactly one entry shorter than the input (the removed front),
     * across single-front and multi-front topologies.
     */
    private static void neverAddsAnExtraCameraToTheList() {
        String[][] inputs = {
                {"0", "1", "120"},
                {"0", "1", "2", "3", "4", "5", "6", "7", "120"},
                {"0", "1", "2", "3", "120"},
        };
        String[][] frontSets = {
                {"1"},
                {"1", "7"},
                {"1", "3"},
        };
        for (int i = 0; i < inputs.length; i++) {
            CameraRoutingPolicy.Route route =
                    CameraRoutingPolicy.createIdempotentWithAliases(
                            inputs[i], frontSets[i][0], frontSets[i], CANDIDATES);
            String[] exposed = route.exposedIds();
            assertEquals(String.valueOf(inputs[i].length - 1),
                    String.valueOf(exposed.length));
            int processedCount = 0;
            for (String id : exposed) {
                if ("120".equals(id)) processedCount++;
                // No id may appear in the exposed list that was not already
                // present in the input (nothing invented).
                assertTrue(Arrays.asList(inputs[i]).contains(id));
            }
            assertEquals("1", String.valueOf(processedCount));
        }
    }

    private static void retainsCachedAliasesAcrossProviderRestart() {
        CameraRoutingPolicy.Route route = CameraRoutingPolicy.createIdempotentWithAliases(
                new String[] {"120"}, null, new String[] {"7"}, CANDIDATES);
        assertArrayEquals(new String[] {"120"}, route.exposedIds());
        assertEquals("120", route.cameraIdForOpen("7"));
        assertFalse(route.isActive());
        assertTrue(route.canRedirect());
    }

    private static void keepsAliasOnIdempotentSecondPass() {
        CameraRoutingPolicy.Route route = CameraRoutingPolicy.createIdempotentWithAliases(
                new String[] {"0", "120", "2", "3", "4", "5", "6", "7"},
                "7", new String[] {"7"}, CANDIDATES);
        assertArrayEquals(
                new String[] {"0", "120", "2", "3", "4", "5", "6", "7"},
                route.exposedIds());
        assertEquals("120", route.cameraIdForOpen("7"));
        assertFalse(route.isActive());
        assertTrue(route.canRedirect());
    }

    private static void aliasFallsBackWhenProcessedCameraIsAbsent() {
        String[] original = {"0", "1", "2", "7"};
        CameraRoutingPolicy.Route route = CameraRoutingPolicy.createIdempotentWithAliases(
                original, "1", new String[] {"7"}, CANDIDATES);
        assertArrayEquals(original, route.exposedIds());
        assertEquals("1", route.cameraIdForOpen("1"));
        assertEquals("7", route.cameraIdForOpen("7"));
        assertFalse(route.isActive());
        assertFalse(route.canRedirect());
    }

    private static void preservesSecondaryFrontAndAuxiliaryCameras() {
        CameraRoutingPolicy.Route route = CameraRoutingPolicy.create(
                new String[] {"0", "1", "2", "3", "120"}, "1", CANDIDATES);
        assertArrayEquals(new String[] {"0", "120", "2", "3"}, route.exposedIds());
        assertEquals("2", route.cameraIdForOpen("2"));
        assertEquals("3", route.cameraIdForOpen("3"));
    }

    private static void movesExistingProcessedIdIntoFrontPosition() {
        CameraRoutingPolicy.Route route = CameraRoutingPolicy.create(
                new String[] {"120", "0", "1", "2"}, "1", CANDIDATES);
        assertArrayEquals(new String[] {"0", "120", "2"}, route.exposedIds());
    }

    private static void honorsProcessedCameraPriorityWithoutHidingOthers() {
        CameraRoutingPolicy.Route route = CameraRoutingPolicy.create(
                new String[] {"0", "1", "100", "101", "120"}, "1", CANDIDATES);
        assertArrayEquals(new String[] {"0", "101", "100", "120"}, route.exposedIds());
        assertEquals("101", route.cameraIdForOpen("1"));
    }

    private static void detectsAlreadyRoutedListBeforeSecondaryFront() {
        String[] onceRouted = {"0", "120", "2", "3", "4", "5", "6", "7"};
        assertTrue(CameraRoutingPolicy.processedAlreadyOccupiesFrontSlot(
                onceRouted, "7", CANDIDATES));
        assertFalse(CameraRoutingPolicy.processedAlreadyOccupiesFrontSlot(
                new String[] {"0", "1", "2", "7", "120"}, "1", CANDIDATES));
    }

    private static void secondRoutingPassIsANoOp() {
        CameraRoutingPolicy.Route first = CameraRoutingPolicy.createIdempotent(
                new String[] {"0", "1", "2", "3", "4", "5", "6", "7", "120"},
                "1", CANDIDATES);
        CameraRoutingPolicy.Route second = CameraRoutingPolicy.createIdempotent(
                first.exposedIds(), "7", CANDIDATES);
        assertArrayEquals(first.exposedIds(), second.exposedIds());
        assertEquals("7", second.cameraIdForOpen("7"));
        assertFalse(second.isActive());
    }

    private static void fallsBackWhenProcessedCameraIsAbsent() {
        String[] original = {"0", "1", "2"};
        CameraRoutingPolicy.Route route = CameraRoutingPolicy.create(
                original, "1", CANDIDATES);
        assertArrayEquals(original, route.exposedIds());
        assertEquals("1", route.cameraIdForOpen("1"));
        assertFalse(route.isActive());
    }

    private static void fallsBackWhenPhysicalFrontIsAbsent() {
        String[] original = {"0", "2", "120"};
        CameraRoutingPolicy.Route route = CameraRoutingPolicy.create(
                original, null, CANDIDATES);
        assertArrayEquals(original, route.exposedIds());
        assertEquals("120", route.cameraIdForOpen("120"));
        assertFalse(route.isActive());
    }

    private static void assertArrayEquals(String[] expected, String[] actual) {
        if (!Arrays.equals(expected, actual)) {
            throw new AssertionError("expected " + Arrays.toString(expected)
                    + " but got " + Arrays.toString(actual));
        }
    }

    private static void assertEquals(String expected, String actual) {
        if (!expected.equals(actual)) {
            throw new AssertionError("expected " + expected + " but got " + actual);
        }
    }

    private static void assertFalse(boolean value) {
        if (value) {
            throw new AssertionError("expected false");
        }
    }


    private static void assertTrue(boolean value) {
        if (!value) {
            throw new AssertionError("expected true");
        }
    }
}
