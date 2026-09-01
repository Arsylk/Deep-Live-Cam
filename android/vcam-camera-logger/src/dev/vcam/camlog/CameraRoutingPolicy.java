package dev.vcam.camlog;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Pure camera-ID routing policy used by the Xposed hook.
 *
 * <p>The selected processed camera takes the default physical front camera's
 * position in the public ID list. Every other camera remains present and in
 * its original relative order. Keeping this logic independent of Android and
 * Xposed makes the safety-critical enumeration behavior host-testable.</p>
 */
final class CameraRoutingPolicy {

    static final class Route {
        private final String[] exposedIds;
        private final String physicalFrontId;
        private final String processedFrontId;
        private final Set<String> redirectedPhysicalFrontIds;

        private Route(String[] exposedIds, String physicalFrontId, String processedFrontId,
                Set<String> redirectedPhysicalFrontIds) {
            this.exposedIds = exposedIds;
            this.physicalFrontId = physicalFrontId;
            this.processedFrontId = processedFrontId;
            this.redirectedPhysicalFrontIds = Collections.unmodifiableSet(
                    new LinkedHashSet<>(redirectedPhysicalFrontIds));
        }

        String[] exposedIds() {
            return Arrays.copyOf(exposedIds, exposedIds.length);
        }

        String physicalFrontId() {
            return physicalFrontId;
        }

        String processedFrontId() {
            return processedFrontId;
        }

        String[] redirectedPhysicalFrontIds() {
            return redirectedPhysicalFrontIds.toArray(new String[0]);
        }

        boolean isActive() {
            return physicalFrontId != null && processedFrontId != null;
        }

        boolean canRedirect() {
            return processedFrontId != null && !redirectedPhysicalFrontIds.isEmpty();
        }

        boolean isProcessedCameraId(String cameraId) {
            return canRedirect() && processedFrontId.equals(cameraId);
        }

        String cameraIdForOpen(String requestedId) {
            if (canRedirect() && redirectedPhysicalFrontIds.contains(requestedId)) {
                return processedFrontId;
            }
            return requestedId;
        }
    }

    private CameraRoutingPolicy() {}

    /**
     * Creates a route while treating an already-substituted ID list as final.
     * This is the safe entry point for framework hooks, which may be chained
     * more than once by different package classloaders in the same process.
     */
    static Route createIdempotent(String[] cameraIds, String defaultPhysicalFrontId,
            String[] processedCandidates) {
        if (processedAlreadyOccupiesFrontSlot(
                cameraIds, defaultPhysicalFrontId, processedCandidates)) {
            return create(cameraIds, null, processedCandidates);
        }
        return create(cameraIds, defaultPhysicalFrontId, processedCandidates);
    }

    /**
     * Creates an idempotent route while retaining explicit physical-front aliases.
     * Alias IDs remain visible in enumeration, but characteristics and open requests
     * for them resolve to the processed camera. Aliases may be temporarily absent
     * from {@code cameraIds}, allowing cached app choices to survive a provider restart.
     */
    static Route createIdempotentWithAliases(String[] cameraIds,
            String defaultPhysicalFrontId, String[] redirectedPhysicalFrontIds,
            String[] processedCandidates) {
        if (processedAlreadyOccupiesFrontSlot(
                cameraIds, defaultPhysicalFrontId, processedCandidates)) {
            return createWithAliases(cameraIds, null, redirectedPhysicalFrontIds,
                    processedCandidates);
        }
        return createWithAliases(cameraIds, defaultPhysicalFrontId,
                redirectedPhysicalFrontIds, processedCandidates);
    }

    static boolean processedAlreadyOccupiesFrontSlot(String[] cameraIds,
            String nextPhysicalFrontId, String[] processedCandidates) {
        if (cameraIds == null || nextPhysicalFrontId == null
                || processedCandidates == null) {
            return false;
        }
        Set<String> candidates = new HashSet<>(Arrays.asList(processedCandidates));
        int processedIndex = -1;
        int physicalFrontIndex = -1;
        for (int index = 0; index < cameraIds.length; index++) {
            String id = cameraIds[index];
            if (processedIndex < 0 && candidates.contains(id)) {
                processedIndex = index;
            }
            if (nextPhysicalFrontId.equals(id)) {
                physicalFrontIndex = index;
                break;
            }
        }
        return processedIndex >= 0 && physicalFrontIndex >= 0
                && processedIndex < physicalFrontIndex;
    }

    static Route create(String[] cameraIds, String defaultPhysicalFrontId,
            String[] processedCandidates) {
        String[] original = cameraIds == null
                ? new String[0]
                : Arrays.copyOf(cameraIds, cameraIds.length);
        if (defaultPhysicalFrontId == null
                || !Arrays.asList(original).contains(defaultPhysicalFrontId)) {
            return new Route(original, null, null, Collections.emptySet());
        }
        return createWithAliases(original, defaultPhysicalFrontId,
                new String[] {defaultPhysicalFrontId}, processedCandidates);
    }

    static Route createWithAliases(String[] cameraIds, String defaultPhysicalFrontId,
            String[] redirectedPhysicalFrontIds, String[] processedCandidates) {
        String[] original = cameraIds == null
                ? new String[0]
                : Arrays.copyOf(cameraIds, cameraIds.length);
        if (processedCandidates == null) {
            return new Route(original, null, null, Collections.emptySet());
        }

        Set<String> present = new HashSet<>(Arrays.asList(original));
        String processedFrontId = null;
        for (String candidate : processedCandidates) {
            if (candidate != null && present.contains(candidate)
                    && !candidate.equals(defaultPhysicalFrontId)) {
                processedFrontId = candidate;
                break;
            }
        }
        if (processedFrontId == null) {
            return new Route(original, null, null, Collections.emptySet());
        }

        boolean replacePrimary = defaultPhysicalFrontId != null
                && present.contains(defaultPhysicalFrontId);
        Set<String> redirects = new LinkedHashSet<>();
        if (replacePrimary) {
            redirects.add(defaultPhysicalFrontId);
        }
        if (redirectedPhysicalFrontIds != null) {
            for (String id : redirectedPhysicalFrontIds) {
                if (id != null && !processedFrontId.equals(id)) {
                    redirects.add(id);
                }
            }
        }
        if (redirects.isEmpty()) {
            return new Route(original, null, processedFrontId, Collections.emptySet());
        }

        // Remove only the chosen processed ID's old occurrence, then insert it
        // exactly where the physical default-front ID used to be. All other
        // physical and virtual cameras retain their relative order. Additional
        // aliases are deliberately left visible and resolve only when queried.
        List<String> exposed = new ArrayList<>(original.length);
        if (replacePrimary) {
            for (String id : original) {
                if (processedFrontId.equals(id)) {
                    continue;
                }
                if (defaultPhysicalFrontId.equals(id)) {
                    exposed.add(processedFrontId);
                } else {
                    exposed.add(id);
                }
            }
        } else {
            exposed.addAll(Arrays.asList(original));
        }
        return new Route(exposed.toArray(new String[0]),
                replacePrimary ? defaultPhysicalFrontId : null,
                processedFrontId, redirects);
    }
}
