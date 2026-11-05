package io.github.rapidai.rapiddoc;

import java.util.Map;

final class NativeBindings {
    private NativeBindings() {}

    static native long create(String configJson, String resourcesJson);

    static native void close(long handle);

    static native Result parsePath(
            long handle, String path, String optionsJson, ProgressListener listener);

    static native Result parseBytes(
            long handle, byte[] data, String optionsJson, ProgressListener listener);

    static native void warmup(long handle, String optionsJson, ProgressListener listener);

    static native String loadedModels(long handle);

    static final class Result {
        final String json;
        final Map<String, byte[]> images, drawings;

        Result(String json, Map<String, byte[]> images, Map<String, byte[]> drawings) {
            this.json = json;
            this.images = images;
            this.drawings = drawings;
        }
    }
}
