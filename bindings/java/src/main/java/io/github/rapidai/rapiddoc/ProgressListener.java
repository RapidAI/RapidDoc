package io.github.rapidai.rapiddoc;

/** Called synchronously on the parsing thread. Throw to cancel; exceptions propagate unchanged. */
@FunctionalInterface
public interface ProgressListener {
    void onProgress(ProgressEvent event);
}
