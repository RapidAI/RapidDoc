package io.github.rapidai.rapiddoc;

/** A native stage event. Unknown numbers are -1; pageIndex is zero-based. */
public final class ProgressEvent {
    private final String stage, detail;
    private final long pageIndex, completed, total;
    private final double elapsedSeconds;

    public ProgressEvent(
            String stage,
            long pageIndex,
            long completed,
            long total,
            double elapsedSeconds,
            String detail) {
        this.stage = stage;
        this.pageIndex = pageIndex;
        this.completed = completed;
        this.total = total;
        this.elapsedSeconds = elapsedSeconds;
        this.detail = detail;
    }

    public String getStage() {
        return stage;
    }

    public long getPageIndex() {
        return pageIndex;
    }

    public long getCompleted() {
        return completed;
    }

    public long getTotal() {
        return total;
    }

    public double getElapsedSeconds() {
        return elapsedSeconds;
    }

    public String getDetail() {
        return detail;
    }
}
