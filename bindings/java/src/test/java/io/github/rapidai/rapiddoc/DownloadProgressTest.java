package io.github.rapidai.rapiddoc;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DownloadProgressTest {
    @Test
    void logsDownloadsWithoutCustomListenerAndPreservesCancellation() {
        Logger logger = Logger.getLogger("io.github.rapidai.rapiddoc.download");
        List<String> messages = new ArrayList<>();
        Handler handler =
                new Handler() {
                    public void publish(LogRecord record) {
                        messages.add(record.getMessage());
                    }

                    public void flush() {}

                    public void close() {}
                };
        logger.addHandler(handler);
        try {
            String detail =
                    "{\"model_id\":\"test\",\"url\":\"http://localhost/model\","
                            + "\"path\":\"models/test.onnx\",\"total\":1048576}";
            ProgressListener progress = DownloadProgress.wrap(null);
            for (String stage :
                    new String[] {
                        "download_started",
                        "download_progress",
                        "download_verifying",
                        "download_completed"
                    }) progress.onProgress(new ProgressEvent(stage, -1, 0, 1048576, 0, detail));
            assertTrue(messages.stream().anyMatch(s -> s.startsWith("Initiating download:")));
            assertTrue(messages.stream().anyMatch(s -> s.contains("1.00 MiB")));
            assertTrue(messages.stream().anyMatch(s -> s.startsWith("Successfully saved to:")));
            RuntimeException cancelled = new RuntimeException("cancel");
            assertSame(
                    cancelled,
                    assertThrows(
                            RuntimeException.class,
                            () ->
                                    DownloadProgress.wrap(
                                                    e -> {
                                                        throw cancelled;
                                                    })
                                            .onProgress(
                                                    new ProgressEvent("layout", -1, 0, 1, 0, ""))));
        } finally {
            logger.removeHandler(handler);
        }
    }
}
