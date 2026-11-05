package io.github.rapidai.rapiddoc;

import org.junit.jupiter.api.Test;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

class NativeCleanerTest {
    @Test
    void queuedAndConcurrentExplicitCleanupRunOnlyOnce() throws Exception {
        NativeCleaner cleaner = new NativeCleaner();
        AtomicInteger calls = new AtomicInteger();
        CountDownLatch cleaned = new CountDownLatch(1);
        NativeCleaner.Registration registration =
                cleaner.register(
                        new Object(),
                        () -> {
                            calls.incrementAndGet();
                            cleaned.countDown();
                        });
        // Enqueue explicitly instead of depending on nondeterministic GC.
        registration.enqueue();
        assertTrue(cleaned.await(5, TimeUnit.SECONDS));
        registration.clean();
        assertEquals(1, calls.get());

        AtomicInteger concurrentCalls = new AtomicInteger();
        NativeCleaner.Registration concurrent =
                cleaner.register(new Object(), concurrentCalls::incrementAndGet);
        Thread first = new Thread(concurrent::clean), second = new Thread(concurrent::clean);
        first.start();
        second.start();
        first.join(5000);
        second.join(5000);
        assertFalse(first.isAlive());
        assertFalse(second.isAlive());
        assertEquals(1, concurrentCalls.get());
    }
}
