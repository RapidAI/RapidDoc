package io.github.rapidai.rapiddoc;

import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

/** Java 8 fallback for native-resource cleanup; explicit close remains preferred. */
final class NativeCleaner {
    private final ReferenceQueue<Object> queue = new ReferenceQueue<>();
    private final Set<Registration> references = ConcurrentHashMap.newKeySet();

    NativeCleaner() {
        Thread thread =
                new Thread(
                        () -> {
                            while (true) {
                                try {
                                    ((Registration) queue.remove()).clean();
                                } catch (InterruptedException ignored) {
                                    /* Keep processing resource cleanup. */
                                } catch (Throwable ignored) {
                                    /* One failed cleanup must not stop the queue. */
                                }
                            }
                        },
                        "rapid-doc-native-cleaner");
        thread.setDaemon(true);
        thread.setContextClassLoader(null);
        thread.start();
    }

    Registration register(Object owner, Runnable action) {
        // Java 8 has no Reference.reachabilityFence. The owner monitor keeps
        // it reachable until its phantom reference is retained by the set.
        synchronized (owner) {
            Registration registration = new Registration(owner, action);
            references.add(registration);
            return registration;
        }
    }

    final class Registration extends PhantomReference<Object> {
        // The action must retain only the native state, never its Java owner.
        private final AtomicReference<Runnable> action;

        Registration(Object owner, Runnable action) {
            super(owner, queue);
            this.action = new AtomicReference<>(action);
        }

        void clean() {
            Runnable pending = action.getAndSet(null);
            if (pending == null) return;
            try {
                pending.run();
            } finally {
                clear();
                references.remove(this);
            }
        }
    }
}
