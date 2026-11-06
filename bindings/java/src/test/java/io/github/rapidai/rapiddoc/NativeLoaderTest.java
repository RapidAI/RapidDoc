package io.github.rapidai.rapiddoc;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import static org.junit.jupiter.api.Assertions.*;

class NativeLoaderTest {
    @TempDir Path temporary;

    @Test
    void detectsBothMacArchitectures() {
        String os = System.getProperty("os.name"), arch = System.getProperty("os.arch");
        try {
            System.setProperty("os.name", "Mac OS X");
            for (String value : new String[] {"aarch64", "arm64"}) {
                System.setProperty("os.arch", value);
                assertEquals("macos-aarch64", NativeLoader.platform());
            }
            for (String value : new String[] {"x86_64", "amd64"}) {
                System.setProperty("os.arch", value);
                assertEquals("macos-x86_64", NativeLoader.platform());
            }
        } finally {
            System.setProperty("os.name", os);
            System.setProperty("os.arch", arch);
        }
    }

    private Path runtime(String name, boolean corrupt) throws Exception {
        Path jar = temporary.resolve(name + ".jar");
        String prefix = "META-INF/rapiddoc-ort/" + NativeLoader.platform() + "/";
        try (ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(jar))) {
            String files = corrupt ? "\"payload\":\"incorrect-hash\"" : "";
            zip.putNextEntry(new ZipEntry(prefix + "manifest.json"));
            zip.write(
                    ("{\"schema\":1,\"platform\":\""
                                    + NativeLoader.platform()
                                    + "\",\"files\":{"
                                    + files
                                    + "}}")
                            .getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
            if (corrupt) {
                zip.putNextEntry(new ZipEntry(prefix + "payload"));
                zip.write(new byte[] {1});
                zip.closeEntry();
            }
        }
        return jar;
    }

    private void child(String mode, Path... runtimes) throws Exception {
        List<String> entries = new ArrayList<>();
        for (String entry : System.getProperty("java.class.path").split(File.pathSeparator))
            if (!entry.contains("rapid-doc-ort-")) entries.add(entry);
        for (Path runtime : runtimes) entries.add(runtime.toString());
        String executable =
                System.getProperty("os.name").startsWith("Windows") ? "java.exe" : "java";
        Process child =
                new ProcessBuilder(
                                Paths.get(System.getProperty("java.home"), "bin", executable)
                                        .toString(),
                                "-Drapid_doc.native.cache=" + temporary.resolve("cache"),
                                "-cp",
                                String.join(File.pathSeparator, entries),
                                Probe.class.getName(),
                                mode)
                        .redirectErrorStream(true)
                        .start();
        try {
            assertTrue(child.waitFor(30, TimeUnit.SECONDS), "Loader probe timed out");
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int size;
            while ((size = child.getInputStream().read(buffer)) != -1)
                output.write(buffer, 0, size);
            assertEquals(
                    0, child.exitValue(), new String(output.toByteArray(), StandardCharsets.UTF_8));
        } finally {
            child.destroyForcibly();
        }
    }

    @Test
    void rejectsTwoRuntimeDistributions() throws Exception {
        child("duplicate", runtime("cpu", false), runtime("cuda", false));
    }

    @Test
    void verifiesIndependentRuntimeChecksums() throws Exception {
        child("checksum", runtime("corrupt", true));
    }

    @Test
    void missingOrtRemainsOptionalForOffice() throws Exception {
        child("missing");
    }

    public static class Probe {
        public static void main(String[] args) {
            if (args[0].equals("missing")) {
                assertNull(NativeLoader.ortDirectory());
                return;
            }
            RapidDocException failure =
                    assertThrows(RapidDocException.class, NativeLoader::ortDirectory);
            assertEquals("runtime_unavailable", failure.getCode());
            assertTrue(
                    failure.getMessage()
                            .contains(
                                    args[0].equals("duplicate")
                                            ? "Multiple native distributions"
                                            : "checksum mismatch"));
        }
    }
}
