package io.github.rapidai.rapiddoc;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Iterator;
import java.util.Enumeration;
import java.net.URL;
import java.util.Locale;
import java.util.Map;

/** Extracts a verified platform bundle once, without a Python environment. */
final class NativeLoader {
    private static Path loaded;
    private static Path ort;

    private NativeLoader() {}

    static synchronized Path load() {
        if (loaded != null) return loaded;
        String explicit = System.getProperty("rapid_doc.native.library");
        if (explicit != null) {
            Path library = Paths.get(explicit).toAbsolutePath();
            System.load(library.toString());
            loaded = library.getParent();
            return loaded;
        }
        String directory = System.getProperty("rapid_doc.native.directory");
        String platform = platform();
        String library =
                platform.startsWith("windows") ? "rapid_doc_jni.dll" : "librapid_doc_jni.so";
        if (directory != null) {
            Path bundle = Paths.get(directory).toAbsolutePath();
            System.load(bundle.resolve(library).toString());
            loaded = bundle;
            return loaded;
        }
        Path root = extract("/META-INF/native/" + platform + "/", platform, library, false);
        System.load(root.resolve(library).toString());
        loaded = root;
        return loaded;
    }

    static synchronized Path ortDirectory() {
        if (System.getProperty("rapid_doc.native.library") != null
                || System.getProperty("rapid_doc.native.directory") != null) return null;
        if (ort == null) {
            String platform = platform();
            ort = extract("/META-INF/rapiddoc-ort/" + platform + "/", platform, null, true);
        }
        return ort;
    }

    private static Path extract(
            String resource, String platform, String required, boolean optional) {
        try {
            Enumeration<URL> manifests =
                    NativeLoader.class
                            .getClassLoader()
                            .getResources((resource + "manifest.json").substring(1));
            if (!manifests.hasMoreElements()) {
                if (optional) return null;
                throw new IOException("Missing native resource " + resource + "manifest.json");
            }
            manifests.nextElement();
            if (manifests.hasMoreElements())
                throw new IOException(
                        "Multiple native distributions found at "
                                + resource
                                + "; keep only one platform and CPU/CUDA runtime on the classpath");
            byte[] manifest = read(resource + "manifest.json");
            JsonNode data = RapidDoc.JSON.readTree(manifest);
            if (data.path("schema").asInt() != 1
                    || !platform.equals(data.path("platform").asText()))
                throw new IOException("Invalid native bundle manifest");
            Path root =
                    Paths.get(
                                    System.getProperty(
                                            "rapid_doc.native.cache",
                                            Paths.get(
                                                            System.getProperty("user.home"),
                                                            ".cache",
                                                            "rapid_doc",
                                                            "native")
                                                    .toString()),
                                    sha(manifest))
                            .toAbsolutePath()
                            .normalize();
            Files.createDirectories(root);
            Iterator<Map.Entry<String, JsonNode>> files = data.path("files").fields();
            boolean found = false;
            while (files.hasNext()) {
                Map.Entry<String, JsonNode> entry = files.next();
                String name = entry.getKey(), expected = entry.getValue().asText();
                Path output = root.resolve(name).normalize();
                if (!output.startsWith(root) || output.equals(root) || name.contains("\\"))
                    throw new IOException("Unsafe native bundle path: " + name);
                if (required == null || name.equals(required)) found = true;
                if (Files.isRegularFile(output) && expected.equals(sha(output))) continue;
                Files.createDirectories(output.getParent());
                Path temporary = Files.createTempFile(output.getParent(), ".rapid-doc-", ".tmp");
                try {
                    try (InputStream input =
                            NativeLoader.class.getResourceAsStream(resource + name)) {
                        if (input == null)
                            throw new IOException("Missing native resource: " + name);
                        Files.copy(input, temporary, StandardCopyOption.REPLACE_EXISTING);
                    }
                    if (!expected.equals(sha(temporary)))
                        throw new IOException("Native bundle checksum mismatch: " + name);
                    try {
                        try {
                            Files.move(
                                    temporary,
                                    output,
                                    StandardCopyOption.ATOMIC_MOVE,
                                    StandardCopyOption.REPLACE_EXISTING);
                        } catch (java.nio.file.AtomicMoveNotSupportedException e) {
                            Files.move(temporary, output, StandardCopyOption.REPLACE_EXISTING);
                        }
                    } catch (IOException e) {
                        // Another JVM may have installed and loaded this DLL.
                        // Windows cannot replace it, even with identical bytes.
                        if (!Files.isRegularFile(output) || !expected.equals(sha(output))) throw e;
                    }
                } finally {
                    Files.deleteIfExists(temporary);
                }
            }
            if (!found)
                throw new IOException("Required library absent from native bundle: " + required);
            return root;
        } catch (IOException e) {
            throw new RapidDocException("runtime_unavailable", e.getMessage());
        }
    }

    static String platform() {
        String arch = System.getProperty("os.arch").toLowerCase(Locale.ROOT);
        if (!arch.equals("amd64") && !arch.equals("x86_64"))
            throw new RapidDocException(
                    "unsupported", "No bundled native runtime for architecture " + arch);
        String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        if (os.startsWith("windows")) return "windows-x86_64";
        if (os.startsWith("linux")) return "linux-x86_64";
        throw new RapidDocException("unsupported", "No bundled native runtime for OS " + os);
    }

    private static byte[] read(String resource) throws IOException {
        try (InputStream stream = NativeLoader.class.getResourceAsStream(resource)) {
            if (stream == null)
                throw new IOException(
                        "Missing native resource "
                                + resource
                                + "; install the matching rapid-doc-native platform artifact");
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int length;
            while ((length = stream.read(buffer)) != -1) bytes.write(buffer, 0, length);
            return bytes.toByteArray();
        }
    }

    private static String sha(byte[] bytes) {
        try {
            return hex(MessageDigest.getInstance("SHA-256").digest(bytes));
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    private static String sha(Path path) throws IOException {
        try (InputStream input = Files.newInputStream(path)) {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] buffer = new byte[65536];
            int length;
            while ((length = input.read(buffer)) != -1) digest.update(buffer, 0, length);
            return hex(digest.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder(64);
        for (byte value : bytes) result.append(String.format("%02x", value & 255));
        return result.toString();
    }
}
