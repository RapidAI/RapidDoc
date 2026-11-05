import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.github.rapidai.rapiddoc.*;
import java.nio.file.*;
import java.util.*;
import java.security.MessageDigest;

public class ModelCompatProbe {
    public static void main(String[] args) throws Exception {
        Path root = Paths.get(args[0]);
        Map<String, Object> ocr = new HashMap<String, Object>();
        ocr.put("seal_enable", false);
        ocr.put(
                "Det.model_path",
                root.resolve("rapid_doc/resources/PP-OCRv6_det_small.onnx").toString());
        ocr.put(
                "Rec.model_path",
                root.resolve("rapid_doc/resources/PP-OCRv6_rec_small.onnx").toString());
        ObjectMapper mapper = new ObjectMapper();
        ObjectNode all = mapper.createObjectNode();
        try (RapidDoc engine =
                RapidDoc.builder()
                        .device(args.length > 2 ? args[2] : "cpu")
                        .layoutModel("docvortex_rule")
                        .formulaEnabled(false)
                        .tableEnabled(false)
                        .cropSampling("opencv4")
                        .ocrConfig(ocr)
                        .allowDownload(false)
                        .build()) {
            engine.warmup();
            System.out.println("SESSIONS|" + engine.getLoadedModels());
            for (String name : new String[] {"pdf_scan.pdf", "pdf_mixed.pdf"}) {
                DocumentResult result =
                        engine.parse(root.resolve("tests/migration/inputs").resolve(name));
                if (result.getMarkdown().trim().isEmpty())
                    throw new AssertionError("No OCR text: " + name);
                ObjectNode value = all.putObject(name);
                value.set("markdown", result.getMarkdownValue());
                value.set("middle_json", result.getMiddleJson());
                value.set("content_list_json", result.getContentListJson());
                ObjectNode images = value.putObject("images");
                for (Map.Entry<String, byte[]> entry : result.getImages().entrySet()) {
                    StringBuilder hash = new StringBuilder();
                    for (byte b : MessageDigest.getInstance("SHA-256").digest(entry.getValue()))
                        hash.append(String.format("%02x", b & 255));
                    images.put(entry.getKey(), hash.toString());
                }
                System.out.println("PARSED|" + name + "|" + result.getMarkdown().length());
            }
        } catch (RapidDocException error) {
            System.err.println("NATIVE_ERROR_CODE|" + error.getCode());
            throw error;
        }
        mapper.writeValue(Paths.get(args[1]).toFile(), all);
    }
}
