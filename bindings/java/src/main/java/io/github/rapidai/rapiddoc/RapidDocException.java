package io.github.rapidai.rapiddoc;

/** Native failure with a stable machine-readable error code. */
public final class RapidDocException extends RuntimeException {
    private static final long serialVersionUID = 1L;
    private final String code;

    public RapidDocException(String code, String message) {
        super(hint(message));
        this.code = code;
    }

    private static String hint(String message) {
        if (System.getProperty("os.name", "").startsWith("Windows")
                && message.contains("onnxruntime")
                && message.contains("LoadLibraryExW failed")) {
            return message
                    + ". Check ORT's dependencies. Older Windows JDKs can preload an incompatible MSVC runtime; use an updated JDK build with compatible MSVC libraries (updating the system redistributable alone may not suffice).";
        }
        return message;
    }

    public String getCode() {
        return code;
    }
}
