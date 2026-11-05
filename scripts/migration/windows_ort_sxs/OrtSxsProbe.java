public class OrtSxsProbe {
    private static native long activate();

    private static native long loadOrt();

    private static native void deactivate(long cookie);

    public static void main(String[] args) throws Exception {
        System.out.println(
                System.getProperty("java.vendor")
                        + " "
                        + System.getProperty("java.runtime.version"));
        System.load(args[0]);
        if (args.length > 2 && args[2].equals("native")) {
            long error = loadOrt();
            System.out.println("NATIVE_LOAD_ERROR|" + error);
            if (error != 0) throw new IllegalStateException("ORT load: " + error);
            return;
        }
        long cookie = activate();
        System.out.println("ACTCTX|" + cookie);
        if (cookie <= 0) throw new IllegalStateException("Activation context failed: " + cookie);
        try {
            if (args.length > 2 && args[2].startsWith("model"))
                ModelCompatProbe.main(new String[] {args[3], args[4]});
            else {
                System.load(args[1]);
                System.out.println("ORT LOAD PASSED");
            }
            if (args.length > 2 && args[2].equals("modelhold")) {
                System.out.println("HOLD");
                System.out.flush();
                new java.io.BufferedReader(new java.io.InputStreamReader(System.in)).readLine();
            }
        } finally {
            deactivate(cookie);
        }
    }
}
