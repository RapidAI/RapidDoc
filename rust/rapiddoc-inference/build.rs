fn main() {
    println!("cargo:rerun-if-changed=src/seal_offset.cpp");
    println!("cargo:rerun-if-changed=vendor/clipper6");
    let mut offset = cc::Build::new();
    offset
        .cpp(true)
        .std("c++17")
        .file("src/seal_offset.cpp")
        .file("vendor/clipper6/clipper.cpp")
        .include("vendor/clipper6")
        .warnings(false);
    if offset.get_compiler().is_like_msvc() {
        offset.flag("/EHsc");
    }
    offset.compile("rapiddoc_seal_clipper6");
    println!("cargo:rerun-if-changed=src/numpy_argsort.cpp");
    println!("cargo:rerun-if-changed=vendor/x86-simd-sort");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    if !matches!(arch.as_str(), "x86" | "x86_64") {
        return;
    }
    for isa in ["avx2", "avx512"] {
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .std("c++17")
            .file("src/numpy_argsort.cpp")
            .include("vendor/x86-simd-sort/src")
            .opt_level(2);
        build.define(
            "RAPIDDOC_SORT_NAME",
            Some(format!("rd_numpy_argsort_{isa}").as_str()),
        );
        if build.get_compiler().is_like_msvc() {
            build.flag("/EHsc").flag(if isa == "avx2" {
                "/arch:AVX2"
            } else {
                "/arch:AVX512"
            });
            if isa == "avx2" {
                build.define("__AVX2__", None);
            } else {
                for flag in ["__AVX512DQ__", "__AVX512VL__", "__AVX512BW__"] {
                    build.define(flag, None);
                }
            }
        } else {
            build.flag("-mavx2");
            if isa == "avx512" {
                for flag in [
                    "-mavx512f",
                    "-mavx512dq",
                    "-mavx512vl",
                    "-mavx512bw",
                    "-mavx512cd",
                ] {
                    build.flag(flag);
                }
            }
        }
        build.compile(&format!("rapiddoc_numpy_sort_{isa}"));
    }
}
