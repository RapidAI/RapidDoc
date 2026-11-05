fn main() {
    println!("cargo:rerun-if-changed=native/clipper.hpp");
    println!("cargo:rerun-if-changed=native/clipper_bridge.cpp");
    cc::Build::new()
        .cpp(true)
        .file("native/clipper_bridge.cpp")
        .flag_if_supported("-std=c++14")
        .compile("rapiddoc_office_clipper_bridge");
}
