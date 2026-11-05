# libjpeg-turbo otherwise forces /MT[d], conflicting with Rust's default /MD.
# Preserve an explicit -C target-feature=+crt-static build.
if(CMAKE_HOST_WIN32 AND (NOT CMAKE_SYSTEM_NAME OR CMAKE_SYSTEM_NAME STREQUAL "Windows"))
  if(NOT "$ENV{CARGO_ENCODED_RUSTFLAGS}" MATCHES "[+]crt-static")
    set(WITH_CRT_DLL ON CACHE BOOL "Use Cargo's dynamic MSVC runtime")
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
  endif()
endif()
