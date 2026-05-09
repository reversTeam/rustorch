//! rustorch-cuda build script.
//!
//! Adds the cuSPARSELt library link directives when the `cuda` feature is
//! enabled and we're on Linux. cuSPARSELt is a separate apt package
//! (`libcusparselt0-cuda-13` on Ubuntu) that ships with non-standard
//! search paths, so we point the linker at it explicitly.
//!
//! Search order on Ubuntu 24.04 / aarch64 + sbsa:
//!   /usr/lib/aarch64-linux-gnu/libcusparseLt/13/
//!   /usr/lib/x86_64-linux-gnu/libcusparseLt/13/
//!   /usr/lib/libcusparseLt/13/
//!
//! Skips entirely on macOS or when the `cuda` feature is off.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    {
        // Common cuSPARSELt apt-package install paths.
        let candidate_paths = [
            "/usr/lib/aarch64-linux-gnu/libcusparseLt/13",
            "/usr/lib/x86_64-linux-gnu/libcusparseLt/13",
            "/usr/lib/libcusparseLt/13",
            "/usr/local/cuda/lib64",
        ];
        for path in candidate_paths {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={path}");
            }
        }
        // Link directive — rustc will accept it; the actual symbols are
        // resolved at runtime via LD_LIBRARY_PATH.
        println!("cargo:rustc-link-lib=dylib=cusparseLt");
    }
}
