//! Minimal nvrtc smoke test.

#[cfg(not(feature = "cuda"))]
fn main() {}

#[cfg(feature = "cuda")]
fn main() {
    use cudarc::driver::CudaContext;
    let ctx = CudaContext::new(0).expect("ctx");
    let src = "extern \"C\" __global__ void empty() {}";

    for arch in [
        None,
        Some("sm_80"),
        Some("sm_90"),
        Some("sm_120"),
        Some("sm_121"),
    ] {
        let opts = cudarc::nvrtc::CompileOptions {
            arch,
            ..Default::default()
        };
        match cudarc::nvrtc::compile_ptx_with_opts(src, opts) {
            Ok(ptx) => match ctx.load_module(ptx) {
                Ok(_) => println!("arch={arch:?} : compile + load OK"),
                Err(e) => println!("arch={arch:?} : compile OK, load FAIL {e:?}"),
            },
            Err(e) => println!("arch={arch:?} : compile FAIL {e:?}"),
        }
    }
}
