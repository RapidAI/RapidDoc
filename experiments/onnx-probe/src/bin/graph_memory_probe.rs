//! Diagnostic graph export only; production initialization never writes graphs.
use ort::session::Session;
use rapiddoc_inference::onnx::{self, Optimization};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let kind: Optimization = serde_json::from_str(&format!("\"{}\"", a[4]))?;
    let env = ort::init_from(&a[1])?.build()?;
    let source = std::fs::read(&a[2])?;
    let out = std::path::Path::new(&a[3]);
    std::fs::create_dir(out)?;
    let builder = Session::builder(&env)?
        .with_intra_threads(1)?
        .with_inter_threads(1)?;
    let builder = if kind != Optimization::Slanet {
        #[cfg(feature = "cuda")]
        {
            builder
                .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])?
        }
        #[cfg(not(feature = "cuda"))]
        {
            builder
        }
    } else {
        builder
    };
    let graph = if kind == Optimization::Slanet {
        let (e, d) = onnx::slanet_split(source)?;
        std::fs::write(out.join("decoder.onnx"), d)?;
        e
    } else {
        onnx::before(source, &kind)?
    };
    std::fs::write(out.join("before.onnx"), &graph)?;
    let compiled = onnx::optimize_with_ort(builder.clone(), &graph)?;
    std::fs::write(out.join("folded.onnx"), compiled.as_slice())?;
    if kind != Optimization::Slanet {
        let p = onnx::after(compiled.to_vec(), &kind)?;
        std::fs::write(out.join("after.onnx"), p.encoder)?;
    }
    Ok(())
}
