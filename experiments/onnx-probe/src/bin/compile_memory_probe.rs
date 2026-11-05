//! Developer probe: export in-memory ORT optimization only to inspect equivalence.
use ort::{
    compiler::ModelCompiler,
    session::{Session, builder::GraphOptimizationLevel},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let env = ort::init_from(&a[1])?.build()?;
    let mut builder = Session::builder(&env)?
        .with_intra_threads(4)?
        .with_inter_threads(1)?;
    if a.get(4).is_some_and(|s| s == "cuda") {
        #[cfg(feature = "cuda")]
        {
            builder = builder
                .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])?;
        }
    }
    if a.get(4).is_some_and(|s| s == "disable") {
        builder = builder.with_optimization_level(GraphOptimizationLevel::Disable)?;
    }
    let source = std::fs::read(&a[2])?;
    let compiled = ModelCompiler::new(builder)?
        .with_model_from_memory(&source)?
        .with_embed_ep_context()?
        .compile_to_buffer()?;
    println!("bytes {} -> {}", source.len(), compiled.len());
    std::fs::write(&a[3], compiled.as_slice())?;
    Ok(())
}
