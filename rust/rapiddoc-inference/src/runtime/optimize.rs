//! All graph edits and ORT preoptimization stay inside session initialization.
use crate::{
    error,
    onnx::{self, Optimization},
};
use ort::{
    environment::Environment,
    session::{builder::SessionBuilder, Session},
};
use rapiddoc_core::{Error, ErrorCode};
use std::path::Path;
fn compile(builder: SessionBuilder, source: &[u8]) -> Result<Vec<u8>, Error> {
    onnx::optimize_with_ort(builder, source)
}
fn cpu_builder(env: &Environment, threads: usize) -> Result<SessionBuilder, Error> {
    let builder = Session::builder(env).map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?;
    let mut builder = builder
        .with_execution_providers([ort::ep::CPU::default()
            .with_arena_allocator(false)
            .build()
            .error_on_failure()])
        .and_then(|b| b.with_inter_threads(1))
        .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?;
    if threads > 0 {
        builder = builder
            .with_intra_threads(threads)
            .map_err(|e| error(ErrorCode::InvalidInput, e))?;
    }
    Ok(builder)
}
pub(crate) fn load(
    mut builder: SessionBuilder,
    env: &Environment,
    path: &Path,
    kind: &Optimization,
    threads: usize,
) -> Result<(Session, Option<Session>), Error> {
    let started = std::time::Instant::now();
    let source = std::fs::read(path).map_err(|e| error(ErrorCode::ModelMissing, e))?;
    let prepared = if *kind == Optimization::Slanet {
        let (encoder, decoder) = onnx::slanet_split(source)?;
        let folded = compile(cpu_builder(env, threads)?, &encoder)?;
        onnx::slanet_finish(encoder, decoder, folded)?
    } else {
        let graph = onnx::before(source, kind)?;
        let graph = if matches!(
            kind,
            Optimization::OcrDetection | Optimization::OcrRecognition | Optimization::Unet
        ) {
            compile(builder.clone(), &graph)?
        } else {
            graph
        };
        onnx::after(graph, kind)?
    };
    let decoder = if let Some(graph) = prepared.decoder.as_ref() {
        Some(
            cpu_builder(env, threads)?
                .commit_from_memory(graph)
                .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?,
        )
    } else {
        None
    };
    let session = builder
        .commit_from_memory(&prepared.encoder)
        .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?;
    eprintln!(
        "rapiddoc: {kind:?} optimized in memory in {:.3}s (conv={}, reductions={}, partition={})",
        started.elapsed().as_secs_f64(),
        prepared.report.convolutions,
        prepared.report.reductions,
        prepared.report.partitioned
    );
    Ok((session, decoder))
}
