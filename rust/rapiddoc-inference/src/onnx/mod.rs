//! Process-local ONNX optimization. No generated model files or disk cache.
mod partition;
mod proto;
use proto::{invalid, Field, Msg};
use rapiddoc_core::Error;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Optimization {
    OcrDetection,
    OcrRecognition,
    Layout,
    Slanet,
    Unet,
    Formula,
}
#[derive(Default, Serialize, Debug)]
pub struct Report {
    pub convolutions: usize,
    pub reductions: usize,
    pub layout_mask: bool,
    pub partitioned: bool,
}
pub struct Prepared {
    pub encoder: Vec<u8>,
    pub decoder: Option<Vec<u8>>,
    pub report: Report,
}
fn messages(m: &Msg, tag: u32) -> Result<Vec<Msg>, Error> {
    m.all(tag).iter().map(Field::msg).collect()
}
fn node(op: &str, input: &[String], output: &[String], domain: &str, attributes: Vec<Msg>) -> Msg {
    let mut m = Msg::default();
    m.replace(1, input.iter().map(|s| Field::text(1, s)).collect());
    m.replace(2, output.iter().map(|s| Field::text(2, s)).collect());
    m.set_text(4, op);
    if !domain.is_empty() {
        m.set_text(7, domain);
    }
    m.replace(5, attributes.iter().map(|a| Field::nested(5, a)).collect());
    m
}
fn attr_i(name: &str, n: i64) -> Msg {
    let mut a = Msg::default();
    a.set_text(1, name);
    a.set_int(3, n);
    a.set_int(20, 2);
    a
}
fn attr_s(name: &str, s: &str) -> Msg {
    let mut a = Msg::default();
    a.set_text(1, name);
    a.set_text(4, s);
    a.set_int(20, 3);
    a
}
fn attr_ints(name: &str, ns: &[i64]) -> Msg {
    let mut a = Msg::default();
    a.set_text(1, name);
    a.replace(8, ns.iter().map(|n| Field::integer(8, *n)).collect());
    a.set_int(20, 7);
    a
}
fn attrs(n: &Msg) -> Result<BTreeMap<String, Msg>, Error> {
    messages(n, 5)?
        .into_iter()
        .map(|a| Ok((a.text(1)?, a)))
        .collect()
}
fn ai(a: &BTreeMap<String, Msg>, k: &str, d: i64) -> Result<i64, Error> {
    a.get(k).map(|m| m.int(3, d)).unwrap_or(Ok(d))
}
fn ais(a: &BTreeMap<String, Msg>, k: &str, d: &[i64]) -> Result<Vec<i64>, Error> {
    a.get(k).map(|m| m.ints(8)).unwrap_or(Ok(d.to_vec()))
}
fn native(n: &Msg) -> Result<bool, Error> {
    Ok(matches!(n.text(7)?.as_str(), "" | "ai.onnx"))
}
fn opset(m: &Msg) -> Result<i64, Error> {
    for v in messages(m, 8)? {
        if matches!(v.text(1)?.as_str(), "" | "ai.onnx") {
            return v.int(2, 0);
        }
    }
    Err(invalid("ONNX opset missing"))
}
fn custom_opset(m: &mut Msg) -> Result<(), Error> {
    if !messages(m, 8)?
        .iter()
        .any(|v| v.text(1).ok().as_deref() == Some("onnxruntime_ops"))
    {
        let mut v = Msg::default();
        v.set_text(1, "onnxruntime_ops");
        v.set_int(2, 1);
        m.fields.push(Field::nested(8, &v));
    }
    Ok(())
}
fn tensor_type(v: &Msg) -> Result<(i64, usize), Error> {
    let t = v.sub(2)?.sub(1)?;
    let rank = t
        .one(2)
        .map(Field::msg)
        .transpose()?
        .map(|s| s.all(1).len())
        .unwrap_or(0);
    Ok((t.int(1, 0)?, rank))
}
fn types(g: &Msg) -> Result<BTreeMap<String, (i64, usize)>, Error> {
    let mut r = BTreeMap::new();
    for tag in [11, 12, 13] {
        for v in messages(g, tag)? {
            if let Ok(t) = tensor_type(&v) {
                r.insert(v.text(1)?, t);
            }
        }
    }
    for t in messages(g, 5)? {
        r.insert(t.text(8)?, (t.int(2, 0)?, t.ints(1)?.len()));
    }
    for n in messages(g, 1)? {
        if n.text(4)? == "Constant" {
            if let Some(t) = attrs(&n)?.get("value") {
                let t = t.sub(5)?;
                for o in n.texts(2)? {
                    r.insert(o, (t.int(2, 0)?, t.ints(1)?.len()));
                }
            }
        }
    }
    Ok(r)
}
fn names(g: &Msg) -> Result<BTreeSet<String>, Error> {
    let mut r = BTreeSet::new();
    for tag in [11, 12, 13] {
        for v in messages(g, tag)? {
            r.insert(v.text(1)?);
        }
    }
    for t in messages(g, 5)? {
        r.insert(t.text(8)?);
    }
    for n in messages(g, 1)? {
        r.extend(n.texts(1)?);
        r.extend(n.texts(2)?);
    }
    Ok(r)
}
fn fresh(names: &mut BTreeSet<String>, base: &str) -> String {
    let mut n = base.to_owned();
    while names.contains(&n) {
        n.push('_');
    }
    names.insert(n.clone());
    n
}
fn metadata(m: &mut Msg, key: &str, value: &str) -> Result<(), Error> {
    let mut out = messages(m, 14)?;
    out.retain(|a| a.text(1).ok().as_deref() != Some(key));
    let mut v = Msg::default();
    v.set_text(1, key);
    v.set_text(2, value);
    out.push(v);
    m.replace(14, out.iter().map(|v| Field::nested(14, v)).collect());
    Ok(())
}
fn normalize_metadata(m: &mut Msg) -> Result<(), Error> {
    let mut seen = BTreeMap::new();
    let mut out = vec![];
    for v in messages(m, 14)? {
        let key = v.text(1)?;
        let value = v.text(2)?;
        if let Some(old) = seen.insert(key, value.clone()) {
            if old != value {
                return Err(invalid("Conflicting ONNX metadata"));
            }
        } else {
            out.push(Field::nested(14, &v));
        }
    }
    m.replace(14, out);
    Ok(())
}
fn external(g: &Msg) -> Result<(), Error> {
    for t in messages(g, 5)? {
        if t.int(14, 0)? != 0 || !t.all(13).is_empty() {
            return Err(invalid(
                "In-memory optimization requires embedded ONNX weights",
            ));
        }
    }
    Ok(())
}
fn value_info(name: &str, dtype: i64, dimensions: &[i64]) -> Msg {
    let mut shape = Msg::default();
    for n in dimensions {
        let mut dim = Msg::default();
        if *n >= 0 {
            dim.set_int(1, *n);
        }
        shape.fields.push(Field::nested(1, &dim));
    }
    let mut tensor = Msg::default();
    tensor.set_int(1, dtype);
    tensor.set_sub(2, &shape);
    let mut ty = Msg::default();
    ty.set_sub(1, &tensor);
    let mut info = Msg::default();
    info.set_text(1, name);
    info.set_sub(2, &ty);
    info
}
fn compact_ctc(m: &mut Msg) -> Result<(), Error> {
    let mut g = m.sub(7)?;
    let outputs = messages(&g, 12)?;
    if outputs.len() != 1 || tensor_type(&outputs[0])? != (1, 3) {
        return Err(invalid("CTC compaction requires FP32 [N,T,C]"));
    }
    let name = outputs[0].text(1)?;
    let mut used = names(&g)?;
    let indices = fresh(&mut used, "ortops_ctc_indices");
    let scores = fresh(&mut used, "ortops_ctc_scores");
    let axes = fresh(&mut used, "ortops_ctc_axes");
    g.fields.push(Field::nested(
        1,
        &node(
            "ArgMax",
            std::slice::from_ref(&name),
            std::slice::from_ref(&indices),
            "",
            vec![attr_i("axis", 2), attr_i("keepdims", 0)],
        ),
    ));
    let mut inputs = vec![name.clone()];
    let mut attributes = vec![attr_i("keepdims", 0)];
    if opset(m)? >= 18 {
        let mut t = Msg::default();
        t.set_int(1, 1);
        t.set_int(2, 7);
        t.set_int(7, 2);
        t.set_text(8, &axes);
        g.fields.push(Field::nested(5, &t));
        inputs.push(axes);
    } else {
        attributes.push(attr_ints("axes", &[2]));
    }
    g.fields.push(Field::nested(
        1,
        &node(
            "ReduceMax",
            &inputs,
            std::slice::from_ref(&scores),
            "",
            attributes,
        ),
    ));
    let dims = outputs[0].sub(2)?.sub(1)?.sub(2)?.all(1);
    let make = |name: &str, ty: i64| {
        let mut info = outputs[0].clone();
        info.set_text(1, name);
        let mut v = info.sub(2)?.sub(1)?;
        v.set_int(1, ty);
        let mut shape = Msg::default();
        shape.replace(1, dims[..2].to_vec());
        v.set_sub(2, &shape);
        let mut t = info.sub(2)?;
        t.set_sub(1, &v);
        info.set_sub(2, &t);
        Ok::<_, Error>(info)
    };
    g.replace(
        12,
        vec![
            Field::nested(12, &make(&indices, 7)?),
            Field::nested(12, &make(&scores, 1)?),
        ],
    );
    m.set_sub(7, &g);
    metadata(m, "rapiddoc_ctc_compact", "1")
}
fn compact_layout(m: &mut Msg) -> Result<(), Error> {
    let mut g = m.sub(7)?;
    let mut outputs = messages(&g, 12)?;
    if outputs.len() != 3 || tensor_type(&outputs[2])? != (6, 3) {
        return Err(invalid("Unsupported V3 binary mask output"));
    }
    let original = outputs[2].text(1)?;
    let nodes = messages(&g, 1)?;
    let producers: BTreeMap<_, _> = nodes
        .iter()
        .map(|n| Ok((n.texts(2)?.first().cloned().unwrap_or_default(), n)))
        .collect::<Result<_, Error>>()?;
    let mut name = original.clone();
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(name.clone()) {
            return Err(invalid("Cyclic binary mask producer"));
        }
        let n = producers
            .get(&name)
            .ok_or_else(|| invalid("Missing mask producer"))?;
        if !native(n)? {
            return Err(invalid("Unsupported mask producer domain"));
        }
        match n.text(4)?.as_str() {
            "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual" | "Equal" => break,
            "Identity" | "Reshape" => (),
            "Cast" => {
                if ![1, 11, 10, 16, 6, 7, 2, 9].contains(&ai(&attrs(n)?, "to", 0)?) {
                    return Err(invalid("Unsupported binary mask cast"));
                }
            }
            _ => return Err(invalid("Mask is not proven binary")),
        }
        name = n
            .texts(1)?
            .first()
            .cloned()
            .ok_or_else(|| invalid("Missing mask source"))?;
    }
    let internal = fresh(&mut names(&g)?, &(original.clone() + "__binary_source"));
    for tag in [1, 13] {
        let mut values = messages(&g, tag)?;
        for v in &mut values {
            if tag == 1 {
                for io in [1, 2] {
                    let changed = v
                        .texts(io)?
                        .iter()
                        .map(|s| Field::text(io, if *s == original { &internal } else { s }))
                        .collect();
                    v.replace(io, changed);
                }
            } else if v.text(1)? == original {
                v.set_text(1, &internal);
            }
        }
        g.replace(tag, values.iter().map(|v| Field::nested(tag, v)).collect());
    }
    g.fields.push(Field::nested(
        1,
        &node(
            "Cast",
            &[internal],
            std::slice::from_ref(&original),
            "",
            vec![attr_i("to", 2)],
        ),
    ));
    let mut tensor = outputs[2].sub(2)?.sub(1)?;
    tensor.set_int(1, 2);
    let mut ty = outputs[2].sub(2)?;
    ty.set_sub(1, &tensor);
    outputs[2].set_sub(2, &ty);
    g.replace(12, outputs.iter().map(|v| Field::nested(12, v)).collect());
    m.set_sub(7, &g);
    Ok(())
}
fn formula_graph(g: &mut Msg, inside: bool, count: &mut usize) -> Result<(), Error> {
    let mut nodes = messages(g, 1)?;
    for n in &mut nodes {
        let nested = inside || n.text(4)? == "Loop";
        let mut attributes = messages(n, 5)?;
        for a in &mut attributes {
            if a.one(6).is_some() {
                let mut sub = a.sub(6)?;
                formula_graph(&mut sub, nested, count)?;
                a.set_sub(6, &sub);
            }
            let mut graphs = messages(a, 11)?;
            for sub in &mut graphs {
                formula_graph(sub, nested, count)?;
            }
            if !graphs.is_empty() {
                a.replace(11, graphs.iter().map(|v| Field::nested(11, v)).collect());
            }
        }
        n.replace(5, attributes.iter().map(|v| Field::nested(5, v)).collect());
    }
    if inside {
        let known = types(g)?;
        let mut used = names(g)?;
        for n in &mut nodes {
            if native(n)? && n.text(4)? == "ArgMax" {
                let a = attrs(n)?;
                let inputs = n.texts(1)?;
                if ai(&a, "axis", 0)? != -1
                    || ai(&a, "select_last_index", 0)? != 0
                    || known.get(&inputs[0]).map(|t| t.0) != Some(1)
                {
                    continue;
                }
                let outputs = n.texts(2)?;
                let max = fresh(&mut used, &(outputs[0].clone() + "__max_value"));
                *n = node(
                    "MaxWithIndex",
                    &inputs,
                    &[max, outputs[0].clone()],
                    "onnxruntime_ops",
                    vec![
                        attr_i("keepdims", ai(&a, "keepdims", 1)?),
                        attr_i("largest", 1),
                    ],
                );
                *count += 1;
            }
        }
    }
    g.replace(1, nodes.iter().map(|n| Field::nested(1, n)).collect());
    Ok(())
}
fn tensor_ints(t: &Msg) -> Result<Vec<i64>, Error> {
    if let Some(raw) = t.one(9) {
        let b = raw.data.as_ref();
        if t.int(2, 0)? != 7 || b.len() % 8 != 0 {
            return Err(invalid("Expected int64 initializer"));
        }
        return Ok(b
            .chunks_exact(8)
            .map(|v| i64::from_le_bytes(v.try_into().unwrap()))
            .collect());
    }
    t.ints(7)
}
fn convolution(g: &mut Msg, reductions: bool, report: &mut Report) -> Result<(), Error> {
    let known = types(g)?;
    let mut nodes = messages(g, 1)?;
    let initializers: BTreeMap<_, _> = messages(g, 5)?
        .into_iter()
        .map(|t| Ok((t.text(8)?, t)))
        .collect::<Result<_, Error>>()?;
    if reductions {
        for n in &mut nodes {
            if native(n)? && n.text(4)? == "TopK" {
                let inputs = n.texts(1)?;
                let a = attrs(n)?;
                if inputs.len() == 2
                    && initializers
                        .get(&inputs[1])
                        .map(tensor_ints)
                        .transpose()?
                        .as_deref()
                        == Some(&[1])
                {
                    if let Some(&(1, rank)) = known.get(&inputs[0]) {
                        if rank > 0
                            && (ai(&a, "axis", -1)? + rank as i64) % rank as i64 == rank as i64 - 1
                        {
                            *n = node(
                                "MaxWithIndex",
                                &[inputs[0].clone()],
                                &n.texts(2)?,
                                "onnxruntime_ops",
                                vec![
                                    attr_i("keepdims", 1),
                                    attr_i("largest", ai(&a, "largest", 1)?),
                                ],
                            );
                            report.reductions += 1;
                        }
                    }
                }
            }
        }
        let mut removed = BTreeSet::new();
        for i in 0..nodes.len() {
            if removed.contains(&i) || !native(&nodes[i])? || nodes[i].text(4)? != "ArgMax" {
                continue;
            }
            let a = attrs(&nodes[i])?;
            let input = nodes[i].texts(1)?[0].clone();
            let rank = known.get(&input).map(|v| v.1).unwrap_or(0) as i64;
            if known.get(&input).map(|v| v.0) != Some(1)
                || rank == 0
                || ai(&a, "select_last_index", 0)? != 0
                || (ai(&a, "axis", 0)? + rank) % rank != rank - 1
            {
                continue;
            }
            for j in 0..nodes.len() {
                if j == i
                    || removed.contains(&j)
                    || !native(&nodes[j])?
                    || nodes[j].text(4)? != "ReduceMax"
                    || nodes[j].texts(1)?.first() != Some(&input)
                {
                    continue;
                }
                let b = attrs(&nodes[j])?;
                let inputs = nodes[j].texts(1)?;
                let axes = if inputs.len() > 1 {
                    initializers
                        .get(&inputs[1])
                        .map(tensor_ints)
                        .transpose()?
                        .unwrap_or_default()
                } else {
                    ais(&b, "axes", &[])?
                };
                if axes.len() != 1
                    || (axes[0] + rank) % rank != rank - 1
                    || ai(&a, "keepdims", 1)? != ai(&b, "keepdims", 1)?
                {
                    continue;
                }
                let at = i.min(j);
                let out = vec![nodes[j].texts(2)?[0].clone(), nodes[i].texts(2)?[0].clone()];
                nodes[at] = node(
                    "MaxWithIndex",
                    &[input.clone()],
                    &out,
                    "onnxruntime_ops",
                    vec![
                        attr_i("keepdims", ai(&a, "keepdims", 1)?),
                        attr_i("largest", 1),
                    ],
                );
                removed.insert(i.max(j));
                report.reductions += 1;
                break;
            }
        }
        nodes = nodes
            .into_iter()
            .enumerate()
            .filter_map(|(i, n)| (!removed.contains(&i)).then_some(n))
            .collect();
    }
    for n in &mut nodes {
        if !native(n)? || n.text(4)? != "Conv" {
            continue;
        }
        let inputs = n.texts(1)?;
        if inputs.len() < 2 || known.get(&inputs[1]) != Some(&(1, 4)) {
            continue;
        }
        let a = attrs(n)?;
        let pads = ais(&a, "pads", &[0, 0, 0, 0])?;
        let strides = ais(&a, "strides", &[1, 1])?;
        let dilations = ais(&a, "dilations", &[1, 1])?;
        let auto = a
            .get("auto_pad")
            .map(|v| v.text(4))
            .transpose()?
            .unwrap_or_else(|| "NOTSET".into());
        let auto = if auto.is_empty() { "NOTSET" } else { &auto };
        if pads.len() != 4
            || pads[..2] != pads[2..]
            || pads.iter().any(|v| *v < 0)
            || strides.len() != 2
            || dilations.len() != 2
            || strides.iter().chain(&dilations).any(|v| *v <= 0)
            || !["NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"].contains(&auto)
        {
            continue;
        }
        let name = n.text(3)?;
        let out = n.texts(2)?;
        *n = node(
            "CachedConv2D",
            &inputs,
            &out,
            "onnxruntime_ops",
            vec![
                attr_ints("pads", &pads),
                attr_ints("strides", &strides),
                attr_ints("dilations", &dilations),
                attr_i("group", ai(&a, "group", 1)?),
                attr_i("cache_capacity", 64),
                attr_i("use_tf32", 1),
                attr_s("auto_pad", auto),
                attr_s("conv_api", "classic"),
            ],
        );
        if !name.is_empty() {
            n.set_text(3, &name);
        }
        report.convolutions += 1;
    }
    g.replace(1, nodes.iter().map(|n| Field::nested(1, n)).collect());
    Ok(())
}
/// Rewrite standard-only graph pieces before ORT preoptimization.
pub fn before(source: Vec<u8>, kind: &Optimization) -> Result<Vec<u8>, Error> {
    let mut m = Msg::decode(source)?;
    external(&m.sub(7)?)?;
    normalize_metadata(&mut m)?;
    match kind {
        Optimization::OcrRecognition => compact_ctc(&mut m)?,
        Optimization::Layout => compact_layout(&mut m)?,
        Optimization::Formula => {
            let mut g = m.sub(7)?;
            let mut count = 0;
            formula_graph(&mut g, false, &mut count)?;
            if count == 0 {
                return Err(invalid("No supported formula decoder ArgMax"));
            }
            m.set_sub(7, &g);
            custom_opset(&mut m)?;
        }
        _ => (),
    };
    Ok(m.encode())
}
/// Rewrite the resolved ORT graph; no second ONNX shape inference is necessary.
pub fn after(source: Vec<u8>, kind: &Optimization) -> Result<Prepared, Error> {
    let mut m = Msg::decode(source)?;
    let mut report = Report::default();
    match kind {
        Optimization::OcrDetection | Optimization::OcrRecognition | Optimization::Unet => {
            let mut g = m.sub(7)?;
            convolution(&mut g, *kind == Optimization::OcrRecognition, &mut report)?;
            if report.convolutions == 0 {
                return Err(invalid("No supported CUDA convolutions"));
            }
            m.set_sub(7, &g);
            custom_opset(&mut m)?;
        }
        Optimization::Layout => report.layout_mask = true,
        _ => (),
    };
    Ok(Prepared {
        encoder: m.encode(),
        decoder: None,
        report,
    })
}

pub fn slanet_split(source: Vec<u8>) -> Result<(Vec<u8>, Vec<u8>), Error> {
    partition::split(source)
}
pub fn slanet_finish(
    encoder: Vec<u8>,
    decoder: Vec<u8>,
    folded: Vec<u8>,
) -> Result<Prepared, Error> {
    partition::finish(encoder, decoder, folded)
}

/// ORT 1.23's Compile API defaults to disabled graph optimizations, separately
/// from SessionOptions. The current ort-sys binding exposes only its v22 prefix;
/// use the published append-only v23 slot to preserve Python's ORT_ENABLE_ALL.
/// Reference: microsoft/onnxruntime v1.23.0 core/session/compile_api.cc.
pub fn optimize_with_ort(
    builder: ort::session::builder::SessionBuilder,
    source: &[u8],
) -> Result<Vec<u8>, Error> {
    use ort::{
        compiler::{compile_api, ModelCompiler},
        AsPointer,
    };
    #[repr(C)]
    struct CompileApi23 {
        prefix: ort::sys::OrtCompileApi,
        set_flags: *const std::ffi::c_void,
        set_ep_context_information: *const std::ffi::c_void,
        set_graph_level: unsafe extern "system" fn(
            *mut ort::sys::OrtModelCompilationOptions,
            ort::sys::GraphOptimizationLevel,
        ) -> ort::sys::OrtStatusPtr,
    }
    const _: () =
        assert!(std::mem::size_of::<ort::sys::OrtCompileApi>() == 9 * std::mem::size_of::<usize>());
    let compiler = ModelCompiler::new(builder).map_err(invalid)?;
    let api = compile_api().map_err(invalid)?;
    // SAFETY: api-23 is required by this crate's ORT interface. ORT 1.23 appends
    // slots 9/10/11 after the stable nine-slot v22 prefix, without reordering it.
    // compiler owns a live OrtModelCompilationOptions; no buffers escape it.
    unsafe {
        let api = &*(api as *const ort::sys::OrtCompileApi as *const CompileApi23);
        let status = (api.set_graph_level)(
            compiler.ptr().cast_mut(),
            ort::sys::GraphOptimizationLevel::ORT_ENABLE_ALL,
        );
        ort::Error::result_from_status(status).map_err(invalid)?;
    }
    let compiled = compiler
        .with_model_from_memory(source)
        .and_then(|c| c.with_embed_ep_context())
        .and_then(|c| c.compile_to_buffer())
        .map_err(invalid)?;
    Ok(compiled.as_slice().to_vec())
}
