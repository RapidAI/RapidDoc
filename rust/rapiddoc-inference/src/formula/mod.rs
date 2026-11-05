//! PP-FormulaNet Plus token decode, old LaTeX repair and ftfy6.3.1 defaults.
use crate::{error, named, Model};
use rapiddoc_core::{Error, ErrorCode};
use regex::Regex;
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashMap},
    sync::LazyLock,
};
use tokenizers::{AddedToken, Tokenizer};

fn check_text(s: &str) -> Result<(), Error> {
    if s.len() > rapiddoc_core::resources::limit(65_536) {
        Err(error(ErrorCode::ResourceLimit, "公式文本超过64KB"))
    } else {
        Ok(())
    }
}
const ENVS: [&str; 12] = [
    "array", "matrix", "pmatrix", "bmatrix", "vmatrix", "Bmatrix", "Vmatrix", "cases", "aligned",
    "gathered", "align", "align*",
];
static ENV_RE: LazyLock<Vec<(Regex, Regex, Regex)>> = LazyLock::new(|| {
    ENVS.iter()
        .map(|e| {
            // 原Python没有escape(env)，align*中的*仍是正则量词，不能改成字面匹配。
            (
                Regex::new(&format!(r"\\begin\{{{}\}}", e)).unwrap(),
                Regex::new(&format!(r"\\end\{{{}\}}", e)).unwrap(),
                Regex::new(&format!(r"\\begin\{{{}\}}\{{([^}}]*)\}}", e)).unwrap(),
            )
        })
        .collect()
});
static CHINESE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\\text[\s\x1c-\x1f]*\{[\s\x1c-\x1f]*([^}]*?[\x{4e00}-\x{9fff}]+[^}]*?)[\s\x1c-\x1f]*\}",
    )
    .unwrap()
});
static REMOVE_PAIR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\left\.?|\\right\.?").unwrap());
static UP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\up([a-zA-Z]+)").unwrap());
static UNSUPPORTED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\\(?:lefteqn|boldmath|ensuremath|centering|textsubscript|sides|textsl|textcent|emph|protect|null)").unwrap()
});
fn count_command(s: &str, name: &str) -> usize {
    s.match_indices(name)
        .filter(|(i, _)| {
            s.as_bytes()
                .get(i + name.len())
                .is_none_or(|c| !c.is_ascii_alphabetic())
        })
        .count()
}
fn pairs(s: &str) -> Result<String, Error> {
    let chars: Vec<_> = s.chars().collect();
    let mut escaped = Vec::with_capacity(chars.len());
    let mut slashes = 0;
    for &c in &chars {
        escaped.push(slashes % 2 == 1);
        slashes = if c == '\\' { slashes + 1 } else { 0 };
    }
    let mut depth = 0usize;
    let mut stack = Vec::new();
    let mut changes = Vec::new();
    let mut i = 0;
    let mut work = 0usize;
    let left: Vec<_> = "\\left".chars().collect();
    let right: Vec<_> = "\\right".chars().collect();
    while i < chars.len() {
        if escaped[i] {
            i += 1;
            continue;
        }
        if i + 5 < chars.len() && chars[i..i + 5] == left {
            stack.push((i, depth));
            i += 6;
            continue;
        }
        if i + 6 < chars.len() && chars[i..i + 6] == right {
            if let Some((pos, d)) = stack.pop() {
                if d != depth {
                    let mut current = d as isize;
                    for j in pos..chars.len() {
                        work += 1;
                        if work > rapiddoc_core::resources::limit(10_000_000) {
                            return Err(error(ErrorCode::ResourceLimit, "公式括号扫描超过预算"));
                        }
                        if !escaped[j] {
                            if chars[j] == '{' {
                                current += 1;
                            } else if chars[j] == '}' {
                                current -= 1;
                                if current < d as isize {
                                    changes.push((i, i + 7, j));
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            i += 7;
            continue;
        }
        if chars[i] == '{' {
            depth += 1;
        } else if chars[i] == '}' {
            depth = depth.saturating_sub(1);
        }
        i += 1;
    }
    // Python list.insert把整个right片段作为一个元素插入，后续索引仍按该混合列表计算。
    changes.sort_by_key(|a| std::cmp::Reverse(a.0));
    let mut result: Vec<_> = chars.iter().map(char::to_string).collect();
    for (start, end, target) in changes {
        let (start, end) = (start.min(result.len()), end.min(result.len()));
        let part = result[start..end].concat();
        result.drain(start..end);
        result.insert(target.min(result.len()), part);
    }
    Ok(result.concat())
}
/// post_process中ftfy之前的完整旧步骤。不要启用旧源码中已注释的normalize。
pub fn repair(text: &str) -> Result<String, Error> {
    check_text(text)?;
    let mut s = CHINESE
        .replace_all(text, |c: &regex::Captures<'_>| c[1].to_owned())
        .replace('"', "");
    if count_command(&s, "\\left") == count_command(&s, "\\right") {
        s = pairs(&s)?;
    } else {
        s = REMOVE_PAIR.replace_all(&s, "").into_owned();
    }
    for (env, (begin, end, format)) in ENVS.iter().zip(ENV_RE.iter()) {
        let (b, e) = (begin.find_iter(&s).count(), end.find_iter(&s).count());
        if b < e {
            let fmt = format
                .captures(&s)
                .map(|c| format!("{{{}}}", &c[1]))
                .unwrap_or_else(|| {
                    if *env == "array" {
                        "{c}".into()
                    } else {
                        String::new()
                    }
                });
            let prefix = format!("\\begin{{{env}}}{fmt} ").repeat(e - b);
            s = prefix + &s;
        } else if b > e {
            s += &format!(" \\end{{{env}}}").repeat(b - e);
        }
        check_text(&s)?;
    }
    s = UP
        .replace_all(&s, |c: &regex::Captures<'_>| {
            if ["arrow", "downarrow", "lus", "silon"].contains(&&c[1]) {
                c[0].to_owned()
            } else {
                format!("\\{}", &c[1])
            }
        })
        .into_owned();
    Ok(UNSUPPORTED.replace_all(&s, "").into_owned())
}
/// 实际旧post_process：先LaTeX修复，再ftfy；ftfy产生的新反斜杠/引号不再回头修复。
pub fn post_process(s: &str) -> Result<String, Error> {
    crate::unicode_fix::fix_text(&repair(s)?)
}
pub struct Decoder {
    tokenizer: Tokenizer,
    token_lengths: HashMap<u32, usize>,
}
impl Decoder {
    pub fn new(metadata: &str) -> Result<Self, Error> {
        if metadata.len() > rapiddoc_core::resources::limit(8_000_000) {
            return Err(error(ErrorCode::ResourceLimit, "公式字典超过8MB"));
        }
        let data: Value =
            serde_json::from_str(metadata).map_err(|e| error(ErrorCode::InvalidInput, e))?;
        let fast = &data["fast_tokenizer_file"];
        if fast["decoder"]["type"] != "ByteLevel" || fast["model"]["type"] != "BPE" {
            return Err(error(
                ErrorCode::Unsupported,
                "仅支持旧公式BPE/ByteLevel字典",
            ));
        }
        if fast["model"]["vocab"]
            .as_object()
            .is_none_or(|v| v.len() > rapiddoc_core::resources::limit(100_000))
        {
            return Err(error(ErrorCode::ResourceLimit, "公式词表无效或过大"));
        }
        let mut tokenizer = Tokenizer::from_bytes(fast.to_string())
            .map_err(|e| error(ErrorCode::InvalidInput, e))?;
        if let Some(config) = data["tokenizer_config_file"].as_object() {
            let empty = serde_json::Map::new();
            let map = match config.get("added_tokens_decoder") {
                None => &empty,
                Some(v) => v.as_object().ok_or_else(|| {
                    error(ErrorCode::InvalidInput, "added_tokens_decoder须为对象")
                })?,
            };
            if map.len() > rapiddoc_core::resources::limit(1000) {
                return Err(error(ErrorCode::ResourceLimit, "公式额外token超过预算"));
            }
            let mut tokens = Vec::new();
            for (index, value) in map {
                let id = index
                    .parse::<u32>()
                    .map_err(|e| error(ErrorCode::InvalidInput, e))?;
                let token: AddedToken = serde_json::from_value(value.clone())
                    .map_err(|e| error(ErrorCode::InvalidInput, e))?;
                tokens.push((id, token));
            }
            tokens.sort_by_key(|t| t.0);
            for (_, token) in tokens {
                if token.special
                    || ["<s>", "</s>", "<unk>", "<pad>"].contains(&token.content.as_str())
                {
                    tokenizer.add_special_tokens(&[token]);
                } else {
                    tokenizer.add_tokens(&[token]);
                }
            }
            for text in ["<s>", "</s>", "<unk>", "<pad>"] {
                if !map.values().any(|v| v["content"] == text) {
                    tokenizer.add_special_tokens(&[AddedToken::from(text, true)]);
                }
            }
        }
        let vocab = tokenizer.get_vocab(true);
        if vocab.len() > rapiddoc_core::resources::limit(100_000) {
            return Err(error(ErrorCode::ResourceLimit, "公式词表超过预算"));
        }
        let mut token_lengths = HashMap::<u32, usize>::new();
        for (text, id) in vocab {
            let entry = token_lengths.entry(id).or_default();
            *entry = (*entry).max(text.len());
        }
        Ok(Self {
            tokenizer,
            token_lengths,
        })
    }
    pub fn decode(&self, rows: &[Vec<u32>]) -> Result<Vec<String>, Error> {
        self.decode_with_work(rows, &mut crate::unicode_fix::Work::default())
    }
    pub(crate) fn decode_with_work(
        &self,
        rows: &[Vec<u32>],
        work: &mut crate::unicode_fix::Work,
    ) -> Result<Vec<String>, Error> {
        if rows.len() > rapiddoc_core::resources::limit(1000)
            || rows
                .iter()
                .any(|r| r.len() > rapiddoc_core::resources::limit(8192))
            || rows.iter().map(Vec::len).sum::<usize>() > rapiddoc_core::resources::limit(128_000)
        {
            return Err(error(ErrorCode::ResourceLimit, "公式token超过预算"));
        }
        let mut output = Vec::with_capacity(rows.len());
        let mut total = 0usize;
        for row in rows {
            let end = row
                .iter()
                .position(|v| *v == 2)
                .map_or(row.len(), |i| i + 1);
            let bytes = row[..end]
                .iter()
                .map(|id| self.token_lengths.get(id).copied().unwrap_or(0))
                .try_fold(0usize, |a, b| a.checked_add(b))
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "公式token文本大小溢出"))?;
            if bytes > rapiddoc_core::resources::limit(65_536) {
                return Err(error(ErrorCode::ResourceLimit, "公式token文本超过预算"));
            }
            let text = self
                .tokenizer
                .decode(&row[..end], true)
                .map_err(|e| error(ErrorCode::InvalidInput, e))?;
            let s = crate::unicode_fix::with_work(&repair(&text)?, work)?;
            total += s.len();
            if total > rapiddoc_core::resources::limit(2_000_000) {
                return Err(error(ErrorCode::ResourceLimit, "公式解码输出超过预算"));
            }
            output.push(s);
        }
        Ok(output)
    }
}
pub enum FormulaImage<'a> {
    Png(&'a [u8]),
    Rgb(&'a rapiddoc_core::pillow_resize::Rgb),
}
enum Pixels<'a> {
    Decoded(rapiddoc_core::pillow_resize::Rgb),
    Borrowed(&'a rapiddoc_core::pillow_resize::Rgb),
}
impl Pixels<'_> {
    fn rgb(&self) -> &rapiddoc_core::pillow_resize::Rgb {
        match self {
            Self::Decoded(rgb) => rgb,
            Self::Borrowed(rgb) => rgb,
        }
    }
}
fn trace_formula_input(
    values: &[f32],
    count: usize,
    side: usize,
    images: &[Pixels<'_>],
    indices: &[usize],
) -> Result<Option<std::path::PathBuf>, Error> {
    let Some(directory) = std::env::var_os("RAPIDDOC_DIAGNOSTIC_FORMULA_DIR") else {
        return Ok(None);
    };
    use sha2::{Digest, Sha256};
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let prefix = std::path::PathBuf::from(directory)
        .join(format!("{:06}", NEXT.fetch_add(1, Ordering::Relaxed)));
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let metadata = serde_json::json!({
        "shape":[count,1,side,side], "sha256":format!("{:x}",Sha256::digest(&bytes)),
        "sources":indices.iter().map(|&i| { let rgb=images[i].rgb(); serde_json::json!({
            "width":rgb.w,"height":rgb.h,"rgb_sha256":format!("{:x}",Sha256::digest(&rgb.data))
        }) }).collect::<Vec<_>>()
    });
    for (extension, data) in [
        ("f32", bytes),
        ("input.json", serde_json::to_vec(&metadata).unwrap()),
    ] {
        std::fs::File::create_new(prefix.with_extension(extension))
            .and_then(|mut f| f.write_all(&data))
            .map_err(|e| error(ErrorCode::InvalidInput, e))?;
    }
    Ok(Some(prefix))
}
#[derive(Default)]
pub struct FormulaTimings {
    pub validation: std::time::Duration,
    pub preprocessing: std::time::Duration,
    pub inference: std::time::Duration,
    pub decoding: std::time::Duration,
}
impl Model {
    pub fn recognize_formula_batch_png(
        &mut self,
        images: &[Vec<u8>],
        batch_size: usize,
    ) -> Result<Vec<String>, Error> {
        let images: Vec<_> = images.iter().map(|png| FormulaImage::Png(png)).collect();
        self.recognize_formula_images(&images, batch_size, &mut FormulaTimings::default())
    }
    /// Same preprocessing, tensors, batching and decoder as the PNG API, with
    /// borrowed RGB crops for native pipelines. PNG inputs are decoded once.
    pub fn recognize_formula_images(
        &mut self,
        images: &[FormulaImage<'_>],
        batch_size: usize,
        timings: &mut FormulaTimings,
    ) -> Result<Vec<String>, Error> {
        self.recognize_formula_images_with_progress(images, batch_size, timings, &mut |_, _| Ok(()))
    }
    /// Report completed crops without changing tensor preparation or batching.
    pub fn recognize_formula_images_with_progress(
        &mut self,
        images: &[FormulaImage<'_>],
        batch_size: usize,
        timings: &mut FormulaTimings,
        progress: &mut dyn FnMut(usize, usize) -> Result<(), Error>,
    ) -> Result<Vec<String>, Error> {
        let started = std::time::Instant::now();
        if batch_size == 0
            || batch_size > rapiddoc_core::resources::limit(64)
            || images.len() > rapiddoc_core::resources::limit(1000)
        {
            return Err(error(ErrorCode::ResourceLimit, "公式batch/图片数超过预算"));
        }
        if images.is_empty() {
            return Ok(Vec::new());
        }
        if self.session.inputs().len() != 1 || self.session.outputs().len() != 1 {
            return Err(error(ErrorCode::Unsupported, "公式模型须单输入/单输出"));
        }
        let shape = self.session.inputs()[0]
            .dtype()
            .tensor_shape()
            .ok_or_else(|| error(ErrorCode::Unsupported, "公式输入不是张量"))?;
        if shape.len() != 4
            || shape[1] != 1
            || !matches!(shape[2], 384 | 768)
            || shape[2] != shape[3]
        {
            return Err(error(
                ErrorCode::Unsupported,
                "仅支持384/768单通道PP-FormulaNet",
            ));
        }
        let side = shape[2] as usize;
        // Static single-image models retain their original legal input shape.
        let batch_size = if shape[0] == 1 { 1 } else { batch_size };
        if self.formula_decoder.is_none() {
            let metadata = self
                .session
                .metadata()
                .map_err(|e| error(ErrorCode::InvalidInput, e))?
                .custom("character")
                .ok_or_else(|| error(ErrorCode::Unsupported, "公式模型缺少character元数据"))?;
            self.formula_decoder = Some(Decoder::new(&metadata)?);
        }
        let input_name = self.session.inputs()[0].name().to_owned();
        let output_name = self.session.outputs()[0].name().to_owned();
        let mut total = 0usize;
        let mut pixels = 0usize;
        // 先验证全部输入；固定批量受现有max_elements约束。
        let mut decoded = Vec::with_capacity(images.len());
        for source in images {
            let bytes = match source {
                FormulaImage::Png(png) => png.len(),
                FormulaImage::Rgb(rgb) => rgb.data.len(),
            };
            total = total
                .checked_add(bytes)
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "公式输入大小溢出"))?;
            if total > rapiddoc_core::resources::limit(128_000_000) {
                return Err(error(ErrorCode::ResourceLimit, "公式输入超过128MB"));
            }
            let image = match source {
                FormulaImage::Png(png) => {
                    Pixels::Decoded(crate::formula_preprocess::decode_png(png)?)
                }
                FormulaImage::Rgb(rgb) => {
                    let count = rapiddoc_core::pillow_resize::size(rgb.w, rgb.h)?;
                    if count > rapiddoc_core::resources::limit(4_000_000) {
                        return Err(error(
                            ErrorCode::ResourceLimit,
                            "Formula source crop exceeds pixel budget",
                        ));
                    }
                    if rgb.data.len() != count * 3 {
                        return Err(error(ErrorCode::InvalidInput, "Invalid formula RGB crop"));
                    }
                    Pixels::Borrowed(rgb)
                }
            };
            let (w, h) = (image.rgb().w, image.rgb().h);
            pixels += w * h;
            if pixels > rapiddoc_core::resources::limit(32_000_000) {
                return Err(error(ErrorCode::ResourceLimit, "公式源图累计像素超过预算"));
            }
            decoded.push(image);
        }
        if batch_size
            .min(images.len())
            .checked_mul(side * side)
            .is_none_or(|n| n > self.max_elements)
        {
            return Err(error(ErrorCode::ResourceLimit, "公式批张量超过预算"));
        }
        if images
            .len()
            .checked_mul(side * side)
            .is_none_or(|n| n > rapiddoc_core::resources::limit(32_000_000))
        {
            return Err(error(ErrorCode::ResourceLimit, "公式累计张量超过预算"));
        }
        // Long and short formulas sharing a decoder batch wait for the longest
        // sequence. Geometry provides an input-only grouping hint; restore every
        // result to its original slot after recognition.
        let mut order: Vec<usize> = (0..decoded.len()).collect();
        if batch_size > 1 {
            order.sort_by(|&a, &b| {
                let x = decoded[a].rgb();
                let y = decoded[b].rgb();
                ((x.w as u64) * (y.h as u64)).cmp(&((y.w as u64) * (x.h as u64)))
            });
        }
        let mut result = vec![String::new(); decoded.len()];
        timings.validation += started.elapsed();
        let mut text_work = crate::unicode_fix::Work::default();
        let mut output_bytes = 0usize;
        let mut work = 0usize;
        let mut done = 0;
        progress(0, images.len())?;
        for batch in order.chunks(batch_size) {
            let started = std::time::Instant::now();
            let mut values = Vec::with_capacity(batch.len() * side * side);
            for &index in batch {
                let rgb = decoded[index].rgb();
                let p = crate::formula_preprocess::prepare(&rgb.data, rgb.w, rgb.h, side, side)?;
                work = work.checked_add(p.work).ok_or_else(|| {
                    error(
                        ErrorCode::ResourceLimit,
                        "Formula preprocessing work counter overflow",
                    )
                })?;
                if work > rapiddoc_core::resources::limit(256_000_000) {
                    return Err(error(
                        ErrorCode::ResourceLimit,
                        "公式累计预处理工作量超过预算",
                    ));
                }
                values.extend(p.values);
            }
            timings.preprocessing += started.elapsed();
            // Explicit developer diagnostic only; never enabled for normal parsing
            // or performance measurements. Preserve exact FP32 bits for replay.
            let trace = trace_formula_input(&values, batch.len(), side, &decoded, batch)?;
            let started = std::time::Instant::now();
            let mut output = self.run_named(BTreeMap::from([(
                input_name.clone(),
                named::Input {
                    shape: vec![batch.len(), 1, side, side],
                    values,
                },
            )]))?;
            timings.inference += started.elapsed();
            let started = std::time::Instant::now();
            let Some(named::Output::Int64 { shape, values }) = output.remove(&output_name) else {
                return Err(error(ErrorCode::Unsupported, "公式输出须int64 token"));
            };
            if let Some(prefix) = trace {
                std::fs::write(
                    prefix.with_extension("tokens.json"),
                    serde_json::to_vec(&serde_json::json!({"shape":shape,"values":values}))
                        .unwrap(),
                )
                .map_err(|e| error(ErrorCode::InvalidInput, e))?;
            }
            if shape.len() != 2
                || shape[0] != batch.len() as i64
                || shape[1] <= 0
                || shape[1] > rapiddoc_core::resources::limit(8192)
            {
                return Err(error(ErrorCode::Unsupported, "公式输出形状不匹配"));
            }
            let rows = values
                .chunks(shape[1] as usize)
                .map(|r| {
                    r.iter()
                        .map(|v| {
                            u32::try_from(*v)
                                .map_err(|_| error(ErrorCode::InvalidInput, "公式token超出u32"))
                        })
                        .collect()
                })
                .collect::<Result<Vec<Vec<u32>>, Error>>()?;
            let decoded = self
                .formula_decoder
                .as_ref()
                .unwrap()
                .decode_with_work(&rows, &mut text_work)?;
            output_bytes += decoded.iter().map(String::len).sum::<usize>();
            if output_bytes > rapiddoc_core::resources::limit(2_000_000) {
                return Err(error(ErrorCode::ResourceLimit, "公式累计输出超过2MB"));
            }
            for (&index, text) in batch.iter().zip(decoded) {
                result[index] = text;
            }
            timings.decoding += started.elapsed();
            done += batch.len();
            progress(done, images.len())?;
        }
        Ok(result)
    }
}
