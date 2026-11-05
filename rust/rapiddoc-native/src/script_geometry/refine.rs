//! 移植inline/scripts.py的token精炼与公式区域分段；不调用Python。
use super::{invalid, limit, number, scalar};
use crate::{
    median, python_printable,
    text_boundary::{python_alnum, python_alpha, python_digit, python_space},
};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::{
    cell::Cell,
    collections::{BTreeSet, HashMap, HashSet},
};
type Box4 = [f64; 4];
const BODY: u8 = 0;
const SUP: u8 = 1;
const SUB: u8 = 2;
fn connector(s: &str) -> bool {
    matches!(s, "^" | "_" | "ˆ" | "~")
}
fn sign(s: &str) -> bool {
    matches!(s, "+" | "-" | "−" | "–" | "⁻" | "⁺")
}
fn operator(s: &str) -> bool {
    matches!(s, "+" | "-" | "−" | "–")
}
fn joiner(s: &str) -> bool {
    matches!(s, "-" | "−" | "–")
}
fn author(s: &str) -> bool {
    matches!(s, "*" | "†" | "‡" | "∗")
}
fn integer(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_bool().map(i64::from))
}
fn raw_box(v: &Value) -> Option<Box4> {
    let a = v.as_array()?;
    if a.len() != 4 {
        return None;
    }
    Some([
        number(&a[0])?,
        number(&a[1])?,
        number(&a[2])?,
        number(&a[3])?,
    ])
}
fn coerce_box(v: &Value) -> Option<Box4> {
    let b = raw_box(v)?;
    let out = [
        b[0].min(b[2]),
        b[1].min(b[3]),
        b[0].max(b[2]),
        b[1].max(b[3]),
    ];
    (out[2] > out[0] && out[3] > out[1]).then_some(out)
}
fn point(v: &Value) -> Option<[f64; 2]> {
    let a = v.as_array()?;
    Some([number(a.first()?)?, number(a.get(1)?)?])
}
pub fn math_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(c) = chars.next() else {
        return false;
    };
    if chars.next().is_some() {
        return false;
    }
    let n = c as u32;
    let r = super::math_identifiers::RANGES;
    let i = r.partition_point(|r| r.0 <= n);
    i > 0 && n <= r[i - 1].1
}
fn all(s: &str, f: fn(char) -> bool) -> bool {
    !s.is_empty() && s.chars().all(f)
}
#[derive(Clone)]
struct Char {
    text: String,
    index: Option<i64>,
    tight: Option<Box4>,
    origin: Option<[f64; 2]>,
    id: bool,
    space: bool,
    alpha: bool,
    digit: bool,
    decimal: bool,
    alnum: bool,
    printable: bool,
}
impl Char {
    fn height(&self) -> f64 {
        self.tight.map_or(0., |b| (b[3] - b[1]).max(0.))
    }
    fn y(&self) -> Option<f64> {
        self.origin.map(|p| p[1])
    }
}
struct Context {
    data: Vec<Char>,
    work: Cell<usize>,
}
impl Context {
    fn new(v: &Value) -> Result<Self, Error> {
        Self::bounded(v, 1500)
    }
    fn bounded(v: &Value, max_chars: usize) -> Result<Self, Error> {
        let rows = v["chars"].as_array().ok_or_else(invalid)?;
        Self::rows(rows, v.get("tight_bboxes"), v.get("origins"), max_chars)
    }
    fn rows(
        rows: &[Value],
        tight: Option<&Value>,
        origins: Option<&Value>,
        max_chars: usize,
    ) -> Result<Self, Error> {
        if rows.len() > rapiddoc_core::resources::limit(max_chars) {
            return Err(limit());
        }
        for map in [tight, origins] {
            if let Some(map) = map {
                if !map.is_object() {
                    return Err(invalid());
                }
                if map.as_object().unwrap().len() > rapiddoc_core::resources::limit(20000) {
                    return Err(limit());
                }
            }
        }
        let mut data = Vec::with_capacity(rows.len());
        let mut bytes = 0;
        for c in rows {
            if !c.is_object() {
                return Err(invalid());
            }
            let text = scalar(c.get("char"))?;
            bytes += text.len();
            if text.len() > rapiddoc_core::resources::limit(256)
                || bytes > rapiddoc_core::resources::limit(1_000_000)
            {
                return Err(limit());
            }
            let index = integer(&c["char_idx"]);
            let key = index.map(|i| i.to_string());
            let tight = key
                .as_ref()
                .and_then(|k| tight.and_then(|map| raw_box(&map[k])));
            let origin = key
                .as_ref()
                .and_then(|k| origins.and_then(|map| point(&map[k])));
            if tight
                .iter()
                .flatten()
                .chain(origin.iter().flatten())
                .any(|n| n.abs() > 1e9)
            {
                return Err(limit());
            }
            data.push(Char {
                id: math_identifier(&text),
                space: all(&text, python_space),
                alpha: all(&text, python_alpha),
                digit: all(&text, python_digit),
                decimal: !text.is_empty()
                    && text.chars().all(|c| crate::python_decimal(c).is_some()),
                alnum: all(&text, python_alnum),
                printable: text.chars().all(python_printable),
                text,
                index,
                tight,
                origin,
            });
        }
        Ok(Self {
            data,
            work: Cell::new(0),
        })
    }
    fn spend(&self, n: usize) -> bool {
        let total = self.work.get().saturating_add(n);
        self.work.set(total);
        total <= rapiddoc_core::resources::limit(10_000_000)
    }
    fn finish(&self) -> Result<(), Error> {
        if self.work.get() > rapiddoc_core::resources::limit(10_000_000) {
            Err(limit())
        } else {
            Ok(())
        }
    }
    fn aligned(&self, a: usize, b: usize) -> bool {
        if !self.spend(1) {
            return false;
        }
        let (a, b) = (&self.data[a], &self.data[b]);
        let (Some(ay), Some(by), Some(ab), Some(bb)) = (a.y(), b.y(), a.tight, b.tight) else {
            return false;
        };
        let scale = a.height().max(b.height()).max(1.);
        (ay - by).abs() <= 0.35f64.max(0.06 * scale)
            && ((ab[1] + ab[3]) / 2. - (bb[1] + bb[3]) / 2.).abs() <= 0.75f64.max(0.3 * scale)
    }
    fn next(&self, i: usize, d: i32) -> Option<usize> {
        let mut j = i as i32 + d;
        while j >= 0 && (j as usize) < self.data.len() {
            if !self.data[j as usize].space {
                return Some(j as usize);
            }
            j += d;
        }
        None
    }
    fn adjacent_base(&self, i: usize, roles: &[u8]) -> bool {
        if !self.spend(1) || i == 0 || roles[i - 1] != BODY || !self.data[i - 1].id {
            return false;
        }
        let (a, b) = (&self.data[i - 1], &self.data[i]);
        let (Some(ay), Some(by), Some(ab), Some(bb)) = (a.y(), b.y(), a.tight, b.tight) else {
            return false;
        };
        (overlap(ab, bb, 1) > 0. || gap(ab, bb) <= 2f64.max(0.5 * a.height()))
            && (by - ay).abs() >= 0.35f64.max(0.08 * a.height())
    }
    fn tokens(&self) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        let mut token = Vec::new();
        for (i, c) in self.data.iter().enumerate() {
            if c.id || connector(&c.text) {
                token.push(i);
            } else if !token.is_empty() {
                out.push(std::mem::take(&mut token));
            }
        }
        if !token.is_empty() {
            out.push(token);
        }
        out
    }
    fn citations(&self, roles: &[u8]) -> HashSet<usize> {
        let mut out = HashSet::new();
        for (start, c) in self.data.iter().enumerate() {
            let endchar = match c.text.as_str() {
                "[" => "]",
                "［" => "］",
                _ => continue,
            };
            for end in start + 1..self.data.len().min(start + 16) {
                if self.data[end].text != endchar {
                    continue;
                }
                if (start + 1..end).any(|i| roles[i] != BODY && self.data[i].alnum) {
                    out.extend(start..=end);
                }
                break;
            }
        }
        out
    }
    fn split(&self, token: &[usize], roles: &[u8]) -> Option<usize> {
        let ids: Vec<_> = token.iter().copied().filter(|i| self.data[*i].id).collect();
        if ids.len() < 2 {
            return None;
        }
        let first = ids[0];
        let tolerance = 0.35f64.max(0.06 * self.data[first].height());
        if token
            .iter()
            .any(|&i| i < first && connector(&self.data[i].text))
        {
            return Some(ids[1]);
        }
        let scripted: Vec<_> = ids[1..]
            .iter()
            .copied()
            .filter(|&i| roles[i] != BODY)
            .collect();
        if roles[first] == BODY && !scripted.is_empty() {
            let p = scripted[0];
            let prefix: Vec<_> = token.iter().copied().filter(|&i| i < p).collect();
            if prefix.len() >= 2
                && prefix
                    .iter()
                    .all(|&i| roles[i] == BODY && self.data[i].alpha)
                && token
                    .iter()
                    .filter(|&&i| i >= p)
                    .all(|&i| roles[i] == SUP && self.data[i].digit)
            {
                return Some(p);
            }
        }
        for &p in &ids[1..] {
            self.spend(p - first);
            if (first + 1..p).any(|i| connector(&self.data[i].text)) {
                return Some(p);
            }
            if let (Some(a), Some(b)) = (self.data[first].y(), self.data[p].y()) {
                if (a - b).abs() > tolerance {
                    return Some(p);
                }
            }
        }
        if roles[first] == BODY {
            scripted.first().copied()
        } else {
            None
        }
    }
    fn close_operators(&self, roles: &mut [u8]) {
        let snapshot = roles.to_vec();
        for (seed, &role) in snapshot.iter().enumerate() {
            if role == BODY || !self.data[seed].id {
                continue;
            }
            let Some(op) = self.next(seed, 1) else {
                continue;
            };
            if op - seed > 3 || !operator(&self.data[op].text) {
                continue;
            }
            let Some(target) = self.next(op, 1) else {
                continue;
            };
            if target - op > 3
                || !self.data[target].id
                || !self.aligned(seed, op)
                || !self.aligned(seed, target)
            {
                continue;
            }
            roles[op] = role;
            roles[target] = role;
        }
    }
    fn close_suffixes(&self, raw: &[u8], roles: &mut [u8]) {
        if roles.len() < 3 {
            return;
        }
        for join in 1..roles.len() - 1 {
            if !joiner(&self.data[join].text) {
                continue;
            }
            let left = join - 1;
            let role = roles[left];
            if role == BODY || !self.data[left].id {
                continue;
            }
            let mut start = left;
            while start > 0 && roles[start - 1] == role && self.data[start - 1].id {
                start -= 1;
            }
            if left - start + 1 < 2
                || start == 0
                || roles[start - 1] != BODY
                || !self.data[start - 1].id
            {
                continue;
            }
            let mut end = join + 1;
            while end < roles.len() && self.data[end].id {
                end += 1;
            }
            if end - (join + 1) < 2
                || (join..end).any(|i| raw[i] != role)
                || !self.aligned(left, join)
                || (join + 1..end).any(|i| !self.aligned(left, i))
            {
                continue;
            }
            roles[join..end].fill(role);
        }
    }
    fn protected_subs(&self, roles: &[u8], tokens: &[Vec<usize>]) -> BTreeSet<usize> {
        let mut out = BTreeSet::new();
        for token in tokens {
            for (p, &i) in token.iter().enumerate() {
                self.spend(p);
                if roles[i] == SUB
                    && self.data[i].id
                    && (token[..p]
                        .iter()
                        .any(|&j| j < i && roles[j] == BODY && self.data[j].id)
                        || self.adjacent_base(i, roles))
                {
                    out.insert(i);
                }
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for (i, &role) in roles.iter().enumerate() {
                if role != SUB || out.contains(&i) || !self.data[i].id {
                    continue;
                }
                let seeds: Vec<_> = out
                    .range(i.saturating_sub(5)..=i.saturating_add(5))
                    .copied()
                    .collect();
                for seed in seeds {
                    if !self.aligned(seed, i) {
                        continue;
                    }
                    let (a, b) = (seed.min(i), seed.max(i));
                    if (a + 1..b).all(|j| {
                        let c = &self.data[j];
                        c.space || connector(&c.text) || sign(&c.text) || c.text == "."
                    }) {
                        out.insert(i);
                        changed = true;
                        break;
                    }
                }
            }
            if self.work.get() > rapiddoc_core::resources::limit(10_000_000) {
                break;
            }
        }
        out
    }
}
fn gap(a: Box4, b: Box4) -> f64 {
    0f64.max(a[0] - b[2]).max(b[0] - a[2])
}
fn overlap(a: Box4, b: Box4, axis: usize) -> f64 {
    0f64.max(a[axis + 2].min(b[axis + 2]) - a[axis].max(b[axis]))
}
fn parse_roles(v: &Value, n: usize) -> Result<Vec<u8>, Error> {
    let a = v.as_array().filter(|a| a.len() == n).ok_or_else(invalid)?;
    a.iter()
        .map(|r| match r.as_str() {
            Some("body") => Ok(BODY),
            Some("sup") => Ok(SUP),
            Some("sub") => Ok(SUB),
            _ => Err(invalid()),
        })
        .collect()
}
fn names(roles: &[u8]) -> Value {
    json!(roles
        .iter()
        .map(|&r| ["body", "sup", "sub"][r as usize])
        .collect::<Vec<_>>())
}
fn boolean(v: &Value, key: &str, default: bool) -> Result<bool, Error> {
    match v.get(key) {
        None => Ok(default),
        Some(b) => b.as_bool().ok_or_else(invalid),
    }
}
fn refine_roles(
    ctx: &Context,
    v: &Value,
    raw: &[u8],
    formula: bool,
    ordinary: bool,
) -> Result<Vec<u8>, Error> {
    let d = &ctx.data;
    let mut refined = raw.to_vec();
    if ordinary && refined.iter().all(|&r| r == BODY) {
        return Ok(refined);
    }
    let citations = ctx.citations(&refined);
    let mut numeric = HashSet::new();
    if raw.iter().zip(d).any(|(&r, c)| r == SUP && c.decimal) {
        let mut request = v.clone();
        request["roles"] = names(raw);
        request["protected_body_indices"] = json!([]);
        let indices = super::numeric_indices(&request)?;
        for i in indices.as_array().unwrap() {
            let i = i.as_u64().unwrap() as usize;
            if raw[i] == SUP {
                numeric.insert(i);
            }
        }
    }
    let tokens = ctx.tokens();
    let splits: Vec<_> = tokens.iter().map(|t| ctx.split(t, &refined)).collect();
    let ids: Vec<Vec<usize>> = tokens
        .iter()
        .map(|t| t.iter().copied().filter(|&i| d[i].id).collect())
        .collect();
    let mut families: HashMap<&str, Vec<usize>> = HashMap::new();
    for (t, alnum) in ids.iter().enumerate() {
        if alnum.len() >= 2 {
            families.entry(&d[alnum[0]].text).or_default().push(t);
        }
    }
    let trusted: HashSet<_> = families
        .into_iter()
        .filter_map(|(base, group)| {
            (group.len() >= 3
                || group.iter().any(|&i| splits[i].is_some())
                || group
                    .iter()
                    .any(|&i| tokens[i].iter().any(|&j| connector(&d[j].text))))
            .then_some(base)
        })
        .collect();
    let mut complex = false;
    for (t, token) in tokens.iter().enumerate() {
        if token.iter().any(|i| citations.contains(i)) || token.iter().all(|i| numeric.contains(i))
        {
            continue;
        }
        let alnum = &ids[t];
        if alnum.is_empty() || token.iter().all(|&i| refined[i] == BODY) {
            continue;
        }
        if token.iter().any(|&i| refined[i] == SUP) && token.iter().any(|&i| refined[i] == SUB) {
            complex = true;
        }
        if alnum.len() == 1 {
            continue;
        }
        let first = alnum[0];
        if refined[first] == SUP
            && alnum[1..].len() >= 2
            && alnum[1..].iter().all(|&i| refined[i] == BODY && d[i].alpha)
        {
            continue;
        }
        let split =
            splits[t].or_else(|| trusted.contains(d[first].text.as_str()).then_some(alnum[1]));
        let Some(split) = split else {
            if alnum.iter().all(|&i| refined[i] != BODY) {
                for &i in token {
                    refined[i] = BODY;
                }
            }
            continue;
        };
        let origins: Vec<_> = alnum
            .iter()
            .filter(|&&i| i < split)
            .filter_map(|&i| d[i].y())
            .collect();
        let heights: Vec<_> = alnum
            .iter()
            .filter(|&&i| i < split)
            .map(|&i| d[i].height())
            .filter(|&h| h > 0.)
            .collect();
        let origin = (!origins.is_empty()).then(|| median(origins));
        let height = if heights.is_empty() {
            0.
        } else {
            median(heights)
        };
        for &i in token {
            if i < split || connector(&d[i].text) {
                refined[i] = BODY;
                continue;
            }
            if !d[i].id || refined[i] != BODY {
                continue;
            }
            if let (Some(base), Some(y)) = (origin, d[i].y()) {
                let shift = y - base;
                if shift.abs() >= 0.35f64.max(0.08 * height) {
                    refined[i] = if shift > 0. { SUB } else { SUP };
                }
            }
        }
    }
    ctx.close_operators(&mut refined);
    let clear = |roles: &mut [u8]| {
        for (i, r) in roles.iter_mut().enumerate() {
            if !citations.contains(&i) {
                *r = BODY;
            }
        }
    };
    if complex && !formula {
        clear(&mut refined);
    }
    let scripted: Vec<_> = refined
        .iter()
        .enumerate()
        .filter_map(|(i, &r)| (r != BODY && d[i].alnum).then_some(i))
        .collect();
    if !formula && scripted.len() >= 2 && d.iter().any(|c| matches!(c.text.as_str(), "∑" | "∫"))
    {
        clear(&mut refined);
    }
    let multiply = d.iter().enumerate().any(|(i, c)| {
        c.text == "×" && i > 0 && i + 1 < d.len() && !d[i - 1].space && !d[i + 1].space
    });
    if !formula && scripted.len() >= 2 && multiply {
        clear(&mut refined);
    }
    if !formula {
        for (i, c) in d.iter().enumerate() {
            if !matches!(c.text.as_str(), "/" | "⁄") {
                continue;
            }
            let nearby: Vec<_> = scripted
                .iter()
                .copied()
                .filter(|&j| i.abs_diff(j) <= 5)
                .collect();
            if nearby.iter().any(|&j| j < i) && nearby.iter().any(|&j| j > i) {
                for j in nearby {
                    if !citations.contains(&j) {
                        refined[j] = BODY;
                    }
                }
            }
        }
    }
    let protected = ctx.protected_subs(&refined, &tokens);
    for (i, &role) in refined.clone().iter().enumerate() {
        if role == SUB && !citations.contains(&i) && !protected.contains(&i) && d[i].id {
            refined[i] = BODY;
        }
    }
    for (i, &role) in refined.clone().iter().enumerate() {
        if role == BODY || citations.contains(&i) {
            continue;
        }
        let s = d[i].text.as_str();
        if d[i].alnum || author(s) {
            continue;
        }
        if matches!(s, "," | "，")
            && i > 0
            && i + 1 < d.len()
            && [i - 1, i + 1]
                .iter()
                .all(|&j| refined[j] == role && (d[j].alnum || author(&d[j].text)))
        {
            continue;
        }
        if s == "."
            && i > 0
            && i + 1 < d.len()
            && [i - 1, i + 1]
                .iter()
                .all(|&j| refined[j] == role && d[j].digit)
        {
            continue;
        }
        if sign(s)
            && [-1, 1]
                .iter()
                .filter_map(|&step| ctx.next(i, step))
                .any(|j| j.abs_diff(i) <= 3 && refined[j] == role && d[j].digit)
        {
            continue;
        }
        if matches!(s, ")" | "）") && role == SUP {
            if let Some(p) = ctx.next(i, -1) {
                if i - p == 1
                    && p > 0
                    && refined[p - 1] == BODY
                    && d[p - 1].alpha
                    && refined[p] == role
                    && raw[i] == role
                    && d[p].alpha
                    && ctx.aligned(p, i)
                {
                    continue;
                }
            }
        }
        refined[i] = BODY;
    }
    if !formula {
        ctx.close_suffixes(raw, &mut refined);
    }
    ctx.finish()?;
    Ok(refined)
}
pub fn refine(v: &Value) -> Result<Value, Error> {
    let ctx = Context::new(v)?;
    let raw = parse_roles(&v["roles"], ctx.data.len())?;
    Ok(names(&refine_roles(
        &ctx,
        v,
        &raw,
        boolean(v, "formula_region", false)?,
        boolean(v, "ordinary_native_roles", false)?,
    )?))
}

fn drop_cap(ctx: &Context) -> Vec<usize> {
    let d = &ctx.data;
    let word: Vec<_> = d
        .iter()
        .enumerate()
        .take_while(|(_, c)| c.text.is_ascii() && c.alpha)
        .map(|(i, _)| i)
        .collect();
    if !(3..=12).contains(&word.len())
        || !d[0].text.bytes().all(|b| b.is_ascii_uppercase())
        || word[1..]
            .iter()
            .any(|&i| !d[i].text.bytes().all(|b| b.is_ascii_lowercase()))
    {
        return Vec::new();
    }
    if word
        .iter()
        .any(|&i| d[i].tight.is_none() || d[i].origin.is_none())
    {
        return Vec::new();
    }
    let cap = d[0].tight.unwrap();
    let h = median(
        word[1..]
            .iter()
            .map(|&i| {
                let b = d[i].tight.unwrap();
                b[3] - b[1]
            })
            .collect(),
    );
    let ch = cap[3] - cap[1];
    let base = median(word[1..].iter().map(|&i| d[i].y().unwrap()).collect());
    if h <= 0.
        || ch < 3. * h
        || base < cap[1] + 0.25 * ch
        || base > cap[1] + 0.7 * ch
        || d[0].y().unwrap() - base < 1.2 * h
        || !(-0.1 * h..=1.5 * h).contains(&(d[1].tight.unwrap()[0] - cap[2]))
        || word[1..].iter().any(|&i| {
            let b = d[i].tight.unwrap();
            (d[i].y().unwrap() - base).abs() > 0.15 * h
                || b[3] - b[1] < 0.5 * h
                || b[3] - b[1] > 1.5 * h
                || b[0] < cap[2] - 0.1 * h
        })
    {
        Vec::new()
    } else {
        word
    }
}
pub fn classify_runs(v: &Value) -> Result<Value, Error> {
    let ctx = Context::new(v)?;
    let n = ctx.data.len();
    let rows = v["chars"].as_array().unwrap();
    let memberships: Vec<Option<i64>> = match v.get("memberships") {
        None => vec![None; n],
        Some(a) => a
            .as_array()
            .filter(|a| a.len() == n)
            .ok_or_else(invalid)?
            .iter()
            .map(|m| {
                if m.is_null() {
                    Ok(None)
                } else {
                    integer(m).map(Some).ok_or_else(invalid)
                }
            })
            .collect::<Result<_, _>>()?,
    };
    let (mut roles, mut counts, mut flags) = (vec![BODY; n], vec![0usize; n], vec![false; n]);
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && memberships[end] == memberships[start] {
            end += 1;
        }
        let mut request = v.clone();
        request["chars"] = json!(&rows[start..end]);
        request["protected_body_indices"] = json!([]);
        let raw = super::classify(&request)?;
        let raw = parse_roles(&raw, end - start)?;
        let local = Context::new(&request)?;
        let mut refined =
            refine_roles(&local, &request, &raw, memberships[start].is_some(), false)?;
        if memberships[start].is_none() && refined.iter().any(|&r| r != BODY) {
            for i in drop_cap(&local) {
                refined[i] = BODY;
            }
        }
        let visible: Vec<_> = local
            .data
            .iter()
            .enumerate()
            .filter_map(|(i, c)| (c.printable && !c.space).then_some(i))
            .collect();
        let count = visible
            .iter()
            .filter(|&&i| refined[i] == BODY && local.data[i].alnum)
            .count();
        let marked = visible.iter().filter(|&&i| refined[i] != BODY).count();
        let body_heights: Vec<_> = visible
            .iter()
            .filter(|&&i| refined[i] == BODY)
            .filter_map(|&i| local.data[i].tight.map(|b| b[3] - b[1]))
            .collect();
        let script_heights: Vec<_> = visible
            .iter()
            .filter(|&&i| refined[i] != BODY)
            .filter_map(|&i| local.data[i].tight.map(|b| b[3] - b[1]))
            .collect();
        let stable = count > 0
            && !body_heights.is_empty()
            && (script_heights.is_empty()
                || body_heights
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
                    >= 1.1
                        * script_heights
                            .iter()
                            .copied()
                            .fold(f64::NEG_INFINITY, f64::max));
        if memberships[start].is_some() && (!stable || marked >= visible.len()) {
            refined.fill(BODY);
        }
        roles[start..end].copy_from_slice(&refined);
        counts[start..end].fill(count);
        flags[start..end].fill(memberships[start].is_some());
        ctx.spend(local.work.get());
        ctx.finish()?;
        start = end;
    }
    Ok(json!({"roles":names(&roles),"body_counts":counts,"formula_flags":flags}))
}

fn page(v: &Value) -> Result<[f64; 2], Error> {
    page_value(&v["page_size"])
}
fn page_value(v: &Value) -> Result<[f64; 2], Error> {
    let a = v.as_array().filter(|a| a.len() == 2).ok_or_else(invalid)?;
    let p = [
        number(&a[0]).ok_or_else(invalid)?,
        number(&a[1]).ok_or_else(invalid)?,
    ];
    if p.iter().any(|&n| n <= 0. || n > 1e9) {
        return Err(limit());
    }
    Ok(p)
}
fn angle(v: &Value) -> Result<i64, Error> {
    if v.is_null() {
        return Ok(0);
    }
    let n = integer(v)
        .or_else(|| {
            v.as_f64()
                .filter(|n| n.is_finite() && n.abs() <= 1e12)
                .map(|n| n.trunc() as i64)
        })
        .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
        .ok_or_else(invalid)?;
    Ok(n.rem_euclid(360))
}
fn rotate(b: Box4, p: [f64; 2], angle: i64) -> Box4 {
    match angle {
        90 => [b[1], p[0] - b[2], b[3], p[0] - b[0]],
        180 => [p[0] - b[2], p[1] - b[3], p[0] - b[0], p[1] - b[1]],
        270 => [p[1] - b[3], b[0], p[1] - b[1], b[2]],
        _ => b,
    }
}
fn rotate_point(b: [f64; 2], p: [f64; 2], angle: i64) -> [f64; 2] {
    match angle {
        90 => [b[1], p[0] - b[0]],
        180 => [p[0] - b[0], p[1] - b[1]],
        270 => [p[1] - b[1], b[0]],
        _ => b,
    }
}
pub fn fraction_members(v: &Value) -> Result<Value, Error> {
    fraction_members_bounded(v, 1500)
}
/// Full-page evidence uses the caller's existing 20,000-character page contract.
/// Line diagnostics retain their original 1,500-character limit.
pub fn fraction_members_page(v: &Value) -> Result<Value, Error> {
    fraction_members_bounded(v, 20000)
}
fn fraction_members_bounded(v: &Value, max_chars: usize) -> Result<Value, Error> {
    let ctx = Context::bounded(v, max_chars)?;
    let rules = v["drawing_lines"]
        .as_array()
        .filter(|r| r.len() <= rapiddoc_core::resources::limit(5000))
        .ok_or_else(limit)?;
    if rules.is_empty() || ctx.data.is_empty() {
        return Ok(json!([]));
    }
    let p = page(v)?;
    let a = angle(&v["angle"])?;
    let mut ids = Vec::new();
    let mut rows = Vec::new();
    let mut heights = Vec::new();
    for c in &ctx.data {
        if let (Some(index), Some(b)) = (c.index, c.tight) {
            if c.printable && !c.space {
                let b = rotate(b, p, a);
                ids.push(index);
                rows.push((b, c.alnum));
                if b[3] - b[1] > 0. {
                    heights.push(b[3] - b[1]);
                }
            }
        }
    }
    let scale = if rows.is_empty() {
        8.
    } else if heights.is_empty() {
        return Err(invalid());
    } else {
        median(heights)
    };
    if rows.len().saturating_mul(rules.len()) > rapiddoc_core::resources::limit(10_000_000) {
        return Err(limit());
    }
    let prepared = match v.get("prepared_rules") {
        None | Some(Value::Null) => None,
        Some(a) => Some(
            a.as_array()
                .filter(|a| a.len() == rules.len())
                .ok_or_else(invalid)?,
        ),
    };
    let mut boxes = Vec::new();
    for (i, r) in rules.iter().enumerate() {
        let b = if let Some(prepared) = prepared {
            if prepared[i].is_null() {
                None
            } else {
                Some(raw_box(&prepared[i]).ok_or_else(invalid)?)
            }
        } else {
            coerce_box(r.get("bbox").unwrap_or(r)).map(|b| rotate(b, p, a))
        };
        if let Some(b) = b {
            if b.iter().any(|n| n.abs() > 1e9) {
                return Err(limit());
            }
            boxes.push(b);
        }
    }
    let positions = docvortex_core::rule_text::fraction_members(&rows, &boxes, scale);
    let members: BTreeSet<_> = positions.into_iter().map(|i| ids[i]).collect();
    Ok(json!(members))
}
fn paired_geometry(ctx: &Context, v: &Value, roles: &[u8]) -> BTreeSet<(usize, u8)> {
    let d = &ctx.data;
    let rows = v["chars"].as_array().unwrap();
    let valid = |i: usize| {
        super::source(&rows[i]).is_some()
            && !d[i].space
            && !matches!(d[i].text.as_str(), "\r" | "\n")
            && d[i].tight.is_some_and(|b| b[2] > b[0] && b[3] > b[1])
            && d[i].origin.is_some()
    };
    let visible: Vec<_> = d
        .iter()
        .enumerate()
        .filter_map(|(i, c)| (c.printable && !c.space).then_some(i))
        .collect();
    let mut out = BTreeSet::new();
    for (pos, t) in visible.windows(3).enumerate() {
        let (base, first, second) = (t[0], t[1], t[2]);
        if ![base, first, second]
            .iter()
            .all(|&i| valid(i) && d[i].alnum)
            || roles[base] != BODY
            || !((roles[first] == SUP && roles[second] == SUB)
                || (roles[first] == SUB && roles[second] == SUP))
        {
            continue;
        }
        let b = d[base].tight.unwrap();
        let a = d[first].tight.unwrap();
        let c = d[second].tight.unwrap();
        let h = d[base].height();
        if h <= 0.
            || [first, second].iter().any(|&i| d[i].height() > 0.8 * h)
            || a[2].min(c[2]) - a[0].max(c[0]) < 0.5 * (a[2] - a[0]).min(c[2] - c[0])
        {
            continue;
        }
        if [first, second].iter().any(|&i| {
            let bb = d[i].tight.unwrap();
            !(-0.15 * h..=0.5 * h).contains(&(bb[0] - b[2]))
                || (d[i].y().unwrap() - d[base].y().unwrap()).abs() < 0.15 * h
                || (d[i].y().unwrap() - d[base].y().unwrap()).abs() > h
        }) {
            continue;
        }
        if (d[first].y().unwrap() - d[base].y().unwrap())
            * (d[second].y().unwrap() - d[base].y().unwrap())
            >= 0.
        {
            continue;
        }
        if let Some(&following) = visible.get(pos + 3) {
            if valid(following)
                && d[following].alnum
                && d[following].height() <= 0.8 * h
                && (d[following].y().unwrap() - d[second].y().unwrap()).abs() <= 0.15 * h
                && (-0.15 * h..=0.5 * h).contains(&(d[following].tight.unwrap()[0] - c[2]))
            {
                continue;
            }
        }
        out.insert((first, roles[first]));
        out.insert((second, roles[second]));
    }
    out
}
/// 旧paired_script_roles，始终以真实基础几何分类作输入，不注入预分类角色。
pub fn paired_indices(v: &Value) -> Result<Vec<usize>, Error> {
    let ctx = Context::new(v)?;
    let raw = super::classify(v)?;
    let roles = parse_roles(&raw, ctx.data.len())?;
    let result = paired_geometry(&ctx, v, &roles);
    ctx.finish()?;
    Ok(result.into_iter().map(|(i, _)| i).collect())
}
fn strong(ctx: &Context, v: &Value, roles: &[u8]) -> BTreeSet<(usize, u8)> {
    let d = &ctx.data;
    let mut out = paired_geometry(ctx, v, roles);
    for i in ctx.citations(roles) {
        if roles[i] != BODY {
            out.insert((i, roles[i]));
        }
    }
    for (i, &role) in roles.iter().enumerate() {
        if role != SUP || !d[i].id {
            continue;
        }
        let h = if i > 0 { d[i - 1].height() } else { 0. };
        if h <= 0. || d[i].height() > 0.8 * h {
            continue;
        }
        if ctx
            .next(i, 1)
            .is_some_and(|j| d[j].id || connector(&d[j].text))
        {
            continue;
        }
        if ctx.adjacent_base(i, roles) {
            out.insert((i, role));
        }
    }
    out
}
pub fn classify_line(v: &Value) -> Result<Value, Error> {
    classify_line_inner(
        &v["line"],
        &v["page_size"],
        v.get("tight_bboxes"),
        v.get("origins"),
        v.get("fraction_members"),
        Some(v),
    )
}
/// Internal page loop: read full maps without cloning their unrelated entries.
/// The JSON entry above retains its complete caller-supplied options.
pub fn classify_line_parts(
    line: &Value,
    page: &Value,
    tight: &Value,
    origins: &Value,
    fractions: &Value,
) -> Result<Value, Error> {
    classify_line_inner(
        line,
        page,
        Some(tight),
        Some(origins),
        Some(fractions),
        None,
    )
}
fn classify_line_inner(
    line: &Value,
    page: &Value,
    tight_map: Option<&Value>,
    origin_map: Option<&Value>,
    fractions: Option<&Value>,
    base: Option<&Value>,
) -> Result<Value, Error> {
    let rows = line["chars"].as_array().ok_or_else(invalid)?;
    if rows.len() > rapiddoc_core::resources::limit(1500) {
        return Err(limit());
    }
    let mut chars: Vec<_> = rows.iter().filter(|c| c.is_object()).cloned().collect();
    if chars.iter().all(|c| integer(&c["char_idx"]).is_some())
        && chars
            .windows(2)
            .any(|c| integer(&c[0]["char_idx"]) > integer(&c[1]["char_idx"]))
    {
        chars.sort_by_key(|c| integer(&c["char_idx"]));
    }
    if chars.is_empty() {
        return Ok(json!({"chars":[],"roles":[],"body_counts":[],"formula_flags":[]}));
    }
    let p = page_value(page)?;
    let a = angle(&line["angle"])?;
    let original = Context::rows(&chars, tight_map, origin_map, 1500)?;
    let mut request = base.cloned().unwrap_or_else(|| json!({}));
    let mut local = chars.clone();
    let mut tight = serde_json::Map::new();
    let mut origins = serde_json::Map::new();
    for (i, c) in chars.iter().enumerate() {
        if let Some(b) = coerce_box(&c["bbox"]) {
            local[i]["bbox"] = json!(rotate(b, p, a));
        }
        if let Some(index) = integer(&c["char_idx"]) {
            if let Some(b) = original.data[i].tight {
                tight.insert(index.to_string(), json!(rotate(b, p, a)));
            }
            if let Some(o) = original.data[i].origin {
                origins.insert(index.to_string(), json!(rotate_point(o, p, a)));
            }
        }
    }
    let regions = match line.get("inline_math_regions") {
        None => Vec::new(),
        Some(a) => a
            .as_array()
            .filter(|a| a.len() <= rapiddoc_core::resources::limit(2000))
            .ok_or_else(limit)?
            .iter()
            .filter_map(coerce_box)
            .collect::<Vec<_>>(),
    };
    let memberships: Vec<_> = original
        .data
        .iter()
        .map(|c| {
            c.tight.and_then(|b| {
                regions.iter().position(|r| {
                    r[0] <= (b[0] + b[2]) / 2.
                        && (b[0] + b[2]) / 2. <= r[2]
                        && r[1] <= (b[1] + b[3]) / 2.
                        && (b[1] + b[3]) / 2. <= r[3]
                })
            })
        })
        .collect();
    request["chars"] = Value::Array(local);
    request["tight_bboxes"] = Value::Object(tight);
    request["origins"] = Value::Object(origins);
    request["memberships"] = json!(memberships);
    let mut result = classify_runs(&request)?;
    let mut roles = parse_roles(&result["roles"], chars.len())?;
    if boolean(line, "compact_formula_cluster", false)?
        || (boolean(line, "restored_inline_cluster", false)? && !regions.is_empty())
    {
        let ctx = Context::new(&request)?;
        let raw = super::classify(&request)?;
        let raw = parse_roles(&raw, chars.len())?;
        roles.fill(BODY);
        for (i, r) in strong(&ctx, &request, &raw) {
            roles[i] = r;
        }
        ctx.finish()?;
    }
    let fraction: BTreeSet<_> = match fractions {
        None => BTreeSet::new(),
        Some(a) => a
            .as_array()
            .filter(|a| a.len() <= rapiddoc_core::resources::limit(20000))
            .ok_or_else(limit)?
            .iter()
            .map(|n| integer(n).ok_or_else(invalid))
            .collect::<Result<_, _>>()?,
    };
    for (i, c) in chars.iter().enumerate() {
        if integer(&c["char_idx"]).is_some_and(|id| fraction.contains(&id)) {
            roles[i] = BODY;
        }
    }
    result["roles"] = names(&roles);
    result["chars"] = Value::Array(chars);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn borrowed_line_maps_keep_roles_and_validation() {
        let line = json!({"angle":0,"chars":[
            {"char":"x","char_idx":1,"bbox":[0,0,10,10]},
            {"char":"2","char_idx":2,"bbox":[10,0,15,5]}
        ]});
        for angle in [0, 90, 180, 270] {
            for compact in [false, true] {
                let mut line = line.clone();
                line["angle"] = json!(angle);
                line["compact_formula_cluster"] = json!(compact);
                for tight in [
                    json!({"1":[0,0,10,10],"2":[10,0,15,5]}),
                    json!(null),
                    json!([]),
                    json!({"1":[0,0,1e10,10]}),
                ] {
                    let q = json!({"line":line,"page_size":[200,300],"tight_bboxes":tight,
                        "origins":{"1":[0,10],"2":[10,5]},"fraction_members":[]});
                    let public = classify_line(&q);
                    let borrowed = classify_line_parts(
                        &q["line"],
                        &q["page_size"],
                        &q["tight_bboxes"],
                        &q["origins"],
                        &q["fraction_members"],
                    );
                    match (public, borrowed) {
                        (Ok(a), Ok(b)) => assert_eq!(a, b),
                        (Err(a), Err(b)) => {
                            assert_eq!(a.code, b.code);
                            assert_eq!(a.message, b.message);
                        }
                        other => panic!("borrowed/public mismatch: {other:?}"),
                    }
                }
            }
        }
    }
    #[test]
    fn original_page_fraction_capacity() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/native-page-fractions.json")).unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert!(fraction_members(&c["request"]).is_err());
            assert_eq!(
                fraction_members_page(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn old_fraction_members() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/script-fractions.json");
        let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                fraction_members(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn old_fraction_line_pipeline() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/script-fraction-pipeline.json");
        let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let mut changed = 0;
        for c in f["cases"].as_array().unwrap() {
            let members = fraction_members(&c["fraction"]).unwrap();
            assert_eq!(members, c["expected_members"], "{}", c["label"]);
            assert_eq!(
                classify_line(&c["line"]).unwrap(),
                c["before"],
                "{}",
                c["label"]
            );
            let mut request = c["line"].clone();
            request["fraction_members"] = members;
            assert_eq!(
                classify_line(&request).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
            changed += usize::from(c["before"]["roles"] != c["expected"]["roles"]);
        }
        assert_eq!(changed, 20);
    }
    #[test]
    fn old_refine_tokens_and_complete_runs() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/script-refinement.json");
        let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let ranges: Vec<(u32, u32)> =
            serde_json::from_value(f["identifier_ranges"].clone()).unwrap();
        assert_eq!(ranges, super::super::math_identifiers::RANGES);
        for c in f["refine"].as_array().unwrap() {
            assert_eq!(
                refine(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        for c in f["runs"].as_array().unwrap() {
            assert_eq!(
                classify_runs(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        for c in f["lines"].as_array().unwrap() {
            assert_eq!(
                classify_line(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
}

#[cfg(test)]
mod production_capacity_regression {
    use super::*;
    #[test]
    fn dense_page_context_keeps_all_records_in_production() {
        let text = "a".repeat(257);
        let rows = vec![json!({"char":text,"bbox":[0,0,1,1]}); 20001];
        assert!(Context::rows(&rows, None, None, 20000).is_err());
        {
            let _scope = rapiddoc_core::resources::python_compatible();
            let context = Context::rows(&rows, None, None, 20000).unwrap();
            assert_eq!(context.data.len(), rows.len());
            assert_eq!(context.data[0].text.len(), 257);
        }
        assert!(Context::rows(&rows, None, None, 20000).is_err());
    }
}
