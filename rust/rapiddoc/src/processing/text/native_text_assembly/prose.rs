//! Original natural-paragraph and explicit metadata/list boundaries.
use super::*;
fn terminal(t: &str) -> bool {
    regex(r#"[.!?。！？][\])’"']*$"#).is_match(t.trim_end_matches(python_space))
}
pub(super) fn prose(s: &State, l: &Lane, regular: f64, mad: f64) -> Result<BTreeSet<i64>, Error> {
    let rs = ordered(s, l, false, false);
    let w = (l.right - l.left).max(0.1);
    let body = rs
        .iter()
        .filter(|r| ordinary(s, r) && s.l(r)["paragraph_group"].is_null() && width(r) >= 0.7 * w)
        .collect::<Vec<_>>();
    let mut out = BTreeSet::new();
    if body.len() < 3 {
        return Ok(out);
    }
    let em = median(body.iter().map(|r| s.h(r)).collect::<Result<Vec<_>, _>>()?)?;
    let indents = body
        .iter()
        .filter(|r| (0.65 * em..=3. * em).contains(&(r.b.n[0] - l.left)))
        .map(|r| r.b.n[0])
        .collect::<Vec<_>>();
    let mut item = false;
    let mut caption: Option<f64> = None;
    for (i, p) in rs.windows(2).enumerate() {
        let (a, b) = (&p[0], &p[1]);
        let (al, bl) = (s.l(a), s.l(b));
        let (ah, bh) = (s.h(a)?, s.h(b)?);
        if !ordinary(s, a) || b.b.n[1] - a.b.n[3] > 3. * em {
            item = false;
            caption = None
        }
        if al["caption_start"] == true {
            caption = Some(a.b.n[0])
        }
        if caption.is_some_and(|x| (b.b.n[0] - x).abs() > em) {
            caption = None
        }
        if caption.is_some() {
            if let Some(c) = rs.get(i + 2) {
                if al["paragraph_terminal"] == true
                    && ordinary(s, a)
                    && ordinary(s, b)
                    && ordinary(s, c)
                    && bl["caption_start"] != true
                    && s.l(c)["caption_start"] != true
                    && (0.7 * ah..=3. * ah).contains(&s.body_gap(a, b)?)
                    && (1.05 * ah..=1.25 * ah).contains(&bh)
                    && (0.95 * bh..=1.05 * bh).contains(&s.h(c)?)
                    && width(b) >= 0.7 * w
                    && width(c) >= 0.7 * w
                    && fonts(bl, s.l(c))?
                {
                    caption = None
                }
            }
        }
        if regex(r"^\s*(?:[•●▪]|\d+[.)](?:\s|$))").is_match(s.t(a)?)
            || regex(r"^[［\[]\s*\d+\s*[］\]]").is_match(s.t(a)?)
        {
            item = true
        }
        let text = s.t(b)?.trim_matches(python_space);
        let upper = regex(r"^[A-Z][A-Za-z]+").find(text).is_some_and(|m| {
            !text[m.end()..]
                .chars()
                .next()
                .is_some_and(|c| c == '_' || python_alnum(c))
        });
        if integer(al, "angle", 0)? != 0
            || integer(bl, "angle", 0)? != 0
            || !ordinary(s, a)
            || !ordinary(s, b)
            || !al["paragraph_group"].is_null()
            || !bl["paragraph_group"].is_null()
            || math(s, a)
            || math(s, b)
            || item
            || caption.is_some()
            || !(al["paragraph_terminal"] == true || terminal(s.t(a)?))
            || !upper
            || width(b) < 0.7 * w
        {
            continue;
        }
        let gap = s.body_gap(a, b)?;
        let ink_a = if al["ink_bbox"].is_null() {
            a.b.clone()
        } else {
            bbox(&al["ink_bbox"])?
        };
        let ink_b = if bl["ink_bbox"].is_null() {
            b.b.clone()
        } else {
            bbox(&bl["ink_bbox"])?
        };
        if ink_b.n[1] < ink_a.n[3] - 0.15 * em {
            continue;
        }
        if (1.15 * em).max(regular + 0.85 * em + 3. * mad) <= gap
            && gap <= 3. * em
            && (b.b.n[0] - l.left).abs() <= 0.35 * em
            && (a.b.n[0] - l.left).abs() <= 0.35 * em
            && fonts(al, bl)?
            && ah.max(bh) <= 1.2 * ah.min(bh)
        {
            out.insert(s.id(b)?);
            continue;
        }
        if !(-0.25 * em..=regular + (0.75 * em).max(3. * mad)).contains(&gap) {
            continue;
        }
        let repeated = indents
            .iter()
            .filter(|x| (**x - b.b.n[0]).abs() <= 0.25 * em)
            .count()
            >= 2;
        let returns = if let Some(c) = rs.get(i + 2) {
            ordinary(s, c)
                && (c.b.n[0] - l.left).abs() <= 0.35 * em
                && width(c) >= 0.65 * w
                && fonts(bl, s.l(c))?
        } else {
            false
        };
        let indented =
            (0.65 * em..=3. * em).contains(&(b.b.n[0] - l.left)) && (returns || repeated);
        let word = text.split(python_space).next().ok_or_else(invalid)?;
        let glyph = bl
            .get("median_glyph_width")
            .filter(|n| !n.is_null())
            .map(num)
            .transpose()?
            .filter(|&n| n != 0.)
            .unwrap_or(0.5 * em);
        let short = (b.b.n[0] - l.left).abs() <= 0.35 * em
            && (a.b.n[0] - l.left).abs() <= 0.35 * em
            && width(a) <= 0.8 * w
            && l.right - a.b.n[2] >= word.chars().count() as f64 * glyph + 2. * em
            && fonts(al, bl)?;
        if indented || short {
            out.insert(s.id(b)?);
        }
    }
    Ok(out)
}
pub(super) fn cjk_prose(
    s: &State,
    l: &Lane,
    regular: f64,
    mad: f64,
) -> Result<BTreeSet<i64>, Error> {
    let rs = ordered(s, l, false, false);
    let w = (l.right - l.left).max(0.1);
    let mut out = BTreeSet::new();
    for p in rs.windows(2) {
        let (a, b) = (&p[0], &p[1]);
        let mut skip = false;
        for r in [a, b] {
            skip |= !clean(s, r) || math(s, r) || integer(s.l(r), "angle", 0)? != 0
        }
        if skip {
            continue;
        }
        let h = s.h(a)?.max(s.h(b)?);
        if h <= 0.
            || s.l(a)["font_signature"].is_null()
            || !fonts(s.l(a), s.l(b))?
            || s.h(a)?.min(s.h(b)?) < 0.85 * h
            || (b.b.n[0] - a.b.n[0]).abs() > 0.25 * h
            || !regex(r"[。！？][”’）]*$").is_match(s.t(a)?.trim_end_matches(python_space))
            || cjk(s.t(a)?) < 4
            || cjk(s.t(b)?) < 8
        {
            continue;
        }
        let t = s.t(b)?.trim_matches(python_space);
        let labelled = regex(r"^[㐀-鿿]{1,8}[：:]").is_match(t);
        if !regex(r"^[㐀-鿿]").is_match(t) || width(b) < 0.65 * w && !labelled {
            continue;
        }
        let threshold = ((if labelled { 0.5 } else { 0.65 }) * h)
            .max(regular + (if labelled { 0.2 } else { 0.3 }) * h + 3. * mad);
        if (threshold..=3. * h).contains(&s.body_gap(a, b)?) {
            out.insert(s.id(b)?);
        }
    }
    Ok(out)
}
pub(super) fn cjk_entries(s: &State, l: &Lane) -> Result<BTreeSet<i64>, Error> {
    let mut rs = l
        .rows
        .iter()
        .filter(|r| clean(s, r))
        .cloned()
        .collect::<Vec<_>>();
    s.ordered(&mut rs, false);
    let mut nums = vec![];
    let re = regex(r"^\s*(\d{1,2})、\s*[㐀-鿿]");
    for (i, r) in rs.iter().enumerate() {
        if let Some(c) = re.captures(s.t(r)?) {
            nums.push((i, r, number(&c[1])?));
        }
    }
    let mut out = BTreeSet::new();
    for p in nums.windows(3) {
        let h = s.h(p[0].1)?.max(s.h(p[1].1)?).max(s.h(p[2].1)?);
        if p[1].2 != p[0].2 + 1 || p[2].2 != p[1].2 + 1 || s.l(p[0].1)["font_signature"].is_null() {
            continue;
        }
        let mut ok = true;
        for x in &p[1..] {
            ok &= fonts(s.l(p[0].1), s.l(x.1))?
                && (x.1.b.n[0] - p[0].1.b.n[0]).abs() <= 0.3 * h
                && s.h(x.1)? >= 0.85 * h;
        }
        for ab in p.windows(2) {
            let gap = ab[1].1.b.n[1] - ab[0].1.b.n[3];
            ok &= (1..=8).contains(&(ab[1].0 - ab[0].0)) && gap > 0. && gap <= 10. * h;
        }
        if ok {
            for x in p {
                out.insert(s.id(x.1)?);
            }
        }
    }
    let books = rs
        .iter()
        .enumerate()
        .filter(|(_, r)| regex(r"^《[㐀-鿿]").is_match(s.t(r).unwrap().trim_matches(python_space)))
        .collect::<Vec<_>>();
    if books.len() >= 3 {
        for p in books.windows(2) {
            let ((i, a), (j, b)) = (p[0], p[1]);
            let tail = &rs[j - 1];
            let h = s.h(a)?.max(s.h(b)?);
            if (1..=4).contains(&(j - i))
                && !s.l(a)["font_signature"].is_null()
                && fonts(s.l(a), s.l(b))?
                && (a.b.n[0] - b.b.n[0]).abs() <= 0.3 * h
                && s.h(b)? >= 0.85 * h
                && regex(r"[—–-]+\s*\d{4}[-./]\d{1,2}[-./]\d{1,2}\s*$").is_match(s.t(tail)?)
                && (-0.1 * h..=h).contains(&(b.b.n[1] - tail.b.n[3]))
            {
                out.insert(s.id(a)?);
                out.insert(s.id(b)?);
            }
        }
    }
    Ok(out)
}
pub(super) fn typography(
    s: &State,
    l: &Lane,
    regular: f64,
    mad: f64,
) -> Result<BTreeSet<i64>, Error> {
    let rs = ordered(s, l, true, true);
    let w = (l.right - l.left).max(0.1);
    let repeated_e = rs
        .iter()
        .filter(|r| {
            s.l(r)["leading_emphasis_width"]
                .as_f64()
                .is_some_and(|n| (0.03 * w..=0.65 * w).contains(&n))
        })
        .count()
        >= 3;
    let repeated_t = rs
        .iter()
        .filter(|r| {
            s.l(r)["leading_typography_width"]
                .as_f64()
                .is_some_and(|n| (0.02 * w..=0.2 * w).contains(&n))
        })
        .count()
        >= 3;
    let mut out = BTreeSet::new();
    for (i, p) in rs.windows(2).enumerate() {
        let (a, b) = (&p[0], &p[1]);
        let (al, bl) = (s.l(a), s.l(b));
        let h = s.h(a)?.max(s.h(b)?);
        let aw = width(a);
        let bw = width(b);
        let gap = s.body_gap(a, b)?;
        let common = clean(s, a)
            && clean(s, b)
            && !math(s, a)
            && !math(s, b)
            && optional(bl, "inline_math_regions", 1000)?.is_empty();
        if aw <= 0.65 * w
            && (0.4 * w..=0.85 * w).contains(&bw)
            && (a.b.n[0] - l.left).abs() <= 0.35 * h
            && (b.b.n[0] - l.left).min(l.right - b.b.n[2]) >= 2. * h
            && (cx(&b.b) - 0.5 * (l.left + l.right)).abs() <= 0.5 * h
            && s.h(a)?.min(s.h(b)?) >= 0.85 * h
            && regex(r#"[.!?。！？][\])’"'*]*$"#).is_match(s.t(a)?.trim_end_matches(python_space))
            && regex(r#"[.!?。！？][\])’"'*]*$"#).is_match(s.t(b)?.trim_end_matches(python_space))
            && (0.75 * h).max(regular + 0.5 * h + 3. * mad) <= gap
            && gap <= 2. * h
            && common
        {
            out.insert(s.id(b)?);
            continue;
        }
        if repeated_t
            && bl["leading_typography_width"]
                .as_f64()
                .is_some_and(|n| (0.02 * w..=0.2 * w).contains(&n))
            && bw >= 0.75 * w
            && (a.b.n[0] - l.left).abs() <= 0.35 * h
            && (b.b.n[0] - l.left).abs() <= 0.35 * h
            && (0.9 * h).max(regular + 0.55 * h + 3. * mad) <= gap
            && gap <= 3. * h
            && fonts(al, bl)?
            && s.h(a)?.min(s.h(b)?) >= 0.85 * h
            && common
        {
            out.insert(s.id(b)?);
            continue;
        }
        let local = if let Some(c) = rs.get(i + 2) {
            aw <= 0.45 * w
                && bl["dominant_font_weight"].as_f64().unwrap_or(0.) >= 550.
                && !s.l(c)["dominant_font_weight"].is_null()
                && num(&bl["dominant_font_weight"])? >= num(&s.l(c)["dominant_font_weight"])? + 100.
                && s.l(c)["leading_emphasis_width"].is_null()
                && width(c) >= 0.75 * w
                && (c.b.n[0] - l.left).abs() <= 0.5 * h
                && (0.85 * h..=1.15 * h).contains(&s.h(c)?)
        } else {
            false
        };
        let explicit = (repeated_e || local)
            && bl["leading_emphasis_width"]
                .as_f64()
                .is_some_and(|n| n <= 0.65 * w)
            && terminal(s.t(a)?);
        if (bl["leading_typography_width"].is_null() && !explicit)
            || (bl["leading_typography_width"]
                .as_f64()
                .is_some_and(|n| n > 0.2 * w)
                && !explicit)
            || (aw > 0.45 * w && !explicit)
            || bw < 0.75 * w
            || (a.b.n[0] - l.left).abs() > 0.75 * h
            || (b.b.n[0] - l.left).abs() > 0.75 * h
            || math(s, b)
            || !optional(bl, "inline_math_regions", 1000)?.is_empty()
        {
            continue;
        }
        if (-0.25 * h..=regular + (0.75 * h).max(3. * mad)).contains(&gap) {
            out.insert(s.id(b)?);
        }
    }
    Ok(out)
}
pub(super) fn explicit(s: &State, l: &Lane) -> Result<BTreeSet<i64>, Error> {
    let rs = ordered(s, l, true, true);
    let mut out = BTreeSet::new();
    let mut nums = vec![];
    let w = (l.right - l.left).max(0.1);
    for r in &rs {
        let t = s.t(r)?;
        if regex(r"(?i)^\s*(?:abstract|摘\s*要)\s*[:：]").is_match(&fold(t)) {
            out.insert(s.id(r)?);
        }
        if let Some(c) = regex(r"^(\d{1,2})[.]\s+\D").captures(t) {
            nums.push((r, number(&c[1])?));
        }
        if regex(r"^\s*[•●▪]").is_match(t) && width(r) >= 0.8 * w {
            out.insert(s.id(r)?);
        }
    }
    for p in nums.windows(2) {
        let h = s.h(p[0].0)?.max(s.h(p[1].0)?);
        if p[1].1 == p[0].1 + 1 && (p[0].0.b.n[0] - p[1].0.b.n[0]).abs() <= h {
            out.insert(s.id(p[0].0)?);
            out.insert(s.id(p[1].0)?);
        }
    }
    for (i, p) in rs.windows(2).enumerate() {
        let (a, b) = (&p[0], &p[1]);
        let h = s.h(a)?.max(s.h(b)?);
        if regex(&format!(r"(?i)^\s*e(?:{SPACE}|-)*mail\s*[:：]")).is_match(&fold(s.t(b)?))
            && (b.b.n[0] - a.b.n[0]).abs() <= 0.75 * h
        {
            out.insert(s.id(b)?);
        }
        let label = regex(r"^\s*([A-Za-z㐀-鿿]{1,12})\s*[:：]");
        if let (Some(ac), Some(bc)) = (label.captures(s.t(a)?), label.captures(s.t(b)?)) {
            if !regex(r"(?i)^\s*(?:https?://|www\.)").is_match(&fold(s.t(b)?))
                && ac[1].chars().count() >= 4
                && bc[1].chars().count() >= 4
                && cjk(&ac[1]) > 0
                && cjk(&bc[1]) > 0
                && ac[1].to_lowercase() != bc[1].to_lowercase()
                && b.b.n[1] - a.b.n[1] <= 2. * h
                && width(a) <= 0.75 * w
                && width(b) <= 0.75 * w
            {
                out.insert(s.id(b)?);
            }
        }
        let cont = rs.get(i + 2).is_some_and(|c| {
            b.b.n[0] - l.left >= 5f64.max(0.65 * h)
                && c.b.n[0] - l.left <= 0.5 * h
                && (0.5 * h..=2.25 * h).contains(&(c.b.n[1] - b.b.n[1]))
        });
        if regex(r"(?i)^\s*(?:[（(]\s*(?:\d+|[ivxlcdm]+)\s*[）)]|[①-⑳]|[•●▪])")
            .is_match(&fold(s.t(b)?))
            && s.t(a)?
                .trim_end_matches(python_space)
                .ends_with([':', '：'])
            && width(a) <= 0.8 * w
            && cont
        {
            out.insert(s.id(b)?);
        }
    }
    Ok(out)
}
