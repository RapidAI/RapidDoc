//! Original geometric/typographic text-row barriers.
use super::*;
pub(super) fn top(s: &State, l: &Lane, page_h: f64) -> Result<BTreeSet<i64>, Error> {
    let rs = ordered(s, l, false, false);
    let mut out = BTreeSet::new();
    for i in 1..rs.len().saturating_sub(2) {
        let band = &rs[..i];
        let body = &rs[i..i + 3];
        if band.iter().any(|r| r.b.n[3] > 0.08 * page_h) {
            break;
        }
        if band
            .iter()
            .chain(body)
            .any(|r| !ordinary(s, r) || s.l(r)["caption_start"] == true)
        {
            continue;
        }
        let h = median(body.iter().map(|r| s.h(r)).collect::<Result<Vec<_>, _>>()?)?;
        if h <= 0. || body[0].b.n[1] > 0.16 * page_h {
            continue;
        }
        let mut ok = true;
        for r in body {
            ok &=
                (0.85 * h..=1.15 * h).contains(&s.h(r)?) && (r.b.n[0] - body[0].b.n[0]).abs() <= h;
        }
        if !ok
            || body[..2].iter().any(|r| {
                regex(r"[A-Za-z]+|[一-鿿]")
                    .find_iter(s.t(r).unwrap())
                    .count()
                    < 4
            })
            || body
                .windows(2)
                .any(|p| !(-0.4 * h..=h).contains(&(p[1].b.n[1] - p[0].b.n[3])))
            || body[0].b.n[1]
                - band
                    .iter()
                    .map(|r| r.b.n[3])
                    .fold(f64::NEG_INFINITY, f64::max)
                < 0.5 * h
        {
            continue;
        }
        for r in band {
            ok &= (regex(r"^\d{1,4}$").is_match(s.t(r)?.trim_matches(python_space))
                && width(r) <= 4. * h)
                || s.h(r)? <= 0.95 * h;
        }
        if ok {
            out.insert(s.id(&body[0])?);
            break;
        }
    }
    Ok(out)
}
pub(super) fn bullets(s: &State, rs: &[Row]) -> Result<BTreeSet<i64>, Error> {
    let mut cs: Vec<(&Row, Option<&Row>)> = Vec::new();
    let count = |r: &Row| -> Result<usize, Error> {
        Ok(regex(r"[A-Za-z]{2,}|[㐀-鿿]").find_iter(s.t(r)?).count())
    };
    for r in rs {
        if !clean(s, r) || !regex(r"^\s*[•●▪]").is_match(s.t(r)?) {
            continue;
        }
        let h = s.h(r)?;
        let t = s.t(r)?.trim_matches(python_space);
        let body = if ["•", "●", "▪"].contains(&t) {
            if width(r) > h {
                continue;
            }
            let mut hosts = Vec::new();
            for x in rs {
                if x.index != r.index
                    && clean(s, x)
                    && s.l(x)["formula_candidate_only"] != true
                    && (0.0..=3.5 * h).contains(&(x.b.n[0] - r.b.n[2]))
                    && axis(&r.b, &x.b, 1) >= 0.5
                    && (0.7 * h..=1.5 * h).contains(&s.h(x)?)
                    && count(x)? >= 2
                {
                    hosts.push(x)
                }
            }
            if hosts.len() != 1 {
                continue;
            }
            Some(hosts[0])
        } else {
            if count(r)? < 2 {
                continue;
            }
            None
        };
        cs.push((r, body));
    }
    let mut out = BTreeSet::new();
    for (r, body) in &cs {
        let h = s.h(r)?;
        for (x, _) in &cs {
            if x.index == r.index {
                continue;
            }
            let oh = s.h(x)?;
            let max = h.max(oh);
            if max > 1.35 * h.min(oh)
                || (r.b.n[0] - x.b.n[0]).abs() > 0.5 * max
                || !(0.7 * max..=8. * max).contains(&(cy(&r.b) - cy(&x.b)).abs())
            {
                continue;
            }
            out.insert(s.id(r)?);
            if let Some(b) = body {
                out.insert(s.id(b)?);
            }
            break;
        }
        if out.contains(&s.id(r)?) {
            continue;
        }
        let mut peers = Vec::new();
        for (x, _) in &cs {
            if x.index != r.index && eq(&s.l(x)["font_signature"], &s.l(r)["font_signature"]) {
                if h == 0. {
                    return Err(invalid());
                }
                if (0.85..=1.15).contains(&(s.h(x)? / h)) && x.b.n[0] >= r.b.n[2] + h {
                    peers.push(*x)
                }
            }
        }
        let repeated = peers.iter().any(|a| {
            peers.iter().any(|b| {
                a.index != b.index
                    && (a.b.n[0] - b.b.n[0]).abs() <= 0.5 * h
                    && (0.7 * h..=8. * h).contains(&(cy(&a.b) - cy(&b.b)).abs())
            })
        });
        let mut colon = false;
        for x in rs {
            if ordinary(s, x)
                && s.l(x)["paragraph_group"].is_null()
                && s.t(x)?
                    .trim_end_matches(python_space)
                    .ends_with([':', '：'])
                && count(x)? >= 2
                && (0.0..=3. * h).contains(&(r.b.n[1] - x.b.n[3]))
                && (0.0..=2. * h).contains(&(r.b.n[0] - x.b.n[0]))
            {
                colon = true;
                break;
            }
        }
        if repeated && colon {
            out.insert(s.id(r)?);
            if let Some(b) = body {
                out.insert(s.id(b)?);
            }
        }
    }
    Ok(out)
}
pub(super) fn italic(s: &State, l: &Lane) -> Result<BTreeSet<i64>, Error> {
    let rs = ordered(s, l, false, false);
    let mut out = BTreeSet::new();
    for p in rs.windows(4) {
        if p.iter().any(|r| {
            !clean(s, r)
                || s.l(r)["font_signature"].is_null()
                || s.l(r)["formula_candidate_only"] == true
        }) {
            continue;
        }
        if flags(s.l(&p[0]))? & 64 == 0
            || flags(s.l(&p[1]))? & 64 == 0
            || flags(s.l(&p[2]))? & 64 != 0
            || flags(s.l(&p[3]))? & 64 != 0
        {
            continue;
        }
        let hs = p.iter().map(|r| s.h(r)).collect::<Result<Vec<_>, _>>()?;
        let h = median(hs.clone())?;
        if h <= 0. || hs.iter().any(|n| !(0.85 * h..=1.15 * h).contains(n)) {
            continue;
        }
        if !fonts(s.l(&p[0]), s.l(&p[1]))?
            || !fonts(s.l(&p[2]), s.l(&p[3]))?
            || !(s.l(&p[1])["paragraph_terminal"] == true
                || regex(r#"[.!?。！？][\])’"']*$"#)
                    .is_match(s.t(&p[1])?.trim_end_matches(python_space)))
            || (p[0].b.n[0] - p[1].b.n[0]).abs() > 0.25 * h
            || !(0.65 * h..=3. * h).contains(&(p[1].b.n[0] - p[2].b.n[0]))
            || width(&p[2]) < 0.75 * (l.right - l.left)
            || bounded_words(s.t(&p[2])?) < 6
        {
            continue;
        }
        let mut ok = true;
        for a in p.windows(2) {
            ok &= (-0.1 * h..=0.8 * h).contains(&s.gap(&a[0], &a[1])?);
        }
        if ok {
            out.insert(s.id(&p[2])?);
        }
    }
    Ok(out)
}
pub(super) fn caption(s: &State, l: &Lane) -> Result<BTreeSet<i64>, Error> {
    let rs = ordered(s, l, false, false);
    let mut out = BTreeSet::new();
    let (mut active, mut left, mut center) = (false, 0., 0.);
    for (i, p) in rs.windows(2).enumerate() {
        let (a, b) = (&p[0], &p[1]);
        let (al, bl) = (s.l(a), s.l(b));
        let (ah, bh) = (s.h(a)?, s.h(b)?);
        let cont = width(a) >= 0.75 * (l.right - l.left)
            && al["caption_start"] != true
            && !s
                .t(a)?
                .trim_end_matches(python_space)
                .ends_with(['.', ':', '。', '：', '!', '?', '！', '？'])
            && (0.85..=1.15).contains(&(ah / bh.max(0.1)));
        if bl["caption_start"] == true && !cont && s.gap(a, b)? >= 0.5 * ah.max(bh) {
            out.insert(s.id(b)?);
        }
        if al["caption_start"] == true {
            active = true;
            left = a.b.n[0];
            center = cx(&a.b)
        }
        if !ordinary(s, a) || !ordinary(s, b) || bl["caption_start"] == true {
            active = false;
            continue;
        }
        if active {
            if ah == 0. {
                return Err(invalid());
            }
            if (cx(&b.b) - center).abs() <= 0.5 * ah
                && b.b.n[1] - a.b.n[3] <= 0.75 * ah
                && (0.85..=1.15).contains(&(bh / ah))
            {
                left = left.min(b.b.n[0]);
            }
        }
        let w = l.right - l.left;
        let mut credit = active
            && s.t(a)?
                .trim_end_matches(python_space)
                .ends_with([')', '）'])
            && width(a) <= 0.6 * w
            && (cx(&a.b) - (l.left + l.right) * 0.5).abs() <= 0.08 * w
            && (0.4 * ah..=2. * ah).contains(&(b.b.n[0] - l.left))
            && width(b) >= 0.85 * w
            && (b.b.n[2] - l.right).abs() <= 0.3 * ah;
        if credit {
            if ah == 0. {
                return Err(invalid());
            }
            credit = (0.85..=1.15).contains(&(bh / ah)) && i + 3 < rs.len();
            if credit {
                for r in &rs[i + 2..i + 4] {
                    credit &= (r.b.n[0] - l.left).abs() <= 0.3 * ah
                        && (r.b.n[2] - l.right).abs() <= 0.3 * ah
                        && (0.85..=1.15).contains(&(s.h(r)? / ah));
                }
            }
        }
        if credit {
            out.insert(s.id(b)?);
            active = false;
            continue;
        }
        if b.b.n[0] < left - ah || b.b.n[1] - a.b.n[3] > 3. * ah {
            active = false;
            continue;
        }
        if active && bh >= 1.15 * ah && b.b.n[1] - a.b.n[3] >= 0.5 * ah {
            out.insert(s.id(b)?);
            active = false
        }
    }
    Ok(out)
}
pub(super) fn structured(
    s: &State,
    l: &Lane,
    regular: f64,
    mad: f64,
) -> Result<BTreeSet<i64>, Error> {
    let rs = ordered(s, l, false, true);
    let w = (l.right - l.left).max(0.1);
    let mut regions: Vec<Vec<Row>> = Vec::new();
    for r in rs {
        if !ordinary(s, &r) {
            if regions.last().is_some_and(|r| !r.is_empty()) {
                regions.push(vec![])
            }
            continue;
        }
        if regions.is_empty() {
            regions.push(vec![])
        }
        regions.last_mut().unwrap().push(r)
    }
    let mut out = BTreeSet::new();
    for rs in regions {
        let mut cs = vec![];
        for (i, r) in rs.iter().enumerate() {
            let e = &s.l(r)["leading_emphasis_width"];
            if !e.is_null()
                && num(e)? <= 0.2 * w
                && width(r) >= 0.95 * w
                && (r.b.n[0] - l.left).abs() <= 0.75 * s.h(r)?
            {
                cs.push(i)
            }
        }
        if cs.len() < 3 {
            continue;
        }
        for i in cs {
            if i == 0 {
                continue;
            }
            let (a, b) = (&rs[i - 1], &rs[i]);
            let h = s.h(a)?.max(s.h(b)?);
            if (a.b.n[2] - l.left) / w <= 0.8
                && (-0.25 * h..=regular + (0.75 * h).max(3. * mad)).contains(&s.gap(a, b)?)
            {
                out.insert(s.id(b)?);
            }
        }
    }
    Ok(out)
}
pub(super) fn isolated(
    s: &State,
    l: &Lane,
    regular: f64,
    mad: f64,
) -> Result<BTreeSet<i64>, Error> {
    let rs = ordered(s, l, true, true);
    let w = (l.right - l.left).max(0.1);
    let mut out = BTreeSet::new();
    for p in rs.windows(3) {
        let (a, b, c) = (&p[0], &p[1], &p[2]);
        let h = s.h(a)?.max(s.h(b)?).max(s.h(c)?);
        if width(a) > 0.3 * w
            || !regex(r#"[.!?。！？:：;；][\]\)}）】》”’'"]*$"#)
                .is_match(s.t(a)?.trim_end_matches(python_space))
            || !(5f64.max(0.65 * h)..=3. * h).contains(&(b.b.n[0] - l.left))
            || width(b) < 0.75 * w
            || (c.b.n[0] - l.left).abs() > 0.75 * h
            || width(c) < 0.65 * w
            || !fonts(s.l(b), s.l(c))?
        {
            continue;
        }
        let max = regular + (0.75 * h).max(3. * mad);
        if (-0.25 * h..=max).contains(&s.body_gap(a, b)?)
            && (-0.25 * h..=max).contains(&s.body_gap(b, c)?)
        {
            out.insert(s.id(b)?);
        }
    }
    Ok(out)
}
pub(super) fn visual(
    s: &State,
    l: &Lane,
    boxes: &[Box4],
    page_h: f64,
) -> Result<BTreeSet<i64>, Error> {
    let rs = ordered(s, l, true, true);
    let mut out = BTreeSet::new();
    if boxes.is_empty() {
        return Ok(out);
    }
    for p in rs.windows(2) {
        let (a, b) = (&p[0], &p[1]);
        let h = s.h(a)?.max(s.h(b)?);
        if width(a) > 0.7 * width(b)
            || b.b.n[0] > a.b.n[0] - 0.25 * h
            || b.b.n[2] < a.b.n[2] + 0.25 * h
            || (cx(&a.b) - cx(&b.b)).abs() > 0.1 * width(b)
            || !(-0.25 * h..=0.75 * h).contains(&s.gap(a, b)?)
        {
            continue;
        }
        if boxes.iter().any(|v| {
            (-0.25 * h..=(2. * h).max(0.03 * page_h)).contains(&(a.b.n[1] - v.n[3]))
                && axis(&b.b, v, 0) >= 0.8
                && (cx(&b.b) - cx(v)).abs() <= 0.12 * width(b).max(v.n[2] - v.n[0])
        }) {
            out.insert(s.id(b)?);
        }
    }
    Ok(out)
}
pub(super) fn formula(s: &State, l: &Lane) -> Result<BTreeSet<i64>, Error> {
    let rs = ordered(s, l, true, true);
    let w = (l.right - l.left).max(0.1);
    let mut edges = Vec::new();
    for (i, p) in rs.windows(2).enumerate() {
        let (a, b) = (&p[0], &p[1]);
        if s.l(a)["paragraph_formula_context"] != true
            || s.l(b)["paragraph_formula_context"] != true
        {
            continue;
        }
        let (ah, bh) = (s.h(a)?, s.h(b)?);
        if ah.min(bh) < 0.75 * ah.max(bh)
            || width(a).min(width(b)) < 0.45 * w
            || width(a).max(width(b)) > 0.95 * w
        {
            continue;
        }
        let c = 0.5 * (l.left + l.right);
        if (cx(&a.b) - c).abs() > 0.15 * w || (cx(&b.b) - c).abs() > 0.15 * w {
            continue;
        }
        let h = (ah + bh) * 0.5;
        if (a.b.n[3].min(b.b.n[3]) - a.b.n[1].max(b.b.n[1])).max(0.) <= 0.2 * ah.min(bh)
            && (0.9 * h..=2. * h).contains(&(b.b.n[1] - a.b.n[1]))
        {
            edges.push(i)
        }
    }
    let mut out = BTreeSet::new();
    for i in edges {
        for r in &rs[i..(i + 3).min(rs.len())] {
            out.insert(s.id(r)?);
        }
    }
    Ok(out)
}
pub(super) fn front(
    s: &State,
    l: &Lane,
    page_h: f64,
    page_index: Option<i64>,
) -> Result<BTreeSet<i64>, Error> {
    let mut out = BTreeSet::new();
    if page_index != Some(0) {
        return Ok(out);
    }
    for r in &l.rows {
        if ordinary(s, r)
            && r.b.n[1] <= 0.65 * page_h
            && regex(
                r"(?i)^\s*(?:keywords?|key\s+words?|关键词|中图分类号|文献标识码|文章编号)\s*[:：]",
            )
            .is_match(&fold(s.t(r)?))
        {
            out.insert(s.id(r)?);
        }
    }
    Ok(out)
}
