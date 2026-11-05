//! Full existing line_layout semantic/body connection rules, including all exceptions.
use super::*;
pub(super) fn semantic(
    s: &State,
    a: &Row,
    b: &Row,
    lane: &Lane,
    regular: f64,
) -> Result<bool, Error> {
    let (al, bl) = (s.l(a), s.l(b));
    if !eq(&al["semantic_type"], &bl["semantic_type"]) {
        return Ok(false);
    }
    if bl["semantic_type"] == "paragraph_title"
        && bl["explicit_section_title"] == true
        && !bl["title_band_id"].is_null()
        && !eq(&bl["title_band_id"], &al["title_band_id"])
    {
        return Ok(false);
    }
    if s.table_barrier(a, b)? || s.rule_barrier(a, b, lane)? {
        return Ok(false);
    }
    let ah = s.h(a)?;
    let bh = s.h(b)?;
    let h = ah.max(bh);
    let shared = !al["title_band_id"].is_null() && eq(&al["title_band_id"], &bl["title_band_id"]);
    let confirmed = shared && al["structural_title"] == true && bl["structural_title"] == true;
    if h / ah.min(bh) > 1.35 && !confirmed {
        return Ok(false);
    }
    let gap = s.gap(a, b)?;
    if gap < -0.25 * h || gap > (1.25 * h).max(regular + 0.75 * h) {
        return Ok(false);
    }
    if al["semantic_type"] == "paragraph_title" && gap > 0.5 * h && !shared {
        return Ok(false);
    }
    let font_conflict = !al["font_signature"].is_null()
        && !bl["font_signature"].is_null()
        && s.coverage(a)? >= 0.75
        && s.coverage(b)? >= 0.75
        && !eq(&al["font_signature"], &bl["font_signature"]);
    let uncertain = al["semantic_type"] == "doc_title"
        && s.coverage(a)?.min(s.coverage(b)?) < 0.85
        && (al["dominant_font_weight"].is_null()
            || bl["dominant_font_weight"].is_null()
            || (num(&al["dominant_font_weight"])? - num(&bl["dominant_font_weight"])?).abs()
                < 100.);
    if font_conflict && !uncertain && !confirmed {
        return Ok(false);
    }
    let width = (lane.right - lane.left).max(0.1);
    Ok((cx(&a.b) - cx(&b.b)).abs() <= 0.15 * width
        || (a.b.n[0] - b.b.n[0]).abs() <= 0.75 * h
        || axis(&a.b, &b.b, 0) >= 0.35)
}
pub(super) fn text(
    s: &State,
    a: &Row,
    b: &Row,
    lane: &Lane,
    regular: f64,
    mad: f64,
) -> Result<bool, Error> {
    let (al, bl) = (s.l(a), s.l(b));
    let ah = s.h(a)?;
    let bh = s.h(b)?;
    let h = ah.max(bh);
    let min = ah.min(bh);
    let width = (lane.right - lane.left).max(0.1);
    let aw = a.b.n[2] - a.b.n[0];
    let bw = b.b.n[2] - b.b.n[0];
    let gap = s.body_gap(a, b)?;
    if eq(&al["visual_row_id"], &bl["visual_row_id"])
        && (al["split_from_row"] == true || bl["split_from_row"] == true)
    {
        return Ok(false);
    }
    if bh < 0.88 * ah && gap > regular + (0.25 * ah).max(3. * mad) {
        return Ok(false);
    }
    let fill = aw >= 0.8 * width && bw >= 0.8 * width;
    let aligned = (a.b.n[0] - b.b.n[0]).abs() <= 0.5 * h;
    let returns = (b.b.n[0] - lane.left).abs() <= 0.75 * h
        && (-0.5 * h..=2. * h).contains(&(a.b.n[0] - lane.left));
    let ac = &al["native_cjk_style"];
    let bc = &bl["native_cjk_style"];
    let mixed = if !ac.is_null() && !bc.is_null() {
        let ac = array(ac, 4)?;
        let bc = array(bc, 4)?;
        if ac.len() != 4 || bc.len() != 4 {
            return Err(invalid());
        }
        eq(&json!(&ac[..3]), &json!(&bc[..3]))
            && num(&ac[3])? > 0.
            && (0.9..=1.1).contains(&(num(&bc[3])? / num(&ac[3])?))
            && (1.25..=1.45).contains(&(h / min))
            && aw >= 0.8 * width
            && bw >= 0.5 * width
            && returns
            && al["paragraph_terminal"] != true
            && !conflict(al, bl)?
            && (-0.25 * h..=regular + 0.35 * h).contains(&gap)
    } else {
        false
    };
    let af = &al["font_signature"];
    let bf = &bl["font_signature"];
    let reliable = af.is_null()
        || bf.is_null()
        || s.coverage(a)? < 0.75
        || s.coverage(b)? < 0.75
        || eq(af, bf)
        || share(af, bf)?
        || bw <= 0.5 * width && eq(&af[1], &bf[1]);
    let indent = a.b.n[0] - lane.left;
    let mut metric = true;
    for r in [a, b] {
        let l = s.l(r);
        if l["style_scale_repaired"] != true || l["source_bbox"].is_null() {
            metric = false;
            break;
        }
        let raw = bbox(&l["source_bbox"])?;
        if raw.n[3] - raw.n[1] < 4. * s.h(r)? {
            metric = false;
            break;
        }
    }
    let repeated = indent >= 5f64.max((if metric { 1.25 } else { 1.8 }) * h)
        && (b.b.n[0] - a.b.n[0]).abs() <= 0.5 * h
        && reliable
        && (-0.25 * h..=regular + (0.75 * h).max(3. * mad)).contains(&gap);
    let at = s.t(a)?.trim_end_matches(python_space);
    let bt = s.t(b)?.trim_end_matches(python_space);
    let terminal_b =
        bl["paragraph_terminal"] == true || bt.ends_with(['.', '!', '?', '。', '！', '？']);
    let enumeration = al["semantic_type"].is_null()
        && bl["semantic_type"].is_null()
        && at.ends_with([',', '，'])
        && terminal_b
        && words(s.t(b)?) <= 10
        && !af.is_null()
        && !bf.is_null()
        && emphasis(af, bf)?
        && h <= 1.2 * min
        && (-0.1 * h..=0.4 * h).contains(&gap);
    let count = regex(r"[A-Za-z]{2,}").find_iter(s.t(b)?).count();
    let italic = al["semantic_type"].is_null()
        && bl["semantic_type"].is_null()
        && al["paragraph_terminal"] != true
        && !at.ends_with(['.', '!', '?', '。', '！', '？', ':', '：', ';', '；'])
        && al["formula_candidate_only"] != true
        && bl["formula_candidate_only"] != true
        && bl["compact_formula_cluster"] != true
        && terminal_b
        && regex(r"^(?:[+&]\s+|[a-z])").is_match(s.t(b)?.trim_matches(python_space))
        && (1..=10).contains(&count)
        && !af.is_null()
        && !bf.is_null()
        && flags(bl)? & 64 != 0
        && flags(al)? & 64 == 0
        && emphasis(af, bf)?
        && !conflict(al, bl)?
        && h <= 1.2 * min
        && (-0.1 * h..=0.4 * h).contains(&gap);
    let safe = aw >= 0.75 * width
        && bw <= 0.7 * width
        && (aligned || returns)
        && (reliable || enumeration || italic)
        && (!conflict(al, bl)? || enumeration)
        && (-0.25 * h..=regular + (0.75 * h).max(3. * mad)).contains(&gap);
    let changed = !af.is_null() && !bf.is_null() && !eq(&af[1], &bf[1]) && !share(af, bf)?;
    let reliable_conflict = !af.is_null()
        && !bf.is_null()
        && s.coverage(a)? >= 0.75
        && s.coverage(b)? >= 0.75
        && (!eq(af, bf) && !share(af, bf)? || conflict(al, bl)?);
    let fallback = !af.is_null()
        && !bf.is_null()
        && s.coverage(a)? >= 0.75
        && s.coverage(b)? >= 0.75
        && !eq(&af[0], &bf[0])
        && eq(&af[1], &bf[1])
        && aligned
        && h / min <= 1.25
        && !conflict(al, bl)?
        && (-0.25 * h..=regular + (0.35 * h).max(3. * mad)).contains(&gap);
    if changed && conflict(al, bl)? && !fallback && !mixed && !enumeration {
        return Ok(false);
    }
    let structural =
        gap > regular + (0.75 * h).max(3. * mad) && (h / min.max(0.1) >= 1.12 || reliable_conflict);
    if !hyphen(s.t(a)?) && structural && !fallback && !mixed {
        return Ok(false);
    }
    let full = fill && aligned && !changed && gap <= regular + (0.75 * min).max(3. * mad);
    if h / min > 1.35 && !safe && !full && (!fill || !aligned || changed) && !mixed {
        return Ok(false);
    }
    if gap < -0.25 * h {
        return Ok(false);
    }
    if axis(&a.b, &b.b, 0) < 0.5 && (a.b.n[0] - b.b.n[0]).abs() > 1.5 * h && !safe {
        return Ok(false);
    }
    if s.table_barrier(a, b)? || s.rule_barrier(a, b, lane)? {
        return Ok(false);
    }
    if mixed {
        return Ok(true);
    }
    let limit = (regular + (0.5 * h).max(3. * mad)).max(1.1 * h);
    if hyphen(s.t(a)?) {
        return Ok(gap <= limit.max(1.8 * h));
    }
    if gap > limit {
        return Ok(false);
    }
    let terminal = regex(r#"[.!?。！？:：;；][\])}）】》”’'"]*$"#).is_match(at);
    let sparse = lane
        .rows
        .iter()
        .filter(|r| s.l(r)["semantic_type"].is_null())
        .count()
        <= 6;
    if terminal && !repeated && (sparse && gap > 0.65 * h || gap > regular + 0.5 * h) {
        return Ok(false);
    }
    let left = lane.left.min(a.b.n[0]);
    let lw = (lane.right - left).max(0.1);
    let indent = b.b.n[0] - left;
    let previous_fill = (a.b.n[2] - left).max(0.) / lw;
    if indent >= 5f64.max(0.65 * h) && (previous_fill <= 0.8 || terminal) && !safe && !repeated {
        return Ok(false);
    }
    let abnormal = gap > regular + (0.25 * h).max(3. * mad);
    if reliable_conflict && (abnormal || aw.min(bw) <= 0.7 * width) && !fill && !safe && !fallback {
        return Ok(false);
    }
    if abnormal && aw.min(bw) <= 0.65 * width && !safe {
        return Ok(false);
    }
    Ok(true)
}
