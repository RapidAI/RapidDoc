//! Exact decision thresholds from rapid_doc/utils/pdf_classify.py.
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

pub fn space(c: char) -> bool {
    matches!(c as u32,9..=13|28..=32|0x85|0xa0|0x1680|0x2000..=0x200a|0x2028|0x2029|0x202f|0x205f|0x3000)
}
pub fn cjk(c: u32) -> bool {
    matches!(c,0x3400..=0x4dbf|0x4e00..=0x9fff|0xf900..=0xfaff|0x20000..=0x2ebef)
}
fn control(c: u32) -> bool {
    matches!(c,0..=31|127..=159) && !matches!(c, 9 | 10 | 13)
}
pub fn normalize(s: &str) -> String {
    let s = s.trim_matches(space).trim_start_matches('/');
    if s.as_bytes().get(6) == Some(&b'+') && s.as_bytes()[..6].iter().all(u8::is_ascii_uppercase) {
        s[7..].into()
    } else {
        s.into()
    }
}
pub fn indices(count: usize, max: usize) -> Vec<usize> {
    let n = count.min(max);
    if n == 0 {
        return vec![];
    }
    if n == count {
        return (0..count).collect();
    }
    if n == 1 {
        return vec![0];
    }
    let mut ids = BTreeSet::new();
    for i in 0..n {
        ids.insert(
            ((i as f64 * (count - 1) as f64 / (n - 1) as f64).round_ties_even() as usize)
                .min(count - 1),
        );
    }
    for i in 0..count {
        if ids.len() == n {
            break;
        }
        ids.insert(i);
    }
    ids.into_iter().collect()
}
fn n(v: &Value, key: &str) -> usize {
    v[key].as_u64().unwrap_or(0) as usize
}
fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.
    } else {
        a as f64 / b as f64
    }
}
fn increment(map: &mut Map<String, Value>, key: &str) {
    let count = map.get(key).and_then(Value::as_u64).unwrap_or(0);
    map.insert(key.into(), json!(count + 1));
}
pub fn sample(facts: &Value, index: usize) -> Value {
    let chars = facts["chars"].as_array().unwrap();
    let text = facts["text"].as_str().unwrap_or("");
    let (mut null_count, mut replacement, mut controls, mut private, mut errors, mut nongenerated) =
        (0, 0, 0, 0, 0, 0);
    let (mut fonts, mut used, mut cjks, mut badmaps, mut abnormal) =
        (Map::new(), Map::new(), Map::new(), Map::new(), Map::new());
    for c in chars {
        let code = c[0].as_u64().unwrap_or(0) as u32;
        let generated = c[1].as_bool().unwrap_or(false);
        let error = c[2].as_bool().unwrap_or(false);
        let bad = code == 0 || code == 0xfffd || control(code) || matches!(code, 0xe000..=0xf8ff);
        if !generated {
            nongenerated += 1
        }
        if code == 0 {
            null_count += 1
        } else if code == 0xfffd {
            replacement += 1
        } else if control(code) {
            controls += 1
        } else if matches!(code, 0xe000..=0xf8ff) {
            private += 1
        }
        if error {
            errors += 1
        }
        let name = normalize(c[3].as_str().unwrap_or(""));
        if name.is_empty() {
            continue;
        }
        increment(&mut fonts, &name);
        if !generated {
            increment(&mut used, &name);
            if error {
                increment(&mut badmaps, &name)
            }
            if bad {
                increment(&mut abnormal, &name)
            }
            if cjk(code) {
                increment(&mut cjks, &name)
            }
        }
    }
    json!({"page_index":index,"text":text,"cleaned_text":text.chars().filter(|&c|!space(c)).collect::<String>(),"char_count":chars.len(),"null_char_count":null_count,"replacement_char_count":replacement,"control_char_count":controls,"private_use_char_count":private,"unicode_map_error_count":errors,"font_name_counts":fonts,"non_generated_char_count":nongenerated,"font_non_generated_char_counts":used,"font_non_generated_cjk_char_counts":cjks,"font_non_generated_unicode_map_error_counts":badmaps,"font_non_generated_abnormal_char_counts":abnormal})
}
pub fn quality(samples: &[Value]) -> Value {
    let total = samples.iter().map(|s| n(s, "char_count")).sum::<usize>();
    let mut v = json!({"total_chars":total});
    let mut bad = 0;
    for k in [
        "null_char_count",
        "replacement_char_count",
        "control_char_count",
        "private_use_char_count",
    ] {
        let count = samples.iter().map(|s| n(s, k)).sum::<usize>();
        bad += count;
        v[k] = json!(count)
    }
    v["abnormal_ratio"] = json!(ratio(bad, total));
    v
}
pub fn map_errors(samples: &[Value]) -> Value {
    let total = samples.iter().map(|s| n(s, "char_count")).sum::<usize>();
    let errors = samples
        .iter()
        .map(|s| n(s, "unicode_map_error_count"))
        .sum::<usize>();
    json!({"total_chars":total,"unicode_map_error_count":errors,"unicode_map_error_ratio":ratio(errors,total)})
}
pub fn widespread(samples: &[Value], bad: impl Fn(&Value) -> bool) -> bool {
    !samples.is_empty()
        && samples.iter().filter(|s| bad(s)).count()
            >= if samples.len() == 1 {
                1
            } else {
                2.max(samples.len().div_ceil(2))
            }
}
pub fn cid_usage(samples: &[Value], fonts: &Value) -> Value {
    let mut best = json!({"triggered":false,"page_index":null,"font_names":[],"cid_font_char_count":0,"total_chars":0,"cid_font_usage_ratio":0.,"unicode_map_error_ratio":0.,"abnormal_ratio":0.});
    if fonts["triggered"] != true {
        return best;
    }
    for s in samples {
        let total = n(s, "non_generated_char_count");
        if total == 0 {
            continue;
        }
        let key = n(s, "page_index").to_string();
        let names = fonts["page_fonts"][&key]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(normalize)
                    .filter(|s| !s.is_empty())
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        for name in names {
            let count = n(&s["font_non_generated_char_counts"], &name);
            if count == 0 {
                continue;
            }
            let usage = ratio(count, total);
            let errors = ratio(
                n(&s["font_non_generated_unicode_map_error_counts"], &name),
                count,
            );
            let bad = ratio(
                n(&s["font_non_generated_abnormal_char_counts"], &name),
                count,
            );
            let triggered = count >= 30 && usage >= 0.01 && (errors >= 0.04 || bad >= 0.03);
            let signal = json!({"triggered":triggered,"page_index":s["page_index"],"font_names":[name],"cid_font_char_count":count,"total_chars":total,"cid_font_usage_ratio":usage,"unicode_map_error_ratio":errors,"abnormal_ratio":bad});
            if triggered {
                return signal;
            }
            if (usage, count)
                > (
                    best["cid_font_usage_ratio"].as_f64().unwrap(),
                    n(&best, "cid_font_char_count"),
                )
            {
                best = signal
            }
        }
    }
    best
}
pub fn latin_usage(
    samples: &[Value],
    fonts: &Value,
    count_threshold: usize,
    usage_threshold: f64,
    cjk_threshold: f64,
) -> Value {
    let mut best = json!({"triggered":false,"page_index":null,"font_names":[],"font_char_count":0,"cjk_char_count":0,"total_chars":0,"font_usage_ratio":0.,"font_cjk_ratio":0.});
    if fonts["triggered"] != true {
        return best;
    }
    for s in samples {
        let total = n(s, "non_generated_char_count");
        if total == 0 {
            continue;
        }
        let key = n(s, "page_index").to_string();
        let names = fonts["page_fonts"][&key]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(normalize)
                    .filter(|s| !s.is_empty())
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        for name in names {
            let count = n(&s["font_non_generated_char_counts"], &name);
            let cjks = n(&s["font_non_generated_cjk_char_counts"], &name);
            let usage = ratio(count, total);
            let proportion = ratio(cjks, count);
            let triggered =
                count >= count_threshold && usage >= usage_threshold && proportion >= cjk_threshold;
            let signal = json!({"triggered":triggered,"page_index":s["page_index"],"font_names":if count>0{vec![name]}else{vec![]},"font_char_count":count,"cjk_char_count":cjks,"total_chars":total,"font_usage_ratio":usage,"font_cjk_ratio":proportion});
            if triggered {
                return signal;
            }
            if (proportion, usage, count)
                > (
                    best["font_cjk_ratio"].as_f64().unwrap(),
                    best["font_usage_ratio"].as_f64().unwrap(),
                    n(&best, "font_char_count"),
                )
            {
                best = signal
            }
        }
    }
    best
}
const SCRIPTS: &[(u32, u32, &str)] = &[
    (0x0400, 0x052f, "Cyrillic"),
    (0x0600, 0x06ff, "Arabic"),
    (0x0700, 0x074f, "Syriac"),
    (0x0750, 0x077f, "Arabic Supplement"),
    (0x0780, 0x07bf, "Thaana"),
    (0x07c0, 0x07ff, "NKo"),
    (0x0800, 0x083f, "Samaritan"),
    (0x0840, 0x085f, "Mandaic"),
    (0x0860, 0x086f, "Syriac Supplement"),
    (0x0870, 0x089f, "Arabic Extended-B"),
    (0x0900, 0x097f, "Devanagari"),
    (0x0c80, 0x0cff, "Kannada"),
    (0x1000, 0x109f, "Myanmar"),
    (0x1100, 0x11ff, "Hangul Jamo"),
    (0x1200, 0x137f, "Ethiopic"),
    (0x13a0, 0x13ff, "Cherokee"),
    (0x1400, 0x167f, "Canadian Syllabics"),
    (0x1800, 0x18af, "Mongolian"),
    (0x1a20, 0x1aaf, "Tai Tham"),
    (0x2c00, 0x2c5f, "Glagolitic"),
    (0xa000, 0xa48f, "Yi"),
];
pub fn cross_script(samples: &[Value]) -> Value {
    let (mut total, mut cjks, mut suspicious) = (0, 0, 0);
    let mut counts = BTreeMap::<String, usize>::new();
    for s in samples {
        for c in s["cleaned_text"].as_str().unwrap_or("").chars() {
            total += 1;
            if cjk(c as u32) {
                cjks += 1
            }
            if let Some((_, _, name)) = SCRIPTS
                .iter()
                .find(|&&(a, b, _)| (a..=b).contains(&(c as u32)))
            {
                suspicious += 1;
                *counts.entry((*name).into()).or_default() += 1;
            }
        }
    }
    let proportion = ratio(suspicious, total);
    let dense = counts.values().filter(|&&n| n >= 5).count();
    let mut top = counts
        .iter()
        .map(|(k, &v)| (k.clone(), v))
        .collect::<Vec<_>>();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    top.truncate(5);
    json!({"triggered":total>=300&&cjks>=100&&suspicious>=120&&proportion>=0.18&&dense>=3,"total_chars":total,"cjk_chars":cjks,"suspicious_chars":suspicious,"suspicious_ratio":proportion,"script_counts":counts,"top_scripts":top,"dense_script_count":dense})
}
pub fn u72(samples: &[Value]) -> Value {
    let (mut cjks, mut count) = (0, 0);
    for s in samples {
        for c in s["cleaned_text"].as_str().unwrap_or("").chars() {
            if ('\u{4e00}'..='\u{9fff}').contains(&c) {
                cjks += 1
            }
            if ('\u{7280}'..='\u{72df}').contains(&c)
                && !"犀犁犄犊犒犟犬犯状犷犹狂狄狈狐狗狙狞".contains(c)
            {
                count += 1
            }
        }
    }
    json!({"cjk_chars":cjks,"u72xx_count":count,"u72xx_cjk_ratio":ratio(count,cjks)})
}
fn runs(text: &str, predicate: impl Fn(char) -> bool, min: usize) -> usize {
    let (mut total, mut run) = (0, 0);
    for c in text.chars() {
        if predicate(c) {
            run += 1
        } else {
            if run >= min {
                total += run
            }
            run = 0
        }
    }
    if run >= min {
        total += run
    }
    total
}
pub fn punctuation(samples: &[Value]) -> Value {
    let mut best = json!({"triggered":false,"page_index":null,"cleaned_text_chars":0,"ascii_punct_count":0,"ascii_punct_ratio":0.,"ascii_punct_run_chars":0,"punct_run_ratio":0.,"dot_leader_chars":0,"effective_ascii_punct_ratio":0.,"effective_punct_run_ratio":0.});
    for s in samples {
        let text = s["cleaned_text"].as_str().unwrap_or("");
        let total = text.chars().count();
        let count = text.chars().filter(char::is_ascii_punctuation).count();
        let run = runs(text, |c| c.is_ascii_punctuation(), 4);
        let dots = runs(text, |c| c == '.', 8);
        let exempt = if total - count >= 80 { dots } else { 0 };
        let effective = ratio(count.saturating_sub(exempt), total);
        let effective_runs = ratio(run.saturating_sub(exempt), total);
        let triggered = total >= 100 && effective >= 0.25 && effective_runs >= 0.10;
        let signal = json!({"triggered":triggered,"page_index":s["page_index"],"cleaned_text_chars":total,"ascii_punct_count":count,"ascii_punct_ratio":ratio(count,total),"ascii_punct_run_chars":run,"punct_run_ratio":ratio(run,total),"dot_leader_chars":dots,"effective_ascii_punct_ratio":effective,"effective_punct_run_ratio":effective_runs});
        if triggered {
            return signal;
        }
        if (effective_runs, effective, total)
            > (
                best["effective_punct_run_ratio"].as_f64().unwrap(),
                best["effective_ascii_punct_ratio"].as_f64().unwrap(),
                n(&best, "cleaned_text_chars"),
            )
        {
            best = signal
        }
    }
    best
}
pub fn decoded_text_bad(samples: &[Value], fonts: &Value) -> bool {
    let errors = map_errors(samples);
    if errors["unicode_map_error_ratio"].as_f64().unwrap() >= 0.04
        && widespread(samples, |s| {
            ratio(n(s, "unicode_map_error_count"), n(s, "char_count")) >= 0.04
                && n(s, "char_count") > 0
        })
    {
        return true;
    }
    if cid_usage(samples, &fonts["cid_without_to_unicode"])["triggered"] == true
        || latin_usage(
            samples,
            &fonts["latin_charset_with_to_unicode"],
            30,
            0.01,
            0.8,
        )["triggered"]
            == true
    {
        return true;
    }
    let q = quality(samples);
    if n(&q, "total_chars") >= 300
        && q["abnormal_ratio"].as_f64().unwrap() >= 0.03
        && widespread(samples, |s| {
            n(s, "char_count") > 0
                && quality(&[s.clone()])["abnormal_ratio"].as_f64().unwrap() >= 0.03
        })
    {
        return true;
    }
    let suspicious = u72(samples);
    cross_script(samples)["triggered"] == true
        || (n(&suspicious, "u72xx_count") >= 30
            && suspicious["u72xx_cjk_ratio"].as_f64().unwrap() >= 0.026)
        || (punctuation(samples)["triggered"] == true
            && widespread(samples, |s| punctuation(&[s.clone()])["triggered"] == true))
}
