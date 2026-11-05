//! Original BIFF number-format sections and display rules, without host codecs.
use chrono::{Datelike, Timelike};
use num_bigint::BigInt;
use std::sync::OnceLock;
fn re(pattern: &str) -> regex::Regex {
    regex::Regex::new(pattern).unwrap()
}
pub fn builtin(id: u16) -> Option<&'static str> {
    Some(match id {
        1 => "0",
        2 => "0.00",
        3 => "#,##0",
        4 => "#,##0.00",
        9 => "0%",
        10 => "0.00%",
        11 => "0.00E+00",
        12 => "# ?/?",
        13 => "# ??/??",
        14 => "mm-dd-yy",
        15 => "d-mmm-yy",
        16 => "d-mmm",
        17 => "mmm-yy",
        18 => "h:mm AM/PM",
        19 => "h:mm:ss AM/PM",
        20 => "h:mm",
        21 => "h:mm:ss",
        22 => "m/d/yy h:mm",
        37 => "#,##0 ;(#,##0)",
        38 => "#,##0 ;[Red](#,##0)",
        39 => "#,##0.00;(#,##0.00)",
        40 => "#,##0.00;[Red](#,##0.00)",
        45 => "mm:ss",
        46 => "[h]:mm:ss",
        47 => "mmss.0",
        48 => "##0.0E+0",
        49 => "@",
        _ => return None,
    })
}
/// CPython's .15g rounding followed by General's small-integer rendering.
fn significant(value: f64) -> String {
    if !value.is_finite() {
        return if value.is_nan() {
            "nan"
        } else if value < 0. {
            "-inf"
        } else {
            "inf"
        }
        .into();
    }
    let raw = format!("{value:.14e}");
    let (mantissa, exp) = raw.split_once('e').unwrap();
    let exp = exp.parse::<i32>().unwrap();
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let digits = mantissa.trim_start_matches('-').replace('.', "");
    if !(-4..15).contains(&exp) {
        let m = mantissa.trim_end_matches('0').trim_end_matches('.');
        return format!(
            "{m}e{}{v:02}",
            if exp < 0 { '-' } else { '+' },
            v = exp.unsigned_abs()
        );
    }
    let at = exp + 1;
    let mut s = if at <= 0 {
        format!("0.{}{}", "0".repeat((-at) as usize), digits)
    } else if at >= digits.len() as i32 {
        format!("{}{}", digits, "0".repeat(at as usize - digits.len()))
    } else {
        format!("{}.{}", &digits[..at as usize], &digits[at as usize..])
    };
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_owned();
    }
    format!("{sign}{s}")
}
pub fn general(value: f64) -> String {
    if value == 0. {
        return "0".into();
    }
    let s = significant(value);
    let rounded = s.parse::<f64>().unwrap_or(value);
    if rounded.is_finite() && rounded.fract() == 0. && rounded.abs() < 1e15 {
        return format!("{rounded:.0}");
    }
    s
}
#[derive(Clone)]
struct Section {
    pattern: String,
    condition: Option<(String, f64)>,
}
fn split(code: &str) -> Option<Vec<String>> {
    let (mut quote, mut bracket, mut escaped) = (false, 0i32, false);
    let mut all = vec![];
    let mut current = String::new();
    for c in code.chars() {
        if escaped {
            current.push(c);
            escaped = false;
            continue;
        }
        if c == '\\' && !quote {
            current.push(c);
            escaped = true;
            continue;
        }
        if c == '"' {
            current.push(c);
            quote = !quote;
            continue;
        }
        if !quote {
            if c == '[' {
                bracket += 1
            } else if c == ']' {
                bracket -= 1;
                if bracket < 0 {
                    return None;
                }
            }
        }
        if c == ';' && !quote && bracket == 0 {
            all.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    if quote || bracket != 0 || escaped {
        return None;
    }
    all.push(current);
    if all.len() > 4 {
        None
    } else {
        Some(all)
    }
}
fn strip(pattern: &str) -> Section {
    static CONDITION: OnceLock<regex::Regex> = OnceLock::new();
    static COLOR: OnceLock<regex::Regex> = OnceLock::new();
    let condition_re =
        CONDITION.get_or_init(|| re(r"^(<=|>=|<>|=|<|>)([-+]?(?:\d+(?:\.\d*)?|\.\d+))$"));
    let color = COLOR.get_or_init(|| re(r"(?i)^Color\d+$"));
    let mut out = String::new();
    let mut condition = None;
    let mut rest = pattern;
    while !rest.is_empty() {
        if !rest.starts_with('[') {
            let c = rest.chars().next().unwrap();
            out.push(c);
            rest = &rest[c.len_utf8()..];
            continue;
        }
        let Some(end) = rest.find(']') else {
            return Section {
                pattern: pattern.into(),
                condition,
            };
        };
        let inner = &rest[1..end];
        if let Some(c) = condition_re.captures(inner).filter(|_| condition.is_none()) {
            let operand = c[2]
                .chars()
                .map(|ch| {
                    crate::docx::decimal(ch)
                        .and_then(|d| char::from_digit(u32::from(d), 10))
                        .unwrap_or(ch)
                })
                .collect::<String>();
            condition = Some((c[1].into(), operand.parse().unwrap_or(f64::NAN)));
        } else if matches!(
            crate::ole::casefold(inner).as_str(),
            "black" | "blue" | "cyan" | "green" | "magenta" | "red" | "white" | "yellow"
        ) || color.is_match(inner)
        {
        } else if let Some(currency) = inner.strip_prefix('$') {
            out.push_str(currency.split('-').next().unwrap_or(""));
        } else {
            out.push_str(&rest[..=end]);
        }
        rest = &rest[end + 1..];
    }
    Section {
        pattern: out,
        condition,
    }
}
fn sections(code: &str) -> Option<Vec<Section>> {
    let s = split(code)?.iter().map(|s| strip(s)).collect::<Vec<_>>();
    if s.iter().filter(|s| s.condition.is_some()).count() > 2 {
        None
    } else {
        Some(s)
    }
}
fn choose(s: &[Section], v: f64) -> Option<(&Section, f64, bool)> {
    let s = if s.len() == 4 { &s[..3] } else { s };
    if s.iter().any(|s| s.condition.is_some()) {
        for sec in s {
            let matches = sec
                .condition
                .as_ref()
                .is_none_or(|(op, n)| match op.as_str() {
                    "<" => v < *n,
                    "<=" => v <= *n,
                    ">" => v > *n,
                    ">=" => v >= *n,
                    "=" => v == *n,
                    _ => v != *n,
                });
            if matches {
                return Some((sec, v, true));
            }
        }
        return None;
    }
    match s.len() {
        1 => Some((&s[0], v, true)),
        2 => {
            if v < 0. {
                Some((&s[1], v.abs(), false))
            } else {
                Some((&s[0], v, false))
            }
        }
        3 => {
            if v > 0. {
                Some((&s[0], v, false))
            } else if v < 0. {
                Some((&s[1], v.abs(), false))
            } else {
                Some((&s[2], v, false))
            }
        }
        _ => None,
    }
}
fn literal(pattern: &str, syntax: bool) -> String {
    let chars = pattern.chars().collect::<Vec<_>>();
    let (mut i, mut quote) = (0, false);
    let mut out = String::new();
    while i < chars.len() {
        let c = chars[i];
        if c == '"' {
            quote = !quote;
            i += 1;
            continue;
        }
        if !quote && matches!(c, '\\' | '_' | '*') && i + 1 < chars.len() {
            if !syntax {
                match c {
                    '\\' => out.push(chars[i + 1]),
                    '_' => out.push(' '),
                    _ => {}
                }
            }
            i += 2;
            continue;
        }
        if !syntax || !quote {
            out.push(c)
        }
        i += 1;
    }
    out
}
fn date_parts(pattern: &str) -> Option<(bool, bool, bool)> {
    static ELAPSED: OnceLock<regex::Regex> = OnceLock::new();
    static MINUTE: OnceLock<regex::Regex> = OnceLock::new();
    let elapsed_re = ELAPSED.get_or_init(|| re(r"\[(?:h+|m+|s+)\]"));
    let minute = MINUTE.get_or_init(|| re(r"m+\s*:\s*s+|h+\s*:\s*m+"));
    let s = crate::ole::casefold(&literal(pattern, true));
    let elapsed = elapsed_re.is_match(&s);
    let no = elapsed_re.replace_all(&s, "");
    let date = no.contains(['y', 'd']);
    let time = no.contains(['h', 's']) || no.contains("am/pm") || no.contains("a/p");
    let m = no.contains('m');
    if !elapsed && !no.contains(['y', 'd', 'h', 'm', 's']) {
        return None;
    }
    let minute = m && (time || minute.is_match(&s));
    let mut has_date = date || m && !minute;
    let mut has_time = time || minute;
    if elapsed {
        has_time = true;
        if !date {
            has_date = false
        }
    }
    Some((has_date, has_time, elapsed))
}
fn serial(value: f64, epoch: bool, parts: (bool, bool, bool)) -> Option<String> {
    let (date, time, elapsed) = parts;
    if elapsed {
        let seconds = (value.abs() * 86400.).round_ties_even();
        if !seconds.is_finite() {
            return None;
        }
        let n = BigInt::parse_bytes(format!("{seconds:.0}").as_bytes(), 10)?;
        let h = &n / 3600;
        let m = (&n / 60) % 60;
        let s = &n % 60;
        return Some(format!(
            "{}{h}:{m:02}:{s:02}",
            if value < 0. { "-" } else { "" }
        ));
    }
    if !date {
        let seconds = (value.rem_euclid(1.).abs() * 86400.).round_ties_even() as u64 % 86400;
        return Some(format!(
            "{:02}:{:02}:{:02}",
            seconds / 3600,
            seconds / 60 % 60,
            seconds % 60
        ));
    }
    if !epoch && value.floor() == 60. {
        return Some(general(value));
    }
    let corrected = if !epoch && value >= 60. {
        value - 1.
    } else {
        value
    };
    let days = corrected.floor();
    if !(-4_000_000.0..4_000_000.).contains(&days) {
        return Some(general(value));
    }
    let base = chrono::NaiveDate::from_ymd_opt(
        if epoch { 1904 } else { 1899 },
        if epoch { 1 } else { 12 },
        if epoch { 1 } else { 31 },
    )?
    .and_hms_opt(0, 0, 0)?;
    let seconds = (corrected - days) * 86400.;
    let us = ((seconds - seconds.floor()) * 1e6).round_ties_even() as i64;
    let dt = base
        .checked_add_signed(chrono::Duration::days(days as i64))?
        .checked_add_signed(chrono::Duration::seconds(seconds.floor() as i64))?
        .checked_add_signed(chrono::Duration::microseconds(us))?;
    if !(1..=9999).contains(&dt.year()) {
        return Some(general(value));
    }
    Some(if time {
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            dt.year(),
            dt.month(),
            dt.day(),
            dt.hour(),
            dt.minute(),
            dt.second()
        )
    } else {
        format!("{:04}-{:02}-{:02}", dt.year(), dt.month(), dt.day())
    })
}
/// Decimal(.15g).quantize(10^-places, ROUND_HALF_UP), context precision 28.
fn quantize(value: f64, places: usize) -> Option<BigInt> {
    if !value.is_finite() || places > 1_000_026 {
        return None;
    }
    let s = significant(value.abs());
    let (base, exponent) = s
        .split_once('e')
        .map_or((s.as_str(), 0), |(a, b)| (a, b.parse::<i32>().unwrap()));
    let decimals = base.split_once('.').map_or(0, |(_, b)| b.len());
    let digits = base.replace('.', "");
    let n = digits.parse::<BigInt>().ok()?;
    let shift = exponent as i64 - decimals as i64 + places as i64;
    let out = if shift >= 0 {
        if n != BigInt::from(0) && n.to_string().len() + shift as usize > 28 {
            return None;
        }
        if n == BigInt::from(0) {
            n
        } else {
            n * BigInt::from(10).pow(shift as u32)
        }
    } else {
        let k = (-shift) as usize;
        if k > n.to_string().len() {
            BigInt::from(0)
        } else {
            let d = BigInt::from(10).pow(k as u32);
            (&n + &d / 2) / d
        }
    };
    if out.to_string().len() > 28 {
        None
    } else {
        Some(if value.is_sign_negative() { -out } else { out })
    }
}
fn fixed(n: &BigInt, places: usize) -> String {
    let s = n.to_string();
    let sign = if s.starts_with('-') { "-" } else { "" };
    let mut d = s.trim_start_matches('-').to_owned();
    if places == 0 {
        return s;
    }
    if d.len() <= places {
        d = format!("{}{}", "0".repeat(places + 1 - d.len()), d)
    }
    let at = d.len() - places;
    format!("{sign}{}.{}", &d[..at], &d[at..])
}
fn number_span(pattern: &str) -> (String, String, String) {
    static GENERAL: OnceLock<regex::Regex> = OnceLock::new();
    let s = literal(pattern, false);
    if let Some(m) = GENERAL.get_or_init(|| re("(?i)general")).find(&s) {
        return (
            s[..m.start()].into(),
            m.as_str().into(),
            s[m.end()..].into(),
        );
    }
    let positions = s
        .char_indices()
        .filter(|(_, c)| matches!(c, '0' | '#' | '?' | '@'))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let Some(mut start) = positions.first().copied() else {
        return (s, String::new(), String::new());
    };
    let mut end = positions.last().unwrap() + 1;
    while start > 0
        && s.as_bytes()[start - 1].is_ascii()
        && "0123456789#?.,%Ee+-/"
            .as_bytes()
            .contains(&s.as_bytes()[start - 1])
    {
        start -= 1
    }
    while end < s.len()
        && s.as_bytes()[end].is_ascii()
        && "0123456789#?.,%Ee+-/"
            .as_bytes()
            .contains(&s.as_bytes()[end])
    {
        end += 1
    }
    (s[..start].into(), s[start..end].into(), s[end..].into())
}
fn scientific(value: f64, pattern: &str) -> Option<String> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let m = RE
        .get_or_init(|| re(r"([0#?]+)(?:\.([0#?]+))?[Ee]([+-])([0#?]+)"))
        .captures(pattern)?;
    let int = m[1].len();
    let frac = m.get(2).map_or(0, |m| m.len());
    let ep = m[4].len();
    let mut exponent = if value == 0. {
        0
    } else {
        ((value.abs().log10().floor() / int.max(1) as f64).floor() * int.max(1) as f64) as i32
    };
    let mantissa = if value == 0. {
        0.
    } else {
        value / 10f64.powi(exponent)
    };
    let mut rounded = quantize(mantissa, frac)?;
    if rounded.abs_int() >= BigInt::from(10).pow((int + frac) as u32) {
        exponent += int.max(1) as i32;
        rounded /= BigInt::from(10).pow(int.max(1) as u32)
    }
    let sign = if exponent < 0 {
        "-"
    } else if &m[3] == "-" {
        ""
    } else {
        "+"
    };
    Some(format!(
        "{}E{sign}{:0width$}",
        fixed(&rounded, frac),
        exponent.unsigned_abs(),
        width = ep
    ))
}
trait AbsInt {
    fn abs_int(&self) -> BigInt;
}
impl AbsInt for BigInt {
    fn abs_int(&self) -> BigInt {
        if self.sign() == num_bigint::Sign::Minus {
            -self
        } else {
            self.clone()
        }
    }
}
fn rational(value: f64) -> (BigInt, BigInt) {
    let bits = value.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32;
    let m = bits & ((1u64 << 52) - 1);
    let mut m = if e == 0 { m } else { m + (1u64 << 52) };
    let mut power = if e == 0 { -1074 } else { e - 1023 - 52 };
    if m == 0 {
        return (0.into(), 1.into());
    }
    if power < 0 {
        let remove = m.trailing_zeros().min((-power) as u32);
        m >>= remove;
        power += remove as i32;
    }
    let n = BigInt::from(m);
    if power >= 0 {
        (n << power as usize, 1.into())
    } else {
        (n, BigInt::from(1) << (-power) as usize)
    }
}
fn limited_fraction(value: f64, max: BigInt) -> (BigInt, BigInt) {
    let (original_n, original_d) = rational(value);
    if original_d <= max {
        return (original_n, original_d);
    }
    let (mut n, mut d) = (original_n.clone(), original_d.clone());
    let (mut p0, mut q0, mut p1, mut q1) = (
        BigInt::from(0),
        BigInt::from(1),
        BigInt::from(1),
        BigInt::from(0),
    );
    loop {
        let a = &n / &d;
        let q2 = &q0 + &a * &q1;
        if q2 > max {
            break;
        }
        let p2 = &p0 + &a * &p1;
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        let next = &n - &a * &d;
        n = d;
        d = next;
        if d == BigInt::from(0) {
            return (p1, q1);
        }
    }
    let k = (&max - &q0) / &q1;
    let a = &p0 + &k * &p1;
    let b = &q0 + &k * &q1;
    let delta2 = (&p1 * &original_d - &original_n * &q1).abs_int();
    let delta1 = (&a * &original_d - &original_n * &b).abs_int();
    if delta2 * &b <= delta1 * &q1 {
        (p1, q1)
    } else {
        (a, b)
    }
}
fn fraction(value: f64, pattern: &str) -> Option<String> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let m = RE
        .get_or_init(|| re(r"([#0?]*)\s*([0#?]+)/([0-9#?]+)"))
        .captures(pattern)?;
    let mut whole = if m[1].is_empty() {
        0.into()
    } else {
        BigInt::parse_bytes(format!("{:.0}", value.abs().floor()).as_bytes(), 10)?
    };
    let whole_float = whole.to_string().parse::<f64>().ok()?;
    let v = value.abs() - whole_float;
    let fixed = m[3]
        .parse::<BigInt>()
        .ok()
        .filter(|n| *n != BigInt::from(0));
    let (mut n, d) = if let Some(d) = fixed {
        let f = d.to_string().parse::<f64>().ok()?;
        (quantize(v * f, 0)?, d)
    } else {
        limited_fraction(v, BigInt::from(10).pow(m[3].len() as u32) - 1)
    };
    if n == BigInt::from(0) {
        return Some(if whole != BigInt::from(0) {
            whole.to_string()
        } else {
            String::new()
        });
    }
    if n >= d && !m[1].is_empty() {
        whole += &n / &d;
        n %= &d;
    }
    let pad = |v: &BigInt, p: &str| {
        let s = v.to_string();
        format!(
            "{}{}",
            if p.len() > s.len() {
                if p.contains('?') {
                    " "
                } else {
                    "0"
                }
            } else {
                ""
            }
            .repeat(p.len().saturating_sub(s.len())),
            s
        )
    };
    let f = format!("{}/{}", pad(&n, &m[2]), pad(&d, &m[3]));
    Some(if whole != BigInt::from(0) {
        format!("{whole} {f}")
    } else {
        f
    })
}
fn decimal(value: f64, pattern: &str) -> Option<String> {
    let percents = pattern.bytes().filter(|b| *b == b'%').count();
    let mut core = pattern.replace('%', "");
    let last = core
        .char_indices()
        .filter(|(_, c)| matches!(c, '0' | '#' | '?'))
        .map(|(i, _)| i)
        .last();
    let mut commas = 0;
    if let Some(last) = last {
        let suffix = &core[last + 1..];
        if suffix.trim_matches([',', ' ']).is_empty() {
            commas = suffix.bytes().filter(|b| *b == b',').count();
            if commas > 0 {
                core = format!("{}{}", &core[..last + 1], suffix.replace(',', ""));
            }
        }
    }
    let (integer, frac) = core.split_once('.').unwrap_or((&core, ""));
    let ip = integer
        .chars()
        .filter(|c| matches!(c, '0' | '#' | '?'))
        .collect::<String>();
    let fp = frac
        .chars()
        .filter(|c| matches!(c, '0' | '#' | '?'))
        .collect::<String>();
    let scaled = value * 100f64.powi(i32::try_from(percents).ok()?)
        / 1000f64.powi(i32::try_from(commas).ok()?);
    let n = quantize(scaled.abs(), fp.len())?;
    let formatted = fixed(&n, fp.len());
    let (int, frac) = formatted.split_once('.').unwrap_or((&formatted, ""));
    let required = ip.bytes().filter(|b| *b == b'0').count();
    let mut integer = format!("{}{}", "0".repeat(required.saturating_sub(int.len())), int);
    if n == BigInt::from(0) && required == 0 && !ip.is_empty() {
        integer.clear();
    }
    if integer.contains('.') {
        return None;
    }
    if integer != "" && core.split('.').next().unwrap().contains(',') {
        let n = integer.parse::<BigInt>().ok()?.to_string();
        integer = n
            .chars()
            .enumerate()
            .map(|(i, c)| {
                if i > 0 && (n.len() - i) % 3 == 0 {
                    format!(",{c}")
                } else {
                    c.to_string()
                }
            })
            .collect();
    }
    if ip.starts_with('?') && !integer.is_empty() {
        integer = format!(
            "{}{}",
            " ".repeat(ip.len().saturating_sub(integer.len())),
            integer
        )
    }
    let mut visible = frac.chars().map(|c| c.to_string()).collect::<Vec<_>>();
    for (i, p) in fp.bytes().enumerate().rev() {
        if p == b'0' || visible.get(i).is_none_or(|s| s != "0") {
            break;
        }
        visible[i] = if p == b'?' { " ".into() } else { String::new() };
    }
    let frac = visible.concat();
    if core.contains('.') && (!frac.is_empty() || fp.is_empty()) {
        integer.push('.');
        integer.push_str(&frac);
    }
    Some(format!(
        "{}{}{}",
        if scaled < 0. { "-" } else { "" },
        integer,
        "%".repeat(percents)
    ))
}
fn render(pattern: &str, v: f64, minus: bool) -> Option<String> {
    if pattern.is_empty() {
        return Some(String::new());
    }
    let (prefix, core, suffix) = number_span(pattern);
    if core.eq_ignore_ascii_case("general") {
        return Some(format!(
            "{prefix}{}{suffix}",
            general(if minus { v } else { v.abs() })
        ));
    }
    if !core.contains(['0', '#', '?']) {
        return Some(prefix + &suffix);
    }
    static SCI: OnceLock<regex::Regex> = OnceLock::new();
    let mut s = if SCI.get_or_init(|| re(r"[Ee][+-][0#?]")).is_match(&core) {
        scientific(v.abs(), &core)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| general(v.abs()))
    } else if core.contains('/') {
        fraction(v.abs(), &core)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| general(v.abs()))
    } else {
        decimal(v.abs(), &core)?
    };
    if minus && v < 0. {
        s.insert(0, '-')
    }
    Some(prefix + &s + &suffix)
}
fn number_display(value: f64, code: Option<&str>, epoch: bool) -> String {
    let Some(code) = code.filter(|_| value.is_finite()) else {
        return general(value);
    };
    let Some(s) = sections(code) else {
        return general(value);
    };
    let Some((s, v, minus)) = choose(&s, value) else {
        return general(value);
    };
    let s = strip(&s.pattern);
    if let Some(parts) = date_parts(&s.pattern) {
        return serial(v, epoch, parts).unwrap_or_else(|| general(value));
    }
    render(&s.pattern, v, minus).unwrap_or_else(|| general(value))
}
pub fn number(value: f64, code: Option<&str>, epoch: bool) -> crate::Result<String> {
    if value.is_finite() {
        if let Some(s) = code.and_then(sections) {
            if let Some((s, v, _)) = choose(&s, value) {
                if date_parts(&strip(&s.pattern).pattern).is_some_and(|(_, _, elapsed)| elapsed)
                    && !(v.abs() * 86400.).is_finite()
                {
                    return Err(crate::Error::malformed("elapsed serial overflow"));
                }
            }
        }
    }
    Ok(number_display(value, code, epoch))
}
pub fn text(value: &str, code: Option<&str>) -> String {
    let Some(s) = code.and_then(sections) else {
        return value.into();
    };
    if s.len() == 4 {
        return literal(&s[3].pattern, false).replace('@', value);
    }
    if s.len() == 1 && s[0].pattern.contains('@') {
        return literal(&s[0].pattern, false).replace('@', value);
    }
    value.into()
}
