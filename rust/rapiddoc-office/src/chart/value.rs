//! Original openpyxl Excel date conversion and ooxml_chart stringification.
use chrono::{Datelike, Duration, NaiveDate, Timelike};
use num_bigint::BigInt;
use std::sync::OnceLock;
#[derive(Debug, Clone)]
pub enum Value {
    Empty,
    Text(String),
    Bool(bool),
    Int(BigInt),
    Float(f64),
    DateTime(String),
}
impl Value {
    pub fn string(&self, date_hint: bool, date1904: bool) -> String {
        match self {
            Self::Empty => String::new(),
            Self::Text(s) => s.clone(),
            Self::Bool(b) => {
                if date_hint {
                    serial(f64::from(*b), date1904)
                        .unwrap_or_else(|| if *b { "True" } else { "False" }.into())
                } else {
                    if *b { "True" } else { "False" }.into()
                }
            }
            Self::Int(i) => {
                if date_hint {
                    serial(i.to_string().parse().unwrap_or(f64::INFINITY), date1904)
                        .unwrap_or_else(|| i.to_string())
                } else {
                    i.to_string()
                }
            }
            Self::Float(f) => {
                if date_hint {
                    serial(*f, date1904).unwrap_or_else(|| number(*f))
                } else {
                    number(*f)
                }
            }
            Self::DateTime(s) => {
                if date_hint {
                    s.strip_suffix(" 00:00:00").unwrap_or(s).into()
                } else {
                    s.clone()
                }
            }
        }
    }
}
pub fn number(v: f64) -> String {
    if v == 0. {
        return "0".into();
    }
    if v.is_nan() {
        return "nan".into();
    }
    if v == f64::INFINITY {
        return "inf".into();
    }
    if v == f64::NEG_INFINITY {
        return "-inf".into();
    }
    if v.fract() == 0. {
        return format!("{v:.0}");
    }
    if v.abs() < 1e-4 {
        let s = format!("{v:e}");
        let (a, b) = s.split_once('e').unwrap();
        let exp: i32 = b.parse().unwrap();
        return format!(
            "{a}e{}{v:02}",
            if exp < 0 { '-' } else { '+' },
            v = exp.unsigned_abs()
        );
    }
    v.to_string()
}
fn time(hours: u32, minutes: u32, seconds: u32, micro: u32) -> String {
    let mut s = format!("{hours:02}:{minutes:02}:{seconds:02}");
    if micro != 0 {
        s.push_str(&format!(".{micro:06}"));
    }
    s
}
pub fn serial(v: f64, date1904: bool) -> Option<String> {
    excel(v, date1904, true)
}
pub fn excel(v: f64, date1904: bool, short_midnight: bool) -> Option<String> {
    if !v.is_finite() {
        return None;
    }
    let mut day = v.floor();
    let millis = ((v - day) * 86_400. * 1000.).round_ties_even() as i64;
    if (0. ..1.).contains(&v) && millis < 86_400_000 {
        let t = millis as u32;
        return Some(time(
            t / 3_600_000,
            t / 60_000 % 60,
            t / 1000 % 60,
            t % 1000 * 1000,
        ));
    }
    if v > 0. && v < 60. && !date1904 {
        day += 1.;
    }
    if day.abs() > 4_000_000. {
        return None;
    }
    let epoch = if date1904 {
        NaiveDate::from_ymd_opt(1904, 1, 1)?
    } else {
        NaiveDate::from_ymd_opt(1899, 12, 30)?
    };
    let dt = epoch
        .and_hms_opt(0, 0, 0)?
        .checked_add_signed(Duration::days(day as i64))?
        .checked_add_signed(Duration::milliseconds(millis))?;
    if !(1..=9999).contains(&dt.year()) {
        return None;
    }
    let date = format!("{:04}-{:02}-{:02}", dt.year(), dt.month(), dt.day());
    if short_midnight && dt.time().num_seconds_from_midnight() == 0 && dt.nanosecond() == 0 {
        return Some(date);
    }
    Some(format!(
        "{date} {}",
        time(dt.hour(), dt.minute(), dt.second(), dt.nanosecond() / 1000)
    ))
}
pub fn cache(value: &str, date_hint: bool, date1904: bool) -> String {
    if date_hint {
        if let Ok(n) = value.trim().parse::<f64>() {
            if let Some(s) = serial(n, date1904) {
                return s;
            }
        }
    }
    value.into()
}

/// Python timedelta string, with Excel's millisecond rounding for numeric cells.
pub fn duration(days: f64, milliseconds: bool) -> Option<String> {
    if !days.is_finite() || days.abs() >= 1_000_000_000. {
        return None;
    }
    let day = days.floor() as i128;
    let fraction = ((days - days.floor()) * 86_400_000_000.).round_ties_even() as i128;
    let micros = day * 86_400_000_000
        + if milliseconds {
            (fraction as f64 / 1000.).round_ties_even() as i128 * 1000
        } else {
            fraction
        };
    let d = micros.div_euclid(86_400_000_000);
    if !(-999_999_999..=999_999_999).contains(&d) {
        return None;
    }
    let rem = micros.rem_euclid(86_400_000_000);
    let mut out = if d == 0 {
        String::new()
    } else {
        format!("{d} day{}, ", if d == 1 || d == -1 { "" } else { "s" })
    };
    out.push_str(&format!(
        "{}:{:02}:{:02}",
        rem / 3_600_000_000,
        rem / 60_000_000 % 60,
        rem / 1_000_000 % 60
    ));
    if rem % 1_000_000 != 0 {
        out.push_str(&format!(".{:06}", rem % 1_000_000));
    }
    Some(out)
}

/// openpyxl's prefix-matching ISO reader, including Excel's ISO duration cells.
pub fn iso(s: &str) -> Option<String> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(
            r"^(?:(\d{4})-(\d{2})-(\d{2}))?T?(?:(\d{2}):(\d{2})(?::(\d{2})(\.\d{1,3})?)?)?Z?",
        )
        .unwrap()
    });
    let c = re.captures(s)?;
    if c.get(1).is_some() || c.get(4).is_some() {
        let n = |i| c.get(i).map_or(Some(0), |v| v.as_str().parse::<u32>().ok());
        let date = if c.get(1).is_some() {
            {
                let year = n(1)?;
                if year == 0 {
                    return None;
                }
                Some(NaiveDate::from_ymd_opt(year as i32, n(2)?, n(3)?)?)
            }
        } else {
            None
        };
        let t = if c.get(4).is_some() {
            let micro = c
                .get(7)
                .map(|v| {
                    v.as_str()
                        .parse::<f64>()
                        .ok()
                        .map(|v| (v * 1_000_000.) as u32)
                })
                .unwrap_or(Some(0))?;
            Some(chrono::NaiveTime::from_hms_micro_opt(
                n(4)?,
                n(5)?,
                n(6)?,
                micro,
            )?)
        } else {
            None
        };
        return match (date, t) {
            (Some(d), Some(t)) => Some(format!(
                "{d} {}",
                time(t.hour(), t.minute(), t.second(), t.nanosecond() / 1000)
            )),
            (Some(d), None) => Some(d.to_string()),
            (None, Some(t)) => Some(time(
                t.hour(),
                t.minute(),
                t.second(),
                t.nanosecond() / 1000,
            )),
            _ => None,
        };
    }
    static DUR: OnceLock<regex::Regex> = OnceLock::new();
    let re = DUR.get_or_init(|| {
        regex::Regex::new(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d{1,3})?)S)?").unwrap()
    });
    let c = re.captures(s)?;
    if (1..=3).all(|i| c.get(i).is_none()) {
        return None;
    }
    let n = |i| {
        c.get(i)
            .map_or(Some(0.), |v| v.as_str().parse::<f64>().ok())
    };
    duration((n(1)? * 3600. + n(2)? * 60. + n(3)?) / 86400., false)
}
