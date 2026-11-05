use super::*;
fn slot(nodes: &[Node], index: usize) -> Result<String> {
    nodes
        .get(index)
        .map(node)
        .transpose()
        .map(|s| s.unwrap_or_default().trim_matches(space).into())
}
fn scripts(lower: &str, upper: &str) -> String {
    format!(
        "{}{}",
        if lower.is_empty() {
            String::new()
        } else {
            format!("_{{{lower}}}")
        },
        if upper.is_empty() {
            String::new()
        } else {
            format!("^{{{upper}}}")
        }
    )
}
fn character(value: &Character) -> Result<String> {
    let Character::Code(code) = value else {
        let Character::Latex(s) = value else {
            unreachable!()
        };
        return Ok(s.clone());
    };
    let c = char::from_u32(*code).ok_or_else(bad)?;
    if let Some(s) = data().characters.get(&c) {
        return Ok(s.clone());
    }
    Ok(match c {
        '\\' => "\\backslash ".into(),
        '#' | '$' | '%' | '&' | '_' | '{' | '}' => format!("\\{c}"),
        '~' => "\\sim ".into(),
        '^' => "\\hat{}".into(),
        c if c < '\u{20}' => return Err(bad()),
        c => c.to_string(),
    })
}
fn text(value: &str) -> String {
    value
        .chars()
        .map(|c| match c {
            '\\' => "\\textbackslash{}".into(),
            '#' | '$' | '%' | '&' | '_' | '{' | '}' => format!("\\{c}"),
            '^' => "\\textasciicircum{}".into(),
            '~' => "\\textasciitilde{}".into(),
            _ => c.to_string(),
        })
        .collect()
}
pub(super) fn node(n: &Node) -> Result<String> {
    match n {
        Node::Sequence(children) => children.iter().map(node).collect(),
        Node::Character(c) => {
            let mut value = character(&c.value)?;
            for embellishment in &c.embellishments {
                value = data()
                    .embellishments
                    .get(embellishment)
                    .ok_or_else(bad)?
                    .replace("MTEF_VALUE", &value);
            }
            if c.v5 {
                if matches!(c.face, 1 | 12) {
                    value = format!("\\mathrm{{{value}}}");
                }
                value = match c.style {
                    1 => format!("\\mathbf{{{value}}}"),
                    2 => format!("\\mathit{{{value}}}"),
                    3 => format!("\\boldsymbol{{\\mathit{{{value}}}}}"),
                    _ => value,
                };
            }
            Ok(value)
        }
        Node::Text(value) => Ok(format!("\\text{{{}}}", text(value))),
        Node::Function(name) => Ok(
            if [
                "arccos", "arcsin", "arctan", "cos", "cosh", "cot", "coth", "csc", "det", "exp",
                "gcd", "hom", "ker", "lg", "lim", "ln", "log", "max", "min", "sec", "sin", "sinh",
                "sup", "tan", "tanh",
            ]
            .contains(&name.as_str())
            {
                format!("\\{name} ")
            } else {
                format!("\\operatorname{{{name}}}")
            },
        ),
        Node::Template {
            v5,
            selector,
            variation,
            options,
            slots,
        } => {
            if *v5 {
                template_v5(*selector, *variation, *options, slots)
            } else {
                template_v3(*selector, *variation, slots)
            }
        }
        Node::Pile(rows) => Ok(format!(
            "\\begin{{gathered}}{}\\end{{gathered}}",
            rows.iter()
                .map(|n| node(n).map(|s| s.trim_matches(space).to_owned()))
                .collect::<Result<Vec<_>>>()?
                .join("\\\\")
        )),
        Node::Matrix { rows, cols, cells } => {
            let cells = cells
                .iter()
                .map(|n| node(n).map(|s| s.trim_matches(space).to_owned()))
                .collect::<Result<Vec<_>>>()?;
            let rows = (0..*rows)
                .map(|r| cells[r * cols..(r + 1) * cols].join("&"))
                .collect::<Vec<_>>();
            Ok(format!(
                "\\begin{{matrix}}{}\\end{{matrix}}",
                rows.join("\\\\")
            ))
        }
        Node::Embellishment | Node::Metadata => Ok(String::new()),
    }
}
fn fence(selector: u8) -> Result<(&'static str, &'static str)> {
    [
        ("\\langle", "\\rangle"),
        ("(", ")"),
        ("\\{", "\\}"),
        ("[", "]"),
        ("|", "|"),
        ("\\|", "\\|"),
        ("\\lfloor", "\\rfloor"),
        ("\\lceil", "\\rceil"),
        ("\\llbracket", "\\rrbracket"),
    ]
    .get(usize::from(selector))
    .copied()
    .ok_or_else(bad)
}
fn big(operator: &str, slots: &[Node]) -> Result<String> {
    Ok(format!(
        "{operator}{}{{{}}}",
        scripts(&slot(slots, 2)?, &slot(slots, 1)?),
        slot(slots, 0)?
    ))
}
fn template_v3(selector: u8, variation: u16, slots: &[Node]) -> Result<String> {
    let s = |i| slot(slots, i);
    Ok(match selector {
        0..=7 => {
            let (mut l, mut r) = fence(selector)?;
            match variation {
                0 => (),
                1 => r = ".",
                2 => l = ".",
                _ => return Err(bad()),
            };
            format!("\\left{l}{}\\right{r}", s(0)?)
        }
        8 => format!("\\left\\{{\\left\\{{{}", s(0)?),
        9 => format!("{}\\right\\}}\\right\\}}", s(0)?),
        10 => format!("{}\\right\\}}\\left\\{{", s(0)?),
        11 => format!("\\left\\{{{}\\right)", s(0)?),
        12 => format!("\\left({}\\right\\}}", s(0)?),
        13 => {
            let index = s(1)?;
            if variation == 1 && !index.is_empty() {
                format!("\\sqrt[{index}]{{{}}}", s(0)?)
            } else {
                format!("\\sqrt{{{}}}", s(0)?)
            }
        }
        14 => format!("\\frac{{{}}}{{{}}}", s(0)?, s(1)?),
        15 | 44 => scripts(&s(0)?, &s(1)?),
        16 => format!("\\underline{{{}}}", s(0)?),
        17 => format!("\\overline{{{}}}", s(0)?),
        18..=20 => format!(
            "\\{}{{{}}}{{{}}}",
            if variation == 0 {
                "overset"
            } else {
                "underset"
            },
            s(0)?,
            ["\\leftarrow", "\\rightarrow", "\\leftrightarrow"][usize::from(selector - 18)]
        ),
        27 => format!("\\overbrace{{{}}}^{{{}}}", s(0)?, s(1)?),
        28 => format!("\\underbrace{{{}}}_{{{}}}", s(0)?, s(1)?),
        21..=26 | 29..=38 => big(
            match selector {
                21 | 24 => "\\int",
                22 | 25 => "\\iint",
                23 | 26 => "\\iiint",
                29 | 30 => "\\sum",
                31 | 32 => "\\prod",
                33 | 34 => "\\coprod",
                35 | 36 => "\\bigcup",
                _ => "\\bigcap",
            },
            slots,
        )?,
        39 => format!("\\lim_{{{}}}^{{{}}}{{{}}}", s(1)?, s(2)?, s(0)?),
        40 => format!("\\overline{{{}}}\\smash{{\\big) {}}}", s(1)?, s(0)?),
        41 => format!("{{{}}}/{{{}}}", s(0)?, s(1)?),
        42 | 43 => {
            let op = s(3)?;
            big(
                if op.is_empty() {
                    if selector == 42 {
                        "\\int"
                    } else {
                        "\\sum"
                    }
                } else {
                    &op
                },
                slots,
            )?
        }
        45 => format!("\\left\\langle {}\\middle|{}\\right\\rangle", s(0)?, s(1)?),
        46 | 47 => {
            let direction = ["leftarrow", "rightarrow", "leftrightarrow"]
                .get(usize::from(variation))
                .ok_or_else(bad)?;
            format!(
                "\\{}{direction}{{{}}}",
                if selector == 46 { "under" } else { "over" },
                s(0)?
            )
        }
        48 => format!("\\overparen{{{}}}", s(0)?),
        _ => return Err(bad()),
    })
}
pub(super) fn validate_v5(selector: u8, v: u16, o: u8) -> Result<()> {
    let valid = match selector {
        0..=8 => v & !3 == 0 && o <= 2,
        9 => v & !0x33 == 0 && o <= 2,
        10 => v <= 1 && o == 0,
        11 => v & !7 == 0 && o == 0,
        12 | 13 | 23..=29 => v & !1 == 0 && o == 0,
        14 => v & !0x3f == 0 && o == 0,
        15..=22 => v & !0x1ff == 0 && o <= 1,
        30 => v & !3 == 0 && o == 0,
        31 => v & !15 == 0 && o == 0 && !(v & 8 != 0 && v & 3 == 3),
        32..=34 => v == 0 && o == 0,
        36 => v & !7 == 0 && v != 0 && v & 1 == 0 && o == 0,
        37 => o == 0 && matches!(v, 0 | 0x1e),
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(bad())
    }
}
fn template_v5(selector: u8, v: u16, o: u8, slots: &[Node]) -> Result<String> {
    validate_v5(selector, v, o)?;
    let s = |i| slot(slots, i);
    Ok(match selector {
        0..=9 => {
            let (left, right) = if selector == 9 {
                let delimiters = ["(", ")", "[", "]"];
                (
                    delimiters[usize::from(v & 3)],
                    delimiters[usize::from((v >> 4) & 3)],
                )
            } else {
                let (l, r) = fence(selector)?;
                (
                    if v & 1 != 0 { l } else { "." },
                    if v & 2 != 0 { r } else { "." },
                )
            };
            let extend = |mut s: String| {
                if s.strip_prefix('\\')
                    .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphabetic()))
                {
                    s.push(' ');
                }
                s
            };
            let l = if selector == 9 {
                left.into()
            } else {
                let value = s(1)?;
                if value.is_empty() {
                    left.into()
                } else {
                    value
                }
            };
            let r = if selector == 9 {
                right.into()
            } else {
                let value = s(2)?;
                if value.is_empty() {
                    right.into()
                } else {
                    value
                }
            };
            format!("\\left{}{}\\right{}", extend(l), s(0)?, extend(r))
        }
        10 => {
            let index = s(1)?;
            if v == 1 && !index.is_empty() {
                format!("\\sqrt[{index}]{{{}}}", s(0)?)
            } else {
                format!("\\sqrt{{{}}}", s(0)?)
            }
        }
        11 => {
            if v & 2 != 0 {
                format!("{{{}}}/{{{}}}", s(0)?, s(1)?)
            } else {
                format!("\\frac{{{}}}{{{}}}", s(0)?, s(1)?)
            }
        }
        12 | 13 => {
            let command = if selector == 12 {
                "underline"
            } else {
                "overline"
            };
            let value = format!("\\{command}{{{}}}", s(0)?);
            if v & 1 != 0 {
                format!("\\{command}{{{value}}}")
            } else {
                value
            }
        }
        14 => {
            let body = s(0)?;
            let op = s(1)?;
            let op = if !op.is_empty() {
                op
            } else {
                if v & 2 != 0 {
                    "\\leftrightharpoons"
                } else if v & 1 != 0 {
                    "\\Leftrightarrow"
                } else if v & 16 != 0 && v & 32 == 0 {
                    "\\leftarrow"
                } else if v & 32 != 0 && v & 16 == 0 {
                    "\\rightarrow"
                } else {
                    "\\leftrightarrow"
                }
                .into()
            };
            if v & 8 != 0 && v & 4 == 0 {
                format!("\\underset{{{body}}}{{{op}}}")
            } else if body.is_empty() {
                op
            } else {
                format!("\\overset{{{body}}}{{{op}}}")
            }
        }
        15..=22 => {
            let default = [
                "\\int", "\\sum", "\\prod", "\\coprod", "\\bigcup", "\\bigcap", "", "",
            ][usize::from(selector - 15)];
            let op = if (16..=20).contains(&selector) {
                default.into()
            } else {
                let op = s(3)?;
                if op.is_empty() {
                    default.into()
                } else {
                    op
                }
            };
            if op.is_empty() {
                return Err(bad());
            }
            big(&op, slots)?
        }
        23 => format!("\\lim{}{{{}}}", scripts(&s(1)?, &s(2)?), s(0)?),
        24 | 25 => {
            let command = match (selector == 24, v & 1 != 0) {
                (true, true) => "overbrace",
                (true, false) => "underbrace",
                (false, true) => "overbracket",
                _ => "underbracket",
            };
            let small = s(1)?;
            format!(
                "\\{command}{{{}}}{}",
                s(0)?,
                if v & 1 != 0 {
                    scripts("", &small)
                } else {
                    scripts(&small, "")
                }
            )
        }
        26 => {
            if v & 1 != 0 {
                format!("\\overline{{{}}}\\smash{{\\big) {}}}", s(1)?, s(0)?)
            } else {
                format!("\\smash{{\\big) {}}}", s(0)?)
            }
        }
        27..=29 => format!(
            "{}{}",
            if v & 1 != 0 { "{}" } else { "" },
            scripts(
                &if selector == 28 { String::new() } else { s(0)? },
                &if selector == 27 { String::new() } else { s(1)? }
            )
        ),
        30 => match v {
            1 => format!("\\left\\langle {}\\right|", s(0)?),
            2 => format!("\\left|{}\\right\\rangle", s(1)?),
            _ => format!("\\left\\langle {}\\middle|{}\\right\\rangle", s(0)?, s(1)?),
        },
        31 => {
            let direction = if v & 8 != 0 {
                if v & 1 != 0 {
                    "leftharpoon"
                } else {
                    "rightharpoon"
                }
            } else if v & 3 == 3 {
                "leftrightarrow"
            } else if v & 1 != 0 {
                "leftarrow"
            } else {
                "rightarrow"
            };
            format!(
                "\\{}{direction}{{{}}}",
                if v & 4 != 0 { "under" } else { "over" },
                s(0)?
            )
        }
        32..=34 => format!(
            "\\{}{{{}}}",
            ["widetilde", "widehat", "overparen"][usize::from(selector - 32)],
            s(0)?
        ),
        36 => format!(
            "\\{}{{{}}}",
            if v & 6 == 6 {
                "xcancel"
            } else if v & 2 != 0 {
                "cancel"
            } else {
                "bcancel"
            },
            s(0)?
        ),
        37 => format!("\\boxed{{{}}}", s(0)?),
        _ => return Err(bad()),
    })
}
