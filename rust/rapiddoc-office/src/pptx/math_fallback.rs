//! Secondary pypptx-with-oxml 1.0.3 formula engine. This is deliberately separate
//! from DocVortex OMML: its defaults, scripts and function spacing differ.
//! See licenses/PYPPTX-WITH-OXML.txt and math_fallback_data.json provenance.
use super::geometry::child;
use crate::{Error, Result};
use roxmltree::{Node, NodeId};
use serde::Deserialize;
use std::{collections::HashMap, sync::OnceLock};
const M: &str = "http://schemas.openxmlformats.org/officeDocument/2006/math";
const A14: &str = "http://schemas.microsoft.com/office/drawing/2010/main";
#[derive(Deserialize)]
struct Data {
    characters: HashMap<String, String>,
    kinds: HashMap<String, String>,
    alphabetic: Vec<[u32; 2]>,
}
fn alpha(c: char) -> bool {
    let code = c as u32;
    let ranges = &data().alphabetic;
    let i = ranges.partition_point(|range| range[1] < code);
    ranges.get(i).is_some_and(|range| range[0] <= code)
}
fn data() -> &'static Data {
    static DATA: OnceLock<Data> = OnceLock::new();
    DATA.get_or_init(|| serde_json::from_str(include_str!("math_fallback_data.json")).unwrap())
}
fn kind(n: Node<'_, '_>) -> &'static str {
    let prefix = if n.tag_name().namespace() == Some(M) {
        "m"
    } else if n.tag_name().namespace() == Some(A14) {
        "a14"
    } else {
        return "";
    };
    data()
        .kinds
        .get(&format!("{prefix}:{}", n.tag_name().name()))
        .map(String::as_str)
        .unwrap_or("")
}
fn mapped(s: &str) -> String {
    data()
        .characters
        .get(s)
        .cloned()
        .unwrap_or_else(|| s.into())
}
fn val<'a, 'i>(pr: Option<Node<'a, 'i>>, name: &str, default: &'a str) -> Result<&'a str> {
    let Some(n) = pr.and_then(|n| child(n, M, name)) else {
        return Ok(default);
    };
    n.attribute((M, "val"))
        .ok_or_else(|| Error::malformed(format!("missing m:{name}/@m:val")))
}
fn script(s: &str, sign: char, empty: bool) -> String {
    if s.is_empty() && !empty {
        return String::new();
    }
    if s.chars().count() > 1 || s.chars().any(|c| "\\ {}[]()^_".contains(c)) {
        format!("{sign}{{{s}}}")
    } else {
        format!("{sign}{s}")
    }
}
fn children(n: Node<'_, '_>, values: &HashMap<NodeId, String>) -> String {
    n.children()
        .filter_map(|n| values.get(&n.id()))
        .cloned()
        .collect()
}
fn render(n: Node<'_, '_>, values: &HashMap<NodeId, String>) -> Result<String> {
    let get = |name| {
        child(n, M, name)
            .and_then(|n| values.get(&n.id()))
            .cloned()
            .unwrap_or_default()
    };
    let req = |name| -> Result<String> {
        let c = child(n, M, name)
            .ok_or_else(|| Error::malformed(format!("missing secondary math {name}")))?;
        Ok(values.get(&c.id()).cloned().unwrap_or_default())
    };
    let list = |name| {
        n.children()
            .filter(|n| n.has_tag_name((M, name)))
            .map(|n| values.get(&n.id()).cloned().unwrap_or_default())
            .collect::<Vec<_>>()
    };
    Ok(match kind(n) {
        "CT_Math" => {
            let node = child(n, M, "oMathPara")
                .or_else(|| child(n, M, "oMath"))
                .ok_or_else(|| Error::malformed("missing a14:m formula"))?;
            values.get(&node.id()).cloned().unwrap_or_default()
        }
        "CT_MathOmathPara" => req("oMath")?,
        "CT_OMath" | "CT_MathBaseArgument" => children(n, values),
        "CT_MathSubscriptArgument"
        | "CT_MathSuperscriptArgument"
        | "CT_MathNumerator"
        | "CT_MathDenominator" => {
            if child(n, M, "e").is_some() {
                get("e")
            } else {
                children(n, values)
            }
        }
        "CT_MathDegree" => get("e"),
        "CT_MathRun" => {
            let t = child(n, M, "t")
                .ok_or_else(|| Error::malformed("missing m:r/m:t"))?
                .text()
                .unwrap_or("");
            t.chars().map(|c| mapped(&c.to_string())).collect()
        }
        "CT_MathFraction" => format!("\\frac{{{}}}{{{}}}", req("num")?, req("den")?),
        "CT_MathSubscript" => format!("{}{}", req("e")?, script(&req("sub")?, '_', false)),
        "CT_MathSuperscript" => format!("{}{}", req("e")?, script(&req("sup")?, '^', true)),
        "CT_MathSubSup" => format!(
            "{}{}{}",
            req("e")?,
            script(&req("sub")?, '_', false),
            script(&req("sup")?, '^', false)
        ),
        "CT_MathPrescript" => {
            let base = req("e")?;
            let sub = get("sub");
            let sup = get("sup");
            let wrap = |s: &str| {
                let v = script(s, '_', false);
                v.strip_prefix('_').unwrap_or("").to_owned()
            };
            if sub.is_empty() && sup.is_empty() {
                base
            } else {
                format!("{{}}_{{{}}}^{{{}}}{base}", wrap(&sub), wrap(&sup))
            }
        }
        "CT_MathRad" => {
            let base = req("e")?;
            let degree = get("deg");
            let hide = if let Some(flag) = child(n, M, "radPr").and_then(|n| child(n, M, "degHide"))
            {
                match flag
                    .attribute((M, "val"))
                    .unwrap_or("true")
                    .to_lowercase()
                    .as_str()
                {
                    "1" | "true" | "on" => true,
                    "0" | "false" | "off" => false,
                    _ => return Err(Error::malformed("invalid radical degree boolean")),
                }
            } else {
                false
            };
            if !degree.trim().is_empty() && !hide {
                format!("\\sqrt[{degree}]{{{base}}}")
            } else {
                format!("\\sqrt{{{base}}}")
            }
        }
        "CT_MathLimLow" => {
            let mut name = req("e")?.trim().to_owned();
            if !name.starts_with('\\') && name.chars().any(alpha) && name.chars().count() > 1 {
                name = format!("\\operatorname{{{name}}}");
            }
            format!("{name}_{{{}}}", req("lim")?)
        }
        "CT_MathFName" => {
            if child(n, M, "limLow").is_some() {
                get("limLow")
            } else {
                let raw = list("r").join("");
                let name = raw.trim();
                let known="sin cos tan log ln exp det gcd lim mod max min arcsin sinh arccos cosh arctan tanh cot coth sec csc lg inf sup liminf limsup arg deg dim hom ker Pr sgn";
                if name.starts_with('\\') {
                    name.into()
                } else if known.split(' ').any(|s| s == name) {
                    format!("\\{name} ")
                } else if let Some(v) = data().characters.get(name).filter(|s| s.starts_with('\\'))
                {
                    v.clone()
                } else if name.chars().count() > 1 && name.chars().all(alpha) {
                    format!("\\operatorname{{{name}}} ")
                } else {
                    name.into()
                }
            }
        }
        "CT_MathFunc" => {
            let name = req("fName")?;
            let arg = req("e")?;
            let lim = (name.contains('_') || name.contains('^')) && name.trim().ends_with('}');
            let enclosed = [('(', ')'), ('[', ']'), ('{', '}')]
                .iter()
                .any(|(l, r)| arg.starts_with(*l) && arg.ends_with(*r));
            if lim {
                format!(
                    "{}{}{arg}",
                    name.trim_end(),
                    if arg.is_empty() { "" } else { " " }
                )
            } else if !enclosed && !arg.is_empty() {
                format!("{}({arg})", name.trim_end())
            } else {
                format!("{}{arg}", name.trim_end())
            }
        }
        "CT_MathBar" => {
            let base = req("e")?;
            match val(child(n, M, "barPr"), "pos", "top")? {
                "bot" => format!("\\underline{{{base}}}"),
                "top" => {
                    let multi = base.chars().count() > 1
                        && !(base.starts_with('\\') && !base.trim().contains(' '));
                    format!("\\{}{{{base}}}", if multi { "overline" } else { "bar" })
                }
                _ => base,
            }
        }
        "CT_MathMatrixRow" => list("e").join(" & "),
        "CT_MathMatrix" => format!(
            "\\begin{{matrix}} {} \\end{{matrix}}",
            list("mr").join(" \\\\ ")
        ),
        "CT_MathEqArray" => {
            let lines = list("e");
            if lines.len() > 1 {
                format!(
                    "\\begin{{subarray}}{{c}} {} \\end{{subarray}}",
                    lines.join(" \\\\ ")
                )
            } else {
                lines.join("")
            }
        }
        "CT_MathDelimiter" => delimiter(n, values)?,
        "CT_MathNary" => nary(n, values)?,
        "CT_MathGroupChar" | "CT_MathAccent" => {
            let group = kind(n) == "CT_MathGroupChar";
            let base = req("e")?;
            let pr = child(n, M, if group { "groupChrPr" } else { "accPr" });
            let ch = val(pr, "chr", "")?;
            let top = !group || val(pr, "pos", if pr.is_some() { "top" } else { "" })? == "top";
            let command = match ch {
                "→" => Some("vec"),
                "←" if group => Some("overleftarrow"),
                "̅" | "_" => Some("bar"),
                "̂" | "^" => Some("hat"),
                "̃" | "~" => Some("tilde"),
                "̇" => Some("dot"),
                "̈" => Some("ddot"),
                "́" => Some("acute"),
                "̀" => Some("grave"),
                "̌" => Some("check"),
                "̆" => Some("breve"),
                _ => None,
            };
            if !top {
                format!("{{{base}}}")
            } else if let Some(cmd) = command {
                format!("\\{cmd}{{{base}}}")
            } else if !ch.is_empty() {
                if let Some(m) = data().characters.get(ch).filter(|s| s.starts_with('\\')) {
                    let simple = [
                        "\\vec", "\\bar", "\\hat", "\\tilde", "\\dot", "\\ddot", "\\acute",
                        "\\grave", "\\check", "\\breve",
                    ]
                    .iter()
                    .any(|s| m.starts_with(s));
                    if simple {
                        format!("{}{{{base}}}", m.trim())
                    } else if group {
                        format!("{{{ch}}}{{{base}}}")
                    } else {
                        format!("\\operatorname{{{}}}{{{base}}}", m.trim())
                    }
                } else {
                    format!("{{{ch}}}{{{base}}}")
                }
            } else {
                format!("{{{base}}}")
            }
        }
        _ => String::new(),
    })
}
fn nary(n: Node<'_, '_>, values: &HashMap<NodeId, String>) -> Result<String> {
    let pr = child(n, M, "naryPr");
    let ch = val(pr, "chr", "")?;
    let integral = matches!(ch, "" | "∫" | "∬" | "∭" | "∮" | "∯" | "∰" | "⨌");
    let op = match ch {
        "" | "∫" => "\\int ",
        "∬" => "\\iint ",
        "∭" => "\\iiint ",
        "∮" => "\\oint ",
        "∑" => "\\sum ",
        "∏" => "\\prod ",
        "∐" => "\\coprod ",
        "⋃" => "\\bigcup ",
        "⋂" => "\\bigcap ",
        "⋀" => "\\bigwedge ",
        "⋁" => "\\bigvee ",
        "⨆" => "\\bigsqcup ",
        "⨄" => "\\biguplus ",
        "⨁" => "\\bigoplus ",
        "⨂" => "\\bigotimes ",
        "⨀" => "\\bigodot ",
        "⨌" => "\\iiiint ",
        _ => "",
    };
    let op = if op.is_empty() {
        let m = mapped(ch);
        if !m.starts_with('\\') && m.chars().count() > 1 {
            format!("\\operatorname{{{m}}} ")
        } else {
            m
        }
    } else {
        op.into()
    };
    let get = |name| {
        child(n, M, name)
            .and_then(|n| values.get(&n.id()))
            .cloned()
            .unwrap_or_default()
    };
    let hide = |name| -> Result<bool> { Ok(matches!(val(pr, name, "")?, "on" | "1" | "true")) };
    let sub = if hide("subHide")? {
        String::new()
    } else {
        script(&get("sub"), '_', false)
    };
    let sup = if hide("supHide")? {
        String::new()
    } else {
        script(&get("sup"), '^', false)
    };
    let expr = child(n, M, "e").ok_or_else(|| Error::malformed("missing nary expression"))?;
    let mut expr = values.get(&expr.id()).cloned().unwrap_or_default();
    let enclosed = |s: &str| s.starts_with(['{', '(', '[']) || s.starts_with("\\left");
    let mut spacing = " ";
    if integral && !expr.is_empty() {
        static DIFFERENTIAL: OnceLock<regex::Regex> = OnceLock::new();
        let re =
            DIFFERENTIAL.get_or_init(|| regex::Regex::new(r"(?s)^(.*\S)(d[a-zA-Z])$").unwrap());
        // Python's dot does not include newlines.
        let differential = if expr.contains('\n') {
            None
        } else {
            re.captures(&expr)
        };
        if let Some(c) = differential {
            expr = format!("{}\\,{}", &c[1], &c[2]);
            spacing = "";
        } else if enclosed(&expr) {
            spacing = "";
        }
    } else if expr.is_empty() || enclosed(&expr) || expr.starts_with("\\begin") {
        spacing = "";
    }
    Ok(format!("{}{sub}{sup}{spacing}{expr}", op.trim()))
}
fn delimiter(n: Node<'_, '_>, values: &HashMap<NodeId, String>) -> Result<String> {
    let pr = child(n, M, "dPr");
    let begin = val(pr, "begChr", "(")?;
    let end = val(pr, "endChr", ")")?;
    let sep = val(pr, "sepChr", "")?;
    let args = n
        .children()
        .filter(|n| n.has_tag_name((M, "e")))
        .collect::<Vec<_>>();
    if args.len() == 1 {
        let children = args[0]
            .children()
            .filter(|n| n.is_element())
            .collect::<Vec<_>>();
        if children.len() == 1 && kind(children[0]) == "CT_MathMatrix" {
            let env = match (begin, end) {
                ("(", ")") => Some("pmatrix"),
                ("[", "]") => Some("bmatrix"),
                ("{", "}") => Some("Bmatrix"),
                ("|", "|") => Some("vmatrix"),
                ("‖", "‖") => Some("Vmatrix"),
                _ => None,
            };
            if let Some(env) = env {
                let rows = children[0]
                    .children()
                    .filter(|n| n.has_tag_name((M, "mr")))
                    .filter_map(|n| values.get(&n.id()))
                    .cloned()
                    .collect::<Vec<_>>();
                return Ok(format!(
                    "\\begin{{{env}}} {} \\end{{{env}}}",
                    rows.join(" \\\\ ")
                ));
            }
        }
    }
    let mapping = |s: &str, right: bool| -> String {
        let m = match s {
            "{" => "\\{",
            "}" => "\\}",
            "|" => {
                if right {
                    "\\rvert "
                } else {
                    "\\lvert "
                }
            }
            "‖" => {
                if right {
                    "\\rVert "
                } else {
                    "\\lVert "
                }
            }
            "<" | "⟨" | "〈" => "\\langle ",
            ">" | "⟩" | "〉" => "\\rangle ",
            "⌈" => "\\lceil ",
            "⌉" => "\\rceil ",
            "⌊" => "\\lfloor ",
            "⌋" => "\\rfloor ",
            "\\" => "\\backslash ",
            "↑" => "\\uparrow ",
            "↓" => "\\downarrow ",
            "↕" => "\\updownarrow ",
            "⇑" => "\\Uparrow ",
            "⇓" => "\\Downarrow ",
            "⇕" => "\\Updownarrow ",
            "⌜" => "\\ulcorner ",
            "⌝" => "\\urcorner ",
            "⌞" => "\\llcorner ",
            "⌟" => "\\lrcorner ",
            _ => "",
        };
        let mut m = if m.is_empty() { mapped(s) } else { m.into() };
        if m.starts_with('\\') && m.chars().last().is_some_and(alpha) {
            m.push(' ');
        }
        m
    };
    let mut left = mapping(begin, false);
    let right = mapping(end, true);
    if args.is_empty() {
        return Ok(format!(
            "\\left{} \\right{}",
            if left.trim().is_empty() {
                "."
            } else {
                left.trim()
            },
            if right.trim().is_empty() {
                "."
            } else {
                right.trim()
            }
        ));
    }
    let sep = mapped(sep);
    let sep = if [
        "+", "-", "\\pm", "\\times", "\\div", "\\cdot", "=", "<", ">", "\\le", "\\ge", "\\equiv",
        "\\approx",
    ]
    .contains(&sep.trim())
    {
        format!(" {} ", sep.trim())
    } else {
        sep
    };
    let content = args
        .into_iter()
        .filter_map(|n| values.get(&n.id()))
        .cloned()
        .collect::<Vec<_>>()
        .join(&sep);
    if left.trim().is_empty() {
        left = ".".into();
    }
    let right = if right.trim().is_empty() {
        "."
    } else {
        right.trim()
    };
    Ok(format!("\\left{left}{content}\\right{right}"))
}
pub fn convert(root: Node<'_, '_>) -> Result<String> {
    let mut values = HashMap::new();
    let mut work = vec![(root, false)];
    // Follow only the children accessed by the source method. Eagerly walking
    // ignored children changes errors (e.g. hidden n-ary bounds or a numerator
    // that selects its m:e and ignores other malformed sibling runs).
    while let Some((node, ready)) = work.pop() {
        if ready {
            values.insert(node.id(), render(node, &values)?);
            continue;
        }
        work.push((node, true));
        let kind = kind(node);
        let names: Option<Vec<&str>> = match kind {
            "CT_Math" => Some(vec![if child(node, M, "oMathPara").is_some() {
                "oMathPara"
            } else {
                "oMath"
            }]),
            "CT_MathOmathPara" => Some(vec!["oMath"]),
            "CT_MathRun" => Some(vec![]),
            "CT_MathSubscriptArgument"
            | "CT_MathSuperscriptArgument"
            | "CT_MathNumerator"
            | "CT_MathDenominator"
                if child(node, M, "e").is_some() =>
            {
                Some(vec!["e"])
            }
            "CT_MathDegree" => Some(vec!["e"]),
            "CT_MathFName" => Some(vec![if child(node, M, "limLow").is_some() {
                "limLow"
            } else {
                "r"
            }]),
            "CT_MathMatrix" => Some(vec!["mr"]),
            "CT_MathMatrixRow" | "CT_MathEqArray" | "CT_MathDelimiter" => Some(vec!["e"]),
            "CT_MathNary" => {
                let mut names = vec!["e"];
                let pr = child(node, M, "naryPr");
                if !matches!(val(pr, "subHide", "")?, "on" | "1" | "true") {
                    names.push("sub");
                }
                if !matches!(val(pr, "supHide", "")?, "on" | "1" | "true") {
                    names.push("sup");
                }
                Some(names)
            }
            _ => None,
        };
        let dependencies = node
            .children()
            .filter(|n| !self::kind(*n).is_empty())
            .filter(|n| {
                names
                    .as_ref()
                    .is_none_or(|names| names.contains(&n.tag_name().name()))
            })
            .collect::<Vec<_>>();
        work.extend(dependencies.into_iter().rev().map(|n| (n, false)));
    }
    Ok(values.remove(&root.id()).unwrap_or_default())
}
