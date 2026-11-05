//! DocVortex ooxml_chart.py chart description and HTML semantics.
pub mod value;
pub mod workbook;
use crate::xml;
use roxmltree::Node;
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
const CHART: &str = "http://schemas.openxmlformats.org/drawingml/2006/chart";
const DRAWING: &str = "http://schemas.openxmlformats.org/drawingml/2006/main";
fn is(n: Node<'_, '_>, name: &str) -> bool {
    n.is_element() && n.tag_name().name() == name && n.tag_name().namespace() == Some(CHART)
}
fn child<'a, 'd>(n: Node<'a, 'd>, name: &str) -> Option<Node<'a, 'd>> {
    n.children().find(|n| is(*n, name))
}
fn title(n: Option<Node<'_, '_>>) -> String {
    n.map(|n| {
        n.descendants()
            .skip(1)
            .filter(|n| n.tag_name().name() == "t" && n.tag_name().namespace() == Some(DRAWING))
            .filter_map(|n| n.text())
            .collect::<String>()
            .trim()
            .into()
    })
    .unwrap_or_default()
}
#[derive(Default, Debug, Serialize)]
pub struct Series {
    pub name_formula: Option<String>,
    pub literal_name: Option<String>,
    pub cat_formula: Option<String>,
    pub x_formula: Option<String>,
    pub val_formula: Option<String>,
    pub y_formula: Option<String>,
    pub bubble_size_formula: Option<String>,
    pub cached_categories: Vec<String>,
    pub cached_x_values: Vec<String>,
    pub cached_values: Vec<String>,
    pub cached_bubble_sizes: Vec<String>,
}
#[derive(Debug, Serialize)]
pub struct Spec {
    pub chart_type: String,
    pub plot_kind: String,
    pub title: String,
    pub category_axis_title: String,
    pub value_axis_title: String,
    pub x_axis_title: String,
    pub has_date_axis: bool,
    pub date_1904: bool,
    pub series: Vec<Series>,
}
fn reference<'a, 'd>(n: Node<'a, 'd>) -> Option<Node<'a, 'd>> {
    ["strRef", "numRef", "multiLvlStrRef"]
        .into_iter()
        .find_map(|name| child(n, name))
}
fn formula(n: Option<Node<'_, '_>>) -> Option<String> {
    child(reference(n?)?, "f")?.text().map(|s| s.trim().into())
}
fn points(n: Node<'_, '_>, date: bool, date1904: bool) -> Vec<String> {
    let mut points = BTreeMap::new();
    for p in n.children().filter(|n| is(*n, "pt")) {
        let Some(index) = p
            .attribute("idx")
            .and_then(|s| super::docx::integer_text(Some(s)))
            .and_then(|s| s.parse::<num_bigint::BigInt>().ok())
        else {
            continue;
        };
        points.insert(
            index,
            value::cache(
                child(p, "v").and_then(|n| n.text()).unwrap_or(""),
                date,
                date1904,
            ),
        );
    }
    let Some(max) = points.keys().last() else {
        return vec![];
    };
    let Ok(max) = max.to_string().parse::<usize>() else {
        return vec![];
    };
    if max >= 100_000 {
        return vec![];
    }
    (0..=max)
        .map(|i| {
            points
                .get(&num_bigint::BigInt::from(i))
                .cloned()
                .unwrap_or_default()
        })
        .collect()
}
fn cached(n: Option<Node<'_, '_>>, date: bool, date1904: bool) -> Vec<String> {
    let Some(r) = n.and_then(reference) else {
        return vec![];
    };
    if is(r, "multiLvlStrRef") {
        let Some(cache) = child(r, "multiLvlStrCache") else {
            return vec![];
        };
        let mut levels = vec![];
        let mut max = None::<usize>;
        for level in cache.children().filter(|n| is(*n, "lvl")) {
            let mut values = BTreeMap::new();
            for p in level.children().filter(|n| is(*n, "pt")) {
                if let Some(index) = p
                    .attribute("idx")
                    .and_then(|s| super::docx::integer_text(Some(s)))
                    .and_then(|s| s.parse::<num_bigint::BigInt>().ok())
                {
                    if let Ok(i) = index.to_string().parse::<usize>() {
                        max = Some(max.map_or(i, |v| v.max(i)));
                    } else if index.sign() != num_bigint::Sign::Minus {
                        return vec![];
                    }
                    values.insert(
                        index,
                        child(p, "v")
                            .and_then(|n| n.text())
                            .unwrap_or("")
                            .to_owned(),
                    );
                }
            }
            levels.push(values);
        }
        let Some(max) = max.filter(|v| *v < 100_000) else {
            return vec![];
        };
        return (0..=max)
            .map(|i| {
                levels
                    .iter()
                    .filter_map(|m| m.get(&num_bigint::BigInt::from(i)))
                    .filter(|s| !s.is_empty())
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" / ")
            })
            .collect();
    }
    child(r, "strCache")
        .or_else(|| child(r, "numCache"))
        .map(|n| points(n, date, date1904))
        .unwrap_or_default()
}
pub fn parse(data: &[u8]) -> Option<Spec> {
    let text = xml::decode(data).ok()?;
    let tree = xml::parse(&text).ok()?;
    let root = tree.root_element();
    let plot = root.descendants().skip(1).find(|n| is(*n, "plotArea"))?;
    let plots = plot
        .children()
        .filter(|n| {
            n.is_element()
                && matches!(
                    n.tag_name().name(),
                    "areaChart"
                        | "area3DChart"
                        | "barChart"
                        | "bar3DChart"
                        | "bubbleChart"
                        | "doughnutChart"
                        | "lineChart"
                        | "line3DChart"
                        | "ofPieChart"
                        | "pieChart"
                        | "pie3DChart"
                        | "radarChart"
                        | "scatterChart"
                        | "stockChart"
                        | "surfaceChart"
                        | "surface3DChart"
                )
        })
        .collect::<Vec<_>>();
    if plots.is_empty() {
        return None;
    }
    let has_date_axis = child(plot, "dateAx").is_some();
    let date1904 = child(root, "date1904").and_then(|n| n.attribute("val")) == Some("1");
    let kinds = plots
        .iter()
        .map(|n| match n.tag_name().name() {
            "scatterChart" => "scatter",
            "bubbleChart" => "bubble",
            _ => {
                if has_date_axis {
                    "date"
                } else {
                    "category"
                }
            }
        })
        .collect::<BTreeSet<_>>();
    if kinds.len() != 1 {
        return None;
    }
    let kind = *kinds.first()?;
    let category_axis_title = title(
        child(plot, "dateAx")
            .or_else(|| child(plot, "catAx"))
            .and_then(|n| child(n, "title")),
    );
    let (mut x_axis_title, mut value_axis_title) = (String::new(), String::new());
    if matches!(kind, "scatter" | "bubble") {
        for axis in plot.children().filter(|n| is(*n, "valAx")) {
            let pos = child(axis, "axPos")
                .and_then(|n| n.attribute("val"))
                .unwrap_or("");
            let text = title(child(axis, "title"));
            if pos == "b" && x_axis_title.is_empty() {
                x_axis_title = text;
            } else if pos == "l" && value_axis_title.is_empty() {
                value_axis_title = text;
            }
        }
        if x_axis_title.is_empty() {
            x_axis_title = category_axis_title.clone();
        }
    } else {
        value_axis_title = title(child(plot, "valAx").and_then(|n| child(n, "title")));
    }
    let mut series = vec![];
    for p in &plots {
        for s in p.children().filter(|n| is(*n, "ser")) {
            let tx = child(s, "tx");
            let literal_name = if let Some(cache) = tx
                .and_then(|n| child(n, "strRef"))
                .and_then(|n| child(n, "strCache"))
            {
                points(cache, false, false).into_iter().next()
            } else {
                tx.and_then(|n| child(n, "v"))
                    .and_then(|n| n.text())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.trim().into())
            };
            series.push(Series {
                name_formula: tx
                    .and_then(|n| child(n, "strRef"))
                    .and_then(|n| child(n, "f"))
                    .and_then(|n| n.text())
                    .map(|s| s.trim().into()),
                literal_name,
                cat_formula: formula(child(s, "cat")),
                x_formula: formula(child(s, "xVal")),
                val_formula: formula(child(s, "val")),
                y_formula: formula(child(s, "yVal")),
                bubble_size_formula: formula(child(s, "bubbleSize")),
                cached_categories: cached(child(s, "cat"), has_date_axis, date1904),
                cached_x_values: cached(child(s, "xVal"), false, false),
                cached_values: cached(child(s, "val").or_else(|| child(s, "yVal")), false, false),
                cached_bubble_sizes: cached(child(s, "bubbleSize"), false, false),
            });
        }
    }
    Some(Spec {
        chart_type: if plots.len() == 1 {
            plots[0].tag_name().name().into()
        } else {
            "comboChart".into()
        },
        plot_kind: kind.into(),
        title: title(
            root.descendants()
                .skip(1)
                .filter(|n| is(*n, "chart"))
                .find_map(|n| child(n, "title")),
        ),
        category_axis_title,
        value_axis_title,
        x_axis_title,
        has_date_axis,
        date_1904: date1904,
        series,
    })
}
pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}
fn table(headers: &[String], columns: &[Vec<String>]) -> String {
    let rows = columns.iter().map(Vec::len).max().unwrap_or(0);
    if rows == 0 || headers.len() != columns.len() {
        return String::new();
    }
    let mut out = String::from("<table><thead><tr>");
    for h in headers {
        out.push_str(&format!("<th>{}</th>", escape(h)));
    }
    out.push_str("</tr></thead><tbody>");
    for r in 0..rows {
        out.push_str("<tr>");
        for c in columns {
            out.push_str(&format!(
                "<td>{}</td>",
                escape(c.get(r).map(String::as_str).unwrap_or(""))
            ));
        }
        out.push_str("</tr>");
    }
    out.push_str("</tbody></table>");
    out
}
fn name(s: &Series, index: usize, wb: Option<&workbook::Workbook>) -> crate::Result<String> {
    Ok(wb
        .zip(s.name_formula.as_ref())
        .map(|(w, f)| w.scalar(f))
        .transpose()?
        .flatten()
        .filter(|s| !s.is_empty())
        .or_else(|| s.literal_name.clone().filter(|s| !s.is_empty()))
        .unwrap_or_else(|| format!("Series{}", index + 1)))
}
pub fn render(spec: &Spec, wb: Option<&workbook::Workbook>) -> crate::Result<String> {
    let mut names = vec![];
    let mut ys = vec![];
    let mut xs = vec![];
    let mut sizes = vec![];
    if matches!(spec.plot_kind.as_str(), "category" | "date") {
        let cat = if let Some(w) = wb {
            if let Some(s) = spec
                .series
                .iter()
                .find(|s| s.cat_formula.as_ref().is_some_and(|s| !s.is_empty()))
            {
                let Some(v) = w.vector(s.cat_formula.as_ref().unwrap())? else {
                    return Ok(String::new());
                };
                v.iter()
                    .map(|v| v.string(spec.has_date_axis, spec.date_1904))
                    .collect()
            } else {
                vec![]
            }
        } else {
            spec.series
                .iter()
                .find(|s| !s.cached_categories.is_empty())
                .map(|s| s.cached_categories.clone())
                .unwrap_or_default()
        };
        for (i, s) in spec.series.iter().enumerate() {
            let y = if let Some(w) = wb {
                let Some(v) = s
                    .val_formula
                    .as_ref()
                    .map(|f| w.vector(f))
                    .transpose()?
                    .flatten()
                else {
                    return Ok(String::new());
                };
                v.iter().map(|v| v.string(false, false)).collect()
            } else {
                s.cached_values.clone()
            };
            names.push(name(s, i, wb)?);
            ys.push(y);
        }
        if names.is_empty() {
            return Ok(String::new());
        }
        names.insert(0, spec.category_axis_title.clone());
        ys.insert(0, cat);
        return Ok(table(&names, &ys));
    }
    let bubble = spec.plot_kind == "bubble";
    for (i, s) in spec.series.iter().enumerate() {
        let (x, y, size) = if let Some(w) = wb {
            let Some(x) = s
                .x_formula
                .as_ref()
                .map(|f| w.vector(f))
                .transpose()?
                .flatten()
            else {
                return Ok(String::new());
            };
            let Some(y) = s
                .y_formula
                .as_ref()
                .map(|f| w.vector(f))
                .transpose()?
                .flatten()
            else {
                return Ok(String::new());
            };
            let z = if bubble {
                let Some(z) = s
                    .bubble_size_formula
                    .as_ref()
                    .map(|f| w.vector(f))
                    .transpose()?
                    .flatten()
                else {
                    return Ok(String::new());
                };
                z.iter().map(|v| v.string(false, false)).collect()
            } else {
                vec![]
            };
            (
                x.iter().map(|v| v.string(false, false)).collect::<Vec<_>>(),
                y.iter().map(|v| v.string(false, false)).collect::<Vec<_>>(),
                z,
            )
        } else {
            if s.cached_x_values.is_empty()
                || s.cached_values.is_empty()
                || bubble && s.cached_bubble_sizes.is_empty()
            {
                return Ok(String::new());
            }
            (
                s.cached_x_values.clone(),
                s.cached_values.clone(),
                s.cached_bubble_sizes.clone(),
            )
        };
        names.push(name(s, i, wb)?);
        xs.push(x);
        ys.push(y);
        sizes.push(size);
    }
    if names.is_empty() {
        return Ok(String::new());
    }
    let shared = xs.iter().skip(1).all(|v| v == &xs[0]);
    let mut headers = vec![];
    let mut columns = vec![];
    if shared {
        headers.push(spec.x_axis_title.clone());
        columns.push(xs[0].clone());
    }
    for i in 0..names.len() {
        if !shared {
            headers.push(format!("{} X", names[i]));
            columns.push(xs[i].clone());
        }
        headers.push(if shared {
            names[i].clone()
        } else {
            format!("{} Y", names[i])
        });
        columns.push(ys[i].clone());
        if bubble {
            headers.push(format!("{} size", names[i]));
            columns.push(sizes[i].clone());
        }
    }
    Ok(table(&headers, &columns))
}
pub fn extract(chart: &[u8], workbook: Option<Arc<[u8]>>) -> crate::Result<String> {
    let spec = parse(chart);
    let wb = workbook
        .filter(|b| !b.is_empty())
        .and_then(|b| workbook::Workbook::open(b).ok());
    let Some(spec) = spec.filter(|s| !s.series.is_empty()) else {
        return wb.map(|w| w.html()).unwrap_or_else(|| Ok(String::new()));
    };
    let cache = render(&spec, None)?;
    if let Some(w) = wb {
        let html = render(&spec, Some(&w))?;
        if !html.is_empty() {
            return Ok(html);
        }
        if !cache.is_empty() {
            return Ok(cache);
        }
        return w.html();
    }
    Ok(cache)
}
