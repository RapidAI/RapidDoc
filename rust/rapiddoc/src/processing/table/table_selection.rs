//! 旧 selector：原HTML前缀计数、BeautifulSoup文本空格和已转义OCR字面命中。
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use crate::legacy_table_merge::html::Dom;
use rapiddoc_core::Error;
use rapiddoc_render::quality::{select_metrics, Metrics};
use serde_json::{json, Value};

fn text(dom: &Dom, id: usize, output: &mut String) {
    let n = &dom.nodes[id];
    if matches!(n.tag.as_str(), "script" | "style" | "template") {
        return;
    }
    if n.tag.is_empty() || n.tag == "!cdata" {
        output.push_str(&n.text);
    }
    for &child in &n.children {
        text(dom, child, output);
    }
}
fn metrics(html: &str, texts: &[Value], work: &mut Work) -> Result<Metrics, Error> {
    let lower = html.to_ascii_lowercase();
    let cells = lower.matches("<td").count() + lower.matches("<th").count();
    let dom = Dom::parse_normalization(html, work)?;
    let mut blank = 0;
    for cell in dom.find(0, &["td", "th"]) {
        let mut value = String::new();
        text(&dom, cell, &mut value);
        work.charge(value.len())?;
        blank += usize::from(
            value
                .trim_matches(rapiddoc_native::text_boundary::python_space)
                .is_empty(),
        );
    }
    let mut hits = 0;
    for t in texts {
        let t = t.as_str().ok_or_else(invalid)?;
        work.charge(html.len())?;
        hits += usize::from(html.contains(t));
    }
    if blank > cells {
        return Err(invalid());
    }
    Ok(Metrics {
        cells,
        blank,
        ocr_hits: hits,
    })
}
pub fn select(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let texts = request["texts"]
        .as_array()
        .filter(|a| a.len() <= rapiddoc_core::resources::limit(1000))
        .ok_or_else(invalid)?;
    let candidate = |key: &str| -> Result<&str, Error> {
        if request[key].is_null() {
            Ok("")
        } else {
            request[key].as_str().ok_or_else(invalid)
        }
    };
    let wired = candidate("wired_html")?;
    let wireless = candidate("wireless_html")?;
    if wired.len().saturating_add(wireless.len()) > rapiddoc_core::resources::limit(2_000_000) {
        return Err(limit());
    }
    let mut work = Work(0);
    let result = select_metrics(
        metrics(wired, texts, &mut work)?,
        metrics(wireless, texts, &mut work)?,
    );
    let m = |v: &Metrics| json!({"physical_cells":v.cells,"blank_cells":v.blank,"non_blank_cells":v.non_blank(),"ocr_hits":v.ocr_hits});
    Ok(
        json!({"route":result.route,"reason":result.reason,"wired":m(&result.wired),"wireless":m(&result.wireless),
        "html":if result.route == "wired" {wired} else {wireless}}),
    )
}
