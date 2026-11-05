//! Existing PPTX group transforms and picture filtering, without rasterization.
use roxmltree::Node;
use serde::Serialize;

pub type Bbox = [f64; 4];
pub const P: &str = "http://schemas.openxmlformats.org/presentationml/2006/main";
pub const A: &str = "http://schemas.openxmlformats.org/drawingml/2006/main";
const M: &str = "http://schemas.openxmlformats.org/officeDocument/2006/math";

pub(crate) fn child<'a, 'i>(node: Node<'a, 'i>, ns: &str, name: &str) -> Option<Node<'a, 'i>> {
    node.children().find(|n| n.has_tag_name((ns, name)))
}

#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub scale_x: f64,
    pub scale_y: f64,
    pub translate_x: f64,
    pub translate_y: f64,
}
impl Default for Transform {
    fn default() -> Self {
        Self {
            scale_x: 1.,
            scale_y: 1.,
            translate_x: 0.,
            translate_y: 0.,
        }
    }
}
impl Transform {
    pub fn compose(self, inner: Self) -> Self {
        Self {
            scale_x: self.scale_x * inner.scale_x,
            scale_y: self.scale_y * inner.scale_y,
            translate_x: self.scale_x * inner.translate_x + self.translate_x,
            translate_y: self.scale_y * inner.translate_y + self.translate_y,
        }
    }
    pub fn apply(self, b: Bbox) -> Bbox {
        [
            self.scale_x * b[0] + self.translate_x,
            self.scale_y * b[1] + self.translate_y,
            self.scale_x * b[2] + self.translate_x,
            self.scale_y * b[3] + self.translate_y,
        ]
    }
    pub fn group(shape: Node<'_, '_>) -> Self {
        let read = || {
            let x = child(child(shape, P, "grpSpPr")?, A, "xfrm")?;
            let [ox, oy, ex, ey] = dimensions_from_transform(x)?;
            let off = child(x, A, "chOff")?;
            let ext = child(x, A, "chExt")?;
            let cx = number(off, "x")?;
            let cy = number(off, "y")?;
            let w = number(ext, "cx")?;
            let h = number(ext, "cy")?;
            if ex <= 0. || ey <= 0. || w <= 0. || h <= 0. {
                return None;
            }
            Some(Self {
                scale_x: ex / w,
                scale_y: ey / h,
                translate_x: ox - cx * (ex / w),
                translate_y: oy - cy * (ey / h),
            })
        };
        read().unwrap_or_default()
    }
}
fn number(node: Node<'_, '_>, attr: &str) -> Option<f64> {
    // The XML integer type is validated by python-pptx before float conversion.
    node.attribute(attr)?.parse::<i64>().ok().map(|v| v as f64)
}
fn dimensions_from_transform(x: Node<'_, '_>) -> Option<[f64; 4]> {
    let off = child(x, A, "off")?;
    let ext = child(x, A, "ext")?;
    Some([
        number(off, "x")?,
        number(off, "y")?,
        number(ext, "cx")?,
        number(ext, "cy")?,
    ])
}
pub(crate) fn dimensions(shape: Node<'_, '_>) -> [Option<f64>; 4] {
    let x = child(shape, P, "xfrm").or_else(|| {
        child(shape, P, "spPr")
            .or_else(|| child(shape, P, "grpSpPr"))
            .and_then(|n| child(n, A, "xfrm"))
    });
    let Some(x) = x else {
        return [None; 4];
    };
    let off = child(x, A, "off");
    let ext = child(x, A, "ext");
    [
        off.and_then(|n| number(n, "x")),
        off.and_then(|n| number(n, "y")),
        ext.and_then(|n| number(n, "cx")),
        ext.and_then(|n| number(n, "cy")),
    ]
}
pub(crate) fn bbox(d: [Option<f64>; 4]) -> Option<Bbox> {
    let [Some(x), Some(y), Some(w), Some(h)] = d else {
        return None;
    };
    (w > 0. && h > 0.).then_some([x, y, x + w, y + h])
}

#[derive(Clone, Debug, Serialize)]
pub struct Shape<'a, 'i> {
    #[serde(skip)]
    pub node: Node<'a, 'i>,
    pub id: Option<u32>,
    pub bbox: Option<Bbox>,
    pub has_text: bool,
    pub picture: bool,
}
pub(crate) fn shape_nodes<'a, 'i>(tree: Node<'a, 'i>) -> impl Iterator<Item = Node<'a, 'i>> {
    fn is_shape(n: Node<'_, '_>) -> bool {
        n.tag_name().namespace() == Some(P)
            && matches!(
                n.tag_name().name(),
                "sp" | "grpSp" | "graphicFrame" | "cxnSp" | "pic" | "contentPart"
            )
    }
    const MC: &str = "http://schemas.openxmlformats.org/markup-compatibility/2006";
    tree.children().filter_map(|n| {
        if is_shape(n) {
            Some(n)
        } else if n.has_tag_name((MC, "AlternateContent")) {
            // The installed pypptx-with-oxml fork selects the first shape from
            // the first Choice, falling back only when Choice is absent.
            child(n, MC, "Choice")
                .or_else(|| child(n, MC, "Fallback"))
                .and_then(|branch| branch.children().find(|n| is_shape(*n)))
        } else {
            None
        }
    })
}
/// `inherited_dimensions` supplies layout/master dimensions for placeholders.
/// Iterative DFS preserves XML order without an artificial nesting limit.
pub fn flatten<'a, 'i>(
    tree: Node<'a, 'i>,
    mut inherited_dimensions: impl FnMut(Node<'a, 'i>) -> [Option<f64>; 4],
) -> Vec<Shape<'a, 'i>> {
    let mut work = shape_nodes(tree)
        .map(|n| (n, Transform::default()))
        .collect::<Vec<_>>();
    work.reverse();
    let mut result = Vec::new();
    while let Some((node, transform)) = work.pop() {
        if node.has_tag_name((P, "grpSp")) {
            let transform = transform.compose(Transform::group(node));
            let children = shape_nodes(node).collect::<Vec<_>>();
            work.extend(children.into_iter().rev().map(|n| (n, transform)));
            continue;
        }
        let has_frame = child(node, P, "txBody").is_some();
        let has_text = has_frame
            && node.descendants().any(|n| {
                (n.has_tag_name((A, "t")) || n.has_tag_name((M, "t")))
                    && n.text().is_some_and(|t| !t.trim().is_empty())
            });
        let id = node
            .children()
            .flat_map(|n| n.children())
            .find(|n| n.has_tag_name((P, "cNvPr")))
            .and_then(|n| n.attribute("id"))
            .and_then(|s| s.parse().ok());
        result.push(Shape {
            node,
            id,
            bbox: bbox(inherited_dimensions(node)).map(|b| transform.apply(b)),
            has_text,
            picture: node.has_tag_name((P, "pic")) && super::context::placeholder(node).is_none(),
        });
    }
    result
}
pub fn area(b: Bbox) -> f64 {
    (b[2] - b[0]).max(0.) * (b[3] - b[1]).max(0.)
}
pub fn intersection(a: Bbox, b: Bbox) -> Option<Bbox> {
    let c = [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ];
    (c[2] > c[0] && c[3] > c[1]).then_some(c)
}
pub fn union_area(boxes: &[Bbox]) -> f64 {
    let mut xs = boxes.iter().flat_map(|b| [b[0], b[2]]).collect::<Vec<_>>();
    xs.sort_by(f64::total_cmp);
    xs.dedup();
    let mut total = 0.;
    for x in xs.windows(2) {
        if x[1] <= x[0] {
            continue;
        }
        let mut ys = boxes
            .iter()
            .filter(|b| b[0] < x[1] && b[2] > x[0])
            .map(|b| [b[1], b[3]])
            .collect::<Vec<_>>();
        ys.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
        let Some(mut current) = ys.first().copied() else {
            continue;
        };
        let mut height = 0.;
        for y in &ys[1..] {
            if y[0] <= current[1] {
                current[1] = current[1].max(y[1]);
            } else {
                height += (current[1] - current[0]).max(0.);
                current = *y;
            }
        }
        height += (current[1] - current[0]).max(0.);
        total += (x[1] - x[0]) * height;
    }
    total
}
pub fn small_picture(b: Option<Bbox>, width: f64, height: f64) -> bool {
    let Some(b) = b else {
        return false;
    };
    let w = b[2] - b[0];
    let h = b[3] - b[1];
    let a = width * height;
    w > 0. && h > 0. && a > 0. && (w < 0.1 * width || h < 0.1 * height || w * h / a < 0.01)
}
pub fn background_picture(
    b: Option<Bbox>,
    later: impl IntoIterator<Item = (Option<Bbox>, bool)>,
) -> bool {
    let Some(b) = b else {
        return false;
    };
    let a = area(b);
    if a <= 0. {
        return false;
    }
    let overlaps = later
        .into_iter()
        .filter(|(_, text)| *text)
        .filter_map(|(other, _)| intersection(b, other?))
        .collect::<Vec<_>>();
    union_area(&overlaps) / a >= 0.1
}
