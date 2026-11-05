//! Original native XYCut++ with PPTX's beta=2.0. Invalid boxes stay at the tail.
//! Explicit work stack avoids adding a recursion/character budget to PPTX.
use super::geometry::Bbox;

fn stable_order(ids: &mut [usize], boxes: &[Option<Bbox>]) {
    ids.sort_by(|a, b| {
        let x = boxes[*a].unwrap();
        let y = boxes[*b].unwrap();
        x[1].total_cmp(&y[1])
            .then(x[0].total_cmp(&y[0]))
            .then(a.cmp(b))
    });
}
fn cut(ids: &[usize], boxes: &[Option<Bbox>], axis: usize) -> (f64, f64) {
    let mut sorted = ids.to_vec();
    sorted.sort_by(|a, b| {
        let x = boxes[*a].unwrap();
        let y = boxes[*b].unwrap();
        x[axis]
            .total_cmp(&y[axis])
            .then(x[axis + 2].total_cmp(&y[axis + 2]))
            .then(a.cmp(b))
    });
    let (mut position, mut gap, mut end) = (0., 0., None::<f64>);
    for id in sorted {
        let b = boxes[id].unwrap();
        if let Some(e) = end {
            if b[axis] > e && b[axis] - e > gap {
                gap = b[axis] - e;
                position = (e + b[axis]) / 2.;
            }
        }
        end = Some(end.map_or(b[axis + 2], |e| e.max(b[axis + 2])));
    }
    (position, gap)
}
fn overlap(a: Bbox, b: Bbox) -> f64 {
    let width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.);
    let smaller = (a[2] - a[0]).min(b[2] - b[0]);
    if width <= 0. || smaller <= 0. {
        0.
    } else {
        width / smaller
    }
}
fn cross(ids: &[usize], boxes: &[Option<Bbox>], beta: f64) -> Vec<usize> {
    if ids.len() < 3 {
        return vec![];
    }
    let mut widths = ids
        .iter()
        .map(|i| {
            let b = boxes[*i].unwrap();
            b[2] - b[0]
        })
        .collect::<Vec<_>>();
    widths.sort_by(f64::total_cmp);
    let n = widths.len();
    let median = if n % 2 == 1 {
        widths[n / 2]
    } else {
        (widths[n / 2 - 1] + widths[n / 2]) / 2.
    };
    ids.iter()
        .copied()
        .filter(|i| {
            let b = boxes[*i].unwrap();
            b[2] - b[0] >= beta * median
                && ids
                    .iter()
                    .filter(|j| *j != i && overlap(b, boxes[**j].unwrap()) >= 0.1)
                    .take(2)
                    .count()
                    >= 2
        })
        .collect()
}
fn merge(mut main: Vec<usize>, mut cross: Vec<usize>, boxes: &[Option<Bbox>]) -> Vec<usize> {
    stable_order(&mut cross, boxes);
    let mut out = Vec::with_capacity(main.len() + cross.len());
    for c in cross {
        let top = boxes[c].unwrap()[1];
        out.extend(
            main.iter()
                .copied()
                .filter(|i| boxes[*i].unwrap()[3] <= top),
        );
        main.retain(|i| boxes[*i].unwrap()[3] > top);
        out.push(c);
    }
    out.extend(main);
    out
}
enum Work {
    Segment(Vec<usize>),
    Merge(Vec<usize>),
    Join,
}
pub fn sort(boxes: &[Option<Bbox>]) -> Vec<usize> {
    sort_with(boxes, 2., 0.9)
}
pub fn sort_with(boxes: &[Option<Bbox>], beta: f64, density: f64) -> Vec<usize> {
    let (valid, invalid): (Vec<_>, Vec<_>) = (0..boxes.len()).partition(|i| {
        boxes[*i].is_some_and(|b| b.iter().all(|v| v.is_finite()) && b[2] > b[0] && b[3] > b[1])
    });
    let mut tasks = vec![Work::Segment(valid)];
    let mut values: Vec<Vec<usize>> = vec![];
    while let Some(task) = tasks.pop() {
        match task {
            Work::Join => {
                let right = values.pop().unwrap();
                values.last_mut().unwrap().extend(right);
            }
            Work::Merge(c) => {
                let main = values.pop().unwrap();
                values.push(merge(main, c, boxes));
            }
            Work::Segment(ids) => {
                if ids.len() <= 1 {
                    values.push(ids);
                    continue;
                }
                let c = cross(&ids, boxes, beta);
                let cross_ids = c.iter().copied().collect::<std::collections::BTreeSet<_>>();
                let main = ids
                    .into_iter()
                    .filter(|i| !cross_ids.contains(i))
                    .collect::<Vec<_>>();
                tasks.push(Work::Merge(c));
                if main.len() <= 1 {
                    values.push(main);
                    continue;
                }
                let mut bounds = [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ];
                let mut area = 0.;
                for &i in &main {
                    let b = boxes[i].unwrap();
                    bounds[0] = bounds[0].min(b[0]);
                    bounds[1] = bounds[1].min(b[1]);
                    bounds[2] = bounds[2].max(b[2]);
                    bounds[3] = bounds[3].max(b[3]);
                    area += (b[2] - b[0]) * (b[3] - b[1]);
                }
                let region = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]);
                let dense = if region <= 0. {
                    1.
                } else {
                    (area / region).min(1.)
                };
                let horizontal = cut(&main, boxes, 1);
                let mut vertical = cut(&main, boxes, 0);
                if vertical.1 < 5. && main.len() >= 3 {
                    let threshold = (bounds[2] - bounds[0]) * 0.1;
                    let filtered = main
                        .iter()
                        .copied()
                        .filter(|i| {
                            let b = boxes[*i].unwrap();
                            b[2] - b[0] >= threshold
                        })
                        .collect::<Vec<_>>();
                    if filtered.len() >= 2 && filtered.len() != main.len() {
                        let retry = cut(&filtered, boxes, 0);
                        if retry.1 > vertical.1 && retry.1 >= 5. {
                            vertical = retry;
                        }
                    }
                }
                let mut axis = usize::from(dense > density);
                let (mut preferred, fallback) = if axis == 1 {
                    (horizontal, vertical)
                } else {
                    (vertical, horizontal)
                };
                if preferred.1 < 5. && fallback.1 >= 5. {
                    preferred = fallback;
                    axis = 1 - axis;
                }
                let fallback_sort = |mut ids: Vec<usize>| {
                    stable_order(&mut ids, boxes);
                    ids
                };
                if preferred.1 < 5. {
                    values.push(fallback_sort(main));
                    continue;
                }
                let (left, right): (Vec<_>, Vec<_>) = main.iter().copied().partition(|i| {
                    let b = boxes[*i].unwrap();
                    (b[axis] + b[axis + 2]) / 2. < preferred.0
                });
                if left.is_empty() || right.is_empty() {
                    values.push(fallback_sort(main));
                    continue;
                }
                tasks.push(Work::Join);
                tasks.push(Work::Segment(right));
                tasks.push(Work::Segment(left));
            }
        }
    }
    let mut result = values.pop().unwrap_or_default();
    result.extend(invalid);
    result
}
