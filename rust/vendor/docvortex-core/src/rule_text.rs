//! 新增文字规则的批量几何；文字、字体等价关系与正则由 Python 预先计算。

/// 框、字号、水平、非空字体、字体组、短数字、混合正文、长正文、目录条目、候选资格。
pub type NoiseLine = (
    [f64; 4],
    f64,
    bool,
    bool,
    usize,
    bool,
    bool,
    bool,
    bool,
    bool,
);

/// 依照原始行序逐个判断数字，返回原有顺序的参考行；目录参考行保留稳定纵向排序。
pub fn numeric_noise_candidates(
    lines: &[NoiseLine],
    visual: &[[f64; 4]],
) -> Vec<(usize, Vec<usize>)> {
    let mut groups = std::collections::HashMap::<usize, Vec<usize>>::new();
    for (i, line) in lines.iter().enumerate() {
        groups.entry(line.4).or_default().push(i);
    }
    let mut result = Vec::new();
    for (i, &(bbox, height, _, _, font, _, _, _, _, candidate)) in lines.iter().enumerate() {
        if !candidate {
            continue;
        }
        let gutter = visual.iter().any(|a| {
            visual.iter().any(|b| {
                a[2] + 0.2 * height <= bbox[0]
                    && bbox[2] <= b[0] - 0.2 * height
                    && a[1].max(b[1]) <= bbox[1]
                    && bbox[3] <= a[3].min(b[3])
                    && a[3] - a[1] >= 8.0 * height
                    && b[3] - b[1] >= 8.0 * height
            })
        });
        if gutter {
            let refs: Vec<_> = lines
                .iter()
                .enumerate()
                .filter(|(_, p)| p.4 != font && p.6)
                .map(|(j, _)| j)
                .collect();
            if refs.len() >= 3 {
                result.push((i, refs));
                continue;
            }
        }
        let same = &groups[&font];
        // 数字特征独立于候选资格：同字体旋转数字仍属于原有数量限制。
        if same.len() > 4 || same.iter().any(|&j| !lines[j].5) {
            continue;
        }
        let prose: Vec<_> = lines
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                p.2 && p.3
                    && p.4 != font
                    && 0.7 * p.1 <= height
                    && height <= 1.05 * p.1
                    && p.7
                    && p.0[0] <= bbox[0]
                    && bbox[2] <= p.0[2]
                    && ((p.0[1] + p.0[3] - bbox[1] - bbox[3]) / 2.0).abs() <= 3.0 * p.1
            })
            .map(|(j, _)| j)
            .collect();
        if prose.len() >= 2 {
            let mut union = lines[prose[0]].0;
            for &j in &prose[1..] {
                let b = lines[j].0;
                union = [
                    union[0].min(b[0]),
                    union[1].min(b[1]),
                    union[2].max(b[2]),
                    union[3].max(b[3]),
                ];
            }
            let area = (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0);
            let overlap = (bbox[2].min(union[2]) - bbox[0].max(union[0])).max(0.0)
                * (bbox[3].min(union[3]) - bbox[1].max(union[1])).max(0.0);
            if area > 0.0 && overlap / area >= 0.95 {
                result.push((i, prose));
                continue;
            }
        }
        let mut entries: Vec<_> = lines
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                p.2 && p.3
                    && p.4 != font
                    && height <= 0.5 * p.1
                    && p.8
                    && p.0[0] <= bbox[0]
                    && bbox[2] <= p.0[2]
            })
            .map(|(j, _)| j)
            .collect();
        if entries.len() >= 3 {
            let em = entries
                .iter()
                .map(|&j| lines[j].1)
                .reduce(|a, b| if b < a { b } else { a })
                .unwrap();
            entries.sort_by(|&a, &b| lines[a].0[1].partial_cmp(&lines[b].0[1]).unwrap());
            let left_max = entries
                .iter()
                .map(|&j| lines[j].0[0])
                .reduce(|a, b| if b > a { b } else { a })
                .unwrap();
            let left_min = entries
                .iter()
                .map(|&j| lines[j].0[0])
                .reduce(|a, b| if b < a { b } else { a })
                .unwrap();
            if left_max - left_min <= 0.3 * em
                && entries.windows(2).any(|p| {
                    lines[p[0]].0[3] + 0.1 * em <= bbox[1] && bbox[3] <= lines[p[1]].0[1] - 0.1 * em
                })
            {
                result.push((i, entries));
            }
        }
    }
    result
}

/// 旋转后框、原始框、canonical 字号、角度和源索引，用于公式连通判断。
pub type FormulaLine = ([f64; 4], [f64; 4], f64, i64, i64);

/// 独立公式带的邻接索引，保持候选升序；Python 继续决定种子、认领与栈遍历。
pub fn detached_formula_neighbors(
    boxes: &[[f64; 4]],
    operators: &[bool],
    em: f64,
) -> Vec<Vec<usize>> {
    let mut neighbors = vec![Vec::new(); boxes.len()];
    for (i, a) in boxes.iter().enumerate() {
        for (j, b) in boxes.iter().enumerate() {
            if i == j {
                continue;
            }
            if a[2] - a[0] > 3.0 * em
                && b[2] - b[0] > 3.0 * em
                && operators[i]
                && operators[j]
                && ((a[1] + a[3]) / 2.0 - (b[1] + b[3]) / 2.0).abs() > 0.75 * em
            {
                continue;
            }
            let xgap = (a[0] - b[2]).max(b[0] - a[2]).max(0.0);
            let ygap = (a[1] - b[3]).max(b[1] - a[3]).max(0.0);
            let shorter_x = (a[2] - a[0]).min(b[2] - b[0]);
            let shorter_y = (a[3] - a[1]).min(b[3] - b[1]);
            let xoverlap = if shorter_x > 0.0 {
                (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) / shorter_x
            } else {
                0.0
            };
            let yoverlap = if shorter_y > 0.0 {
                (a[3].min(b[3]) - a[1].max(b[1])).max(0.0) / shorter_y
            } else {
                0.0
            };
            if ygap <= 0.85 * em && xoverlap > 0.1 || xgap <= 2.0 * em && yoverlap >= 0.2 {
                neighbors[i].push(j);
            }
        }
    }
    neighbors
}

/// 两行数值连通规则，表格阻断始终使用原始页面框。
fn formula_connected(a: &FormulaLine, b: &FormulaLine, tables: &[[f64; 4]]) -> bool {
    if a.3 != b.3 {
        return false;
    }
    let ac = ((a.1[0] + a.1[2]) / 2.0, (a.1[1] + a.1[3]) / 2.0);
    let bc = ((b.1[0] + b.1[2]) / 2.0, (b.1[1] + b.1[3]) / 2.0);
    let connector = [
        ac.0.min(bc.0) - 0.1,
        ac.1.min(bc.1) - 0.1,
        ac.0.max(bc.0) + 0.1,
        ac.1.max(bc.1) + 0.1,
    ];
    if tables.iter().any(|t| {
        connector[0] < t[2] && t[0] < connector[2] && connector[1] < t[3] && t[1] < connector[3]
    }) {
        return false;
    }
    let height = a.2.max(b.2);
    let vshort = (a.0[3] - a.0[1]).min(b.0[3] - b.0[1]);
    let vo = if vshort > 0.0 {
        (a.0[3].min(b.0[3]) - a.0[1].max(b.0[1])).max(0.0) / vshort
    } else {
        0.0
    };
    let vg = (a.0[1] - b.0[3]).max(b.0[1] - a.0[3]).max(0.0);
    if vo < 0.2 && vg > 0.6 * height {
        return false;
    }
    let hshort = (a.0[2] - a.0[0]).min(b.0[2] - b.0[0]);
    let ho = if hshort > 0.0 {
        (a.0[2].min(b.0[2]) - a.0[0].max(b.0[0])).max(0.0) / hshort
    } else {
        0.0
    };
    let hg = (a.0[0] - b.0[2]).max(b.0[0] - a.0[2]).max(0.0);
    ho > 0.0 || hg <= 1.5 * height
}

/// 逐次扫描候选并立即追加成员，严格保留 Python 的种子、遍历顺序与重复源索引语义。
pub fn grow_formula_component(
    lines: &[FormulaLine],
    seed_count: usize,
    tables: &[[f64; 4]],
) -> Vec<usize> {
    let mut members: Vec<_> = (0..seed_count.min(lines.len())).collect();
    let mut sources: std::collections::HashSet<_> = members.iter().map(|&i| lines[i].4).collect();
    loop {
        let before = members.len();
        for i in seed_count..lines.len() {
            if !sources.contains(&lines[i].4)
                && members
                    .iter()
                    .any(|&j| formula_connected(&lines[j], &lines[i], tables))
            {
                members.push(i);
                sources.insert(lines[i].4);
            }
        }
        if before == members.len() {
            break;
        }
    }
    members.into_iter().skip(seed_count).collect()
}

/// 局部框、canonical 尺度、字体组、覆盖率、字重、基线、恢复簇、视觉行号、源索引和初始栏。
pub type TailLine = (
    [f64; 4],
    f64,
    Option<usize>,
    f64,
    Option<f64>,
    Option<f64>,
    bool,
    Option<i64>,
    i64,
    usize,
);

/// 保留字体、字重、基线净空与视觉行差的短尾接纳条件。
fn tail_accepts(a: &TailLine, b: &TailLine, lane: (f64, f64), median: f64) -> bool {
    let height = a.1.max(b.1).max(median);
    let width = (lane.1 - lane.0).max(0.1);
    let fonts = a.2.is_some() && b.2.is_some() && a.3 >= 0.75 && b.3 >= 0.75 && a.2 != b.2;
    let weights = match (a.4, b.4) {
        (Some(x), Some(y)) => (x - y).abs() >= 100.0 && x.max(y) >= 1.15 * x.min(y),
        _ => false,
    };
    if a.0[2] - a.0[0] < 0.65 * width
        || b.0[2] - b.0[0] > 0.85 * width
        || b.0[0] < lane.0 - 0.75 * height
        || b.0[2] > lane.1 + 0.75 * height
        || (b.0[0] - a.0[0]).abs() > 0.75 * height
        || fonts
        || weights
    {
        return false;
    }
    let mut gap = if a.6 {
        (b.0[1] - a.0[3]).max(-0.25 * a.1)
    } else {
        b.0[1] - (a.0[1] + a.1)
    };
    if let (Some(x), Some(y)) = (a.5, b.5) {
        if y > x {
            let pitch = y - x;
            if 0.5 * a.1.min(b.1) <= pitch && pitch <= 3.0 * a.1.max(b.1) {
                gap = pitch - a.1;
            }
        }
    }
    if gap < -0.25 * height || gap > 0.9 * height {
        return false;
    }
    if let (Some(x), Some(y)) = (a.7, b.7) {
        let delta = i128::from(y) - i128::from(x);
        if delta <= 0 || delta > 2 {
            return false;
        }
    }
    true
}

/// 严格较小顶边才更新前序证据；同高行批量结算后再更新，平局按源索引保持稳定。
pub fn short_tail_destinations(
    lines: &[TailLine],
    lanes: &[(f64, f64)],
    median: f64,
) -> Vec<(usize, usize)> {
    let mut order: Vec<_> = (0..lines.len()).collect();
    order.sort_by(|&a, &b| {
        lines[a].0[1]
            .partial_cmp(&lines[b].0[1])
            .unwrap()
            .then_with(|| lines[a].0[0].partial_cmp(&lines[b].0[0]).unwrap())
            .then_with(|| lines[a].8.cmp(&lines[b].8))
    });
    let mut previous: Vec<Option<usize>> = vec![None; lanes.len()];
    let mut moves = Vec::new();
    let mut cursor = 0;
    while cursor < order.len() {
        let mut end = cursor + 1;
        while end < order.len() && lines[order[end]].0[1] == lines[order[cursor]].0[1] {
            end += 1;
        }
        let mut settled = Vec::new();
        for &i in &order[cursor..end] {
            let matches: Vec<_> = previous
                .iter()
                .enumerate()
                .filter(|(lane, p)| {
                    p.is_some_and(|j| tail_accepts(&lines[j], &lines[i], lanes[*lane], median))
                })
                .map(|(lane, _)| lane)
                .collect();
            let target = if matches.len() == 1 {
                matches[0]
            } else {
                lines[i].9
            };
            if target != lines[i].9 {
                moves.push((i, target));
            }
            settled.push((i, target));
        }
        for (i, target) in settled {
            let newer = previous[target].is_none_or(|j| {
                lines[i].0[1] > lines[j].0[1]
                    || lines[i].0[1] == lines[j].0[1]
                        && (lines[i].0[0] > lines[j].0[0]
                            || lines[i].0[0] == lines[j].0[0] && lines[i].8 < lines[j].8)
            });
            if newer {
                previous[target] = Some(i);
            }
        }
        cursor = end;
    }
    moves
}

/// 保留原生行组插入顺序、首个匹配组和平局顺序；均值由调用方提供原运行时的精确求和。
pub fn fragment_row_groups<E>(
    items: &[(f64, f64, Option<i64>)],
    tolerance: f64,
    mut mean: impl FnMut(&[usize]) -> Result<f64, E>,
) -> Result<Vec<Vec<usize>>, E> {
    let mut seeds = Vec::<Vec<usize>>::new();
    let mut native = std::collections::HashMap::new();
    let mut geometric = Vec::new();
    for (i, item) in items.iter().enumerate() {
        if let Some(id) = item.2 {
            let next = seeds.len();
            let group = *native.entry(id).or_insert_with(|| {
                seeds.push(Vec::new());
                next
            });
            seeds[group].push(i);
        } else {
            geometric.push(vec![i]);
        }
    }
    seeds.extend(geometric);
    let mut prepared = Vec::new();
    for group in seeds {
        let center = mean(&group)?;
        let left = group
            .iter()
            .map(|&i| items[i].1)
            .reduce(|a, b| if b < a { b } else { a })
            .unwrap();
        prepared.push((group, center, left));
    }
    prepared.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap()
            .then_with(|| a.2.partial_cmp(&b.2).unwrap())
    });
    let mut groups = Vec::<Vec<usize>>::new();
    let mut centers = Vec::<f64>::new();
    for (seed, center, _) in prepared {
        if let Some(target) = centers
            .iter()
            .position(|&other| (center - other).abs() <= tolerance)
        {
            groups[target].extend(seed);
            centers[target] = mean(&groups[target])?;
        } else {
            groups.push(seed);
            centers.push(center);
        }
    }
    Ok(groups)
}

/// 弱表格区间的字号、上下边界、完整边框与源成员；最终认领结果不跨调用缓存。
pub type ProseRuleDraft = (f64, f64, f64, [f64; 4], Vec<i64>);
/// 正文的框、源编号和 Python 已计算的文字/语义准入特征。
pub type ProseRuleLine = ([f64; 4], i64, bool);

/// 相同网格和字号只收集一次结构与正文几何，原阈值与严格边界比较保持不变。
pub fn prose_rule_conflicts(
    drafts: &[ProseRuleDraft],
    grids: &[[f64; 4]],
    rules: &[([f64; 4], bool)],
    lines: &[ProseRuleLine],
) -> Vec<bool> {
    let mut geometry = std::collections::HashMap::<(usize, u64), Option<Vec<(i64, f64)>>>::new();
    drafts
        .iter()
        .map(|draft| {
            let (em, first, last, boundary, members) = draft;
            let members: std::collections::HashSet<_> = members.iter().copied().collect();
            for (grid_index, core) in grids.iter().enumerate() {
                let width = core[2] - core[0];
                let height = core[3] - core[1];
                if width < 8.0 * em
                    || height < 4.0 * em
                    || core[1] - first < 5.0 * em
                    || !(core[1] - em <= *last && *last <= core[3] + em)
                    || overlap_ratio(boundary, core, 0) < 0.9
                {
                    continue;
                }
                let evidence = geometry
                    .entry((grid_index, em.to_bits()))
                    .or_insert_with(|| {
                        let mut verticals: Vec<_> = rules
                            .iter()
                            .filter(|(b, vertical)| {
                                let x = (b[0] + b[2]) / 2.0;
                                *vertical
                                    && core[0] - 0.5 * em <= x
                                    && x <= core[2] + 0.5 * em
                                    && b[1] <= core[1] + 0.5 * em
                                    && b[3] >= core[3] - 0.5 * em
                            })
                            .map(|(b, _)| (b[0] + b[2]) / 2.0)
                            .collect();
                        verticals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let mut tracks = Vec::new();
                        for x in verticals {
                            if tracks.last().is_none_or(|&last| x - last > 0.3 * em) {
                                tracks.push(x);
                            }
                        }
                        if tracks.len() < 3
                            || ![core[1], core[3]].into_iter().all(|y| {
                                rules.iter().any(|(b, vertical)| {
                                    !*vertical
                                        && overlap_ratio(b, core, 0) >= 0.9
                                        && ((b[1] + b[3]) / 2.0 - y).abs() <= 0.5 * em
                                })
                            })
                        {
                            return None;
                        }
                        Some(
                            lines
                                .iter()
                                .filter(|(b, _, eligible)| {
                                    *eligible
                                        && b[2] - b[0] >= 0.65 * width
                                        && overlap_ratio(b, core, 0) >= 0.9
                                })
                                .map(|(b, source, _)| (*source, (b[1] + b[3]) / 2.0))
                                .collect(),
                        )
                    });
                let Some(prose) = evidence else {
                    continue;
                };
                let mut centers: Vec<_> = prose
                    .iter()
                    .filter(|(source, y)| {
                        members.contains(source) && first + 0.5 * em < *y && *y < core[1] - em
                    })
                    .map(|(_, y)| *y)
                    .collect();
                centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mut rows = Vec::new();
                for y in centers {
                    if rows.last().is_none_or(|&last| y - last > 0.6 * em) {
                        rows.push(y);
                    }
                }
                if rows.len() >= 3 {
                    return true;
                }
            }
            false
        })
        .collect()
}

/// 与 Python 一致，以较短轴长作分母；退化或倒置轴的覆盖率为零。
fn overlap_ratio(a: &[f64; 4], b: &[f64; 4], axis: usize) -> f64 {
    let overlap = (a[axis + 2].min(b[axis + 2]) - a[axis].max(b[axis])).max(0.0);
    let shorter = (a[axis + 2] - a[axis]).min(b[axis + 2] - b[axis]);
    if shorter > 0.0 {
        overlap / shorter
    } else {
        0.0
    }
}

/// 先检查完整网格集合的必要几何条件，尚不读取正文成员；没有匹配网格的候选不可能冲突。
pub fn prose_rule_eligible(drafts: &[ProseRuleDraft], grids: &[[f64; 4]]) -> Vec<bool> {
    drafts
        .iter()
        .map(|(em, first, last, boundary, _)| {
            grids.iter().any(|core| {
                core[2] - core[0] >= 8.0 * em
                    && core[3] - core[1] >= 4.0 * em
                    && core[1] - first >= 5.0 * em
                    && core[1] - em <= *last
                    && *last <= core[3] + em
                    && overlap_ratio(boundary, core, 0) >= 0.9
            })
        })
        .collect()
}

/// 栏推断的完整框、当前字体尺度、锚点资格及嵌套栏回退资格。
pub type LaneGeometry = ([f64; 4], f64, bool, bool);

/// 左边缘已排序，逐步中位数只需读取完整当前组的中间元素，保留原贪心聚类顺序。
pub fn supported_lane_intervals(
    lines: &[LaneGeometry],
    width: f64,
    median: f64,
) -> Vec<(f64, f64, usize)> {
    let tolerance = 3.0f64.max(0.75 * median);
    let mut regular: Vec<_> = (0..lines.len())
        .filter(|&i| {
            lines[i].2 && lines[i].0[2] - lines[i].0[0] >= (4.0 * lines[i].1).max(0.15 * width)
        })
        .collect();
    regular.sort_by(|&a, &b| lines[a].0[0].partial_cmp(&lines[b].0[0]).unwrap());
    let mut clusters = Vec::<Vec<usize>>::new();
    for &index in &regular {
        if let Some(cluster) = clusters.last_mut() {
            let n = cluster.len();
            let center = if n % 2 == 1 {
                lines[cluster[n / 2]].0[0]
            } else {
                (lines[cluster[n / 2 - 1]].0[0] + lines[cluster[n / 2]].0[0]) / 2.0
            };
            if (lines[index].0[0] - center).abs() <= tolerance {
                cluster.push(index);
                continue;
            }
        }
        clusters.push(vec![index]);
    }
    let mut supported = Vec::new();
    for cluster in clusters.iter().filter(|c| c.len() >= 3) {
        let left = crate::median(cluster.iter().map(|&i| lines[i].0[0]).collect());
        let right = crate::median(cluster.iter().map(|&i| lines[i].0[2]).collect());
        supported.push((left, right, cluster.len()));
    }
    supported.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut filtered: Vec<(f64, f64, usize)> = Vec::new();
    for interval in supported {
        if let Some(previous) = filtered.last_mut() {
            if interval.0 - previous.1 >= 6.0f64.max(0.75 * median) {
                filtered.push(interval);
            } else if interval.2 > previous.2 {
                *previous = interval;
            }
        } else {
            filtered.push(interval);
        }
    }
    if filtered.is_empty() && !lines.is_empty() {
        let source: Vec<_> = if regular.is_empty() {
            (0..lines.len()).collect()
        } else {
            regular
        };
        let left = source
            .iter()
            .map(|&i| lines[i].0[0])
            .reduce(|a, b| if b < a { b } else { a })
            .unwrap();
        let right = source
            .iter()
            .map(|&i| lines[i].0[2])
            .reduce(|a, b| if b > a { b } else { a })
            .unwrap();
        filtered.push((left, right, source.len()));
    }
    filtered
}

/// 按原栏顺序保留首个最佳覆盖；邻栏、同高和跨栏边界沿用原严格比较。
pub fn lane_assignments(
    lines: &[LaneGeometry],
    lanes: &[(f64, f64)],
    tolerance: f64,
    nested: Option<(f64, f64)>,
) -> Vec<Option<usize>> {
    let mut ordered: Vec<_> = (0..lanes.len()).collect();
    ordered.sort_by(|&a, &b| lanes[a].0.partial_cmp(&lanes[b].0).unwrap());
    lines
        .iter()
        .map(|line| {
            let bbox = line.0;
            if nested.is_some_and(|(top, bottom)| {
                line.3
                    || !((top <= (bbox[1] + bbox[3]) / 2.0)
                        && ((bbox[1] + bbox[3]) / 2.0 <= bottom))
            }) {
                return Some(lanes.len());
            }
            let width = 0.1f64.max(bbox[2] - bbox[0]);
            let mut best = 0;
            let mut first = -1.0;
            let mut second = -1.0;
            for (i, &(left, right)) in lanes.iter().enumerate() {
                let coverage = (bbox[2].min(right) - bbox[0].max(left)).max(0.0) / width;
                if coverage > first {
                    second = first;
                    first = coverage;
                    best = i;
                } else if coverage > second {
                    second = coverage;
                }
            }
            let position = ordered.iter().position(|&i| i == best).unwrap();
            let fit = !(position > 0 && bbox[0] < lanes[ordered[position - 1]].1 - tolerance
                || position + 1 < ordered.len()
                    && bbox[2] > lanes[ordered[position + 1]].0 - 0.25 * tolerance)
                && lanes[best].0 - tolerance <= (bbox[0] + bbox[2]) / 2.0
                && (bbox[0] + bbox[2]) / 2.0
                    <= lanes[best].1 + tolerance.max(bbox[2] - lanes[best].1);
            if second >= 0.2 && !fit {
                None
            } else if fit
                || if nested.is_some() {
                    first >= 0.5
                } else {
                    lanes.len() == 1
                }
            {
                Some(best)
            } else {
                None
            }
        })
        .collect()
}

/// 完整局部分式几何按原规则和字符顺序扫描，阈值、重叠、上下侧比较均保留原运算顺序。
pub fn fraction_members(chars: &[([f64; 4], bool)], rules: &[[f64; 4]], scale: f64) -> Vec<usize> {
    let mut members = vec![false; chars.len()];
    for rule in rules {
        let width = rule[2] - rule[0];
        let height = rule[3] - rule[1];
        if width < 2.0f64.max(0.45 * scale)
            || width > 12.0 * scale
            || height > 1.25f64.max(0.25 * scale)
            || width > 8.0 * scale
        {
            continue;
        }
        let y = (rule[1] + rule[3]) / 2.0;
        let mut above = Vec::new();
        let mut below = Vec::new();
        for (i, &(b, alnum)) in chars.iter().enumerate() {
            if !alnum || ((b[1] + b[3]) / 2.0 - y).abs() > 2.25 * scale {
                continue;
            }
            let overlap = 0.0f64.max(b[2].min(rule[2]) - b[0].max(rule[0]));
            if !(overlap > 0.0
                || rule[0] - 0.25 * scale <= (b[0] + b[2]) / 2.0
                    && (b[0] + b[2]) / 2.0 <= rule[2] + 0.25 * scale)
            {
                continue;
            }
            if b[3] <= y + 0.2 * scale && y - b[3] <= 1.75 * scale {
                above.push(i);
            }
            if b[1] >= y - 0.2 * scale && b[1] - y <= 1.75 * scale {
                below.push(i);
            }
        }
        if !above.is_empty() && !below.is_empty() {
            for i in above.into_iter().chain(below) {
                members[i] = true;
            }
        }
    }
    members
        .into_iter()
        .enumerate()
        .filter_map(|(i, v)| v.then_some(i))
        .collect()
}
