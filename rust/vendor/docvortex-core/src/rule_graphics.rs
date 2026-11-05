//! 新增图形规则的批量数值内核，不修改候选集合或判定阈值。

/// 横线对及其完整原行成员索引，顺序与 Python 嵌套循环一致。
pub type CodeRuleWindow = (usize, usize, Vec<usize>);

/// 每个区域返回全部满足原占首框面积比例的成员，重复框和原成员顺序均保留。
pub fn overlap_member_groups(
    members: &[[f64; 4]],
    regions: &[[f64; 4]],
    threshold: f64,
) -> Vec<Vec<usize>> {
    regions
        .iter()
        .map(|&region| {
            members
                .iter()
                .enumerate()
                .filter(|(_, bbox)| code_overlap(**bbox, region, false) >= threshold)
                .map(|(i, _)| i)
                .collect()
        })
        .collect()
}

/// 按原交集及面积公式计算覆盖率；退化框的零值条件也保持一致。
fn code_overlap(first: [f64; 4], second: [f64; 4], smaller: bool) -> f64 {
    let width = 0.0_f64.max(first[2].min(second[2]) - first[0].max(second[0]));
    let height = 0.0_f64.max(first[3].min(second[3]) - first[1].max(second[1]));
    let area = if smaller {
        ((first[2] - first[0]) * (first[3] - first[1]))
            .min((second[2] - second[0]) * (second[3] - second[1]))
    } else {
        0.0_f64.max(first[2] - first[0]) * 0.0_f64.max(first[3] - first[1])
    };
    if area > 0.0 {
        width * height / area
    } else {
        0.0
    }
}

/// 全量扫描横线对及内部轨道，仅迁移数值筛选，文字结构判定和最终框仍在 Python。
pub fn code_rule_windows(
    horizontal: &[[f64; 4]],
    vertical: &[[f64; 4]],
    lines: &[[f64; 4]],
    excluded: &[[f64; 4]],
    em: f64,
    page_height: f64,
) -> Vec<CodeRuleWindow> {
    let tolerance = 2.0_f64.max(0.75 * em);
    let mut output = Vec::new();
    for (top_index, top) in horizontal.iter().enumerate() {
        for (bottom_index, bottom) in horizontal.iter().enumerate().skip(top_index + 1) {
            if (top[0] - bottom[0]).abs() > tolerance || (top[2] - bottom[2]).abs() > tolerance {
                continue;
            }
            let bbox = [
                top[0].min(bottom[0]),
                top[1].min(bottom[1]),
                top[2].max(bottom[2]),
                top[3].max(bottom[3]),
            ];
            let height = bbox[3] - bbox[1];
            if !(6.0 * em <= height && height <= 0.5 * page_height)
                || excluded
                    .iter()
                    .any(|&other| code_overlap(bbox, other, true) >= 0.5)
                || horizontal.iter().any(|&rule| {
                    top[3] + 0.5 * em < rule[1]
                        && rule[1] < bottom[1] - 0.5 * em
                        && code_overlap(rule, bbox, false) >= 0.8
                        && rule[2] - rule[0] >= 0.6 * (bbox[2] - bbox[0])
                })
                || vertical.iter().any(|rule| {
                    let center = (rule[0] + rule[2]) / 2.0;
                    bbox[0] + em < center
                        && center < bbox[2] - em
                        && 0.0_f64.max(rule[3].min(bbox[3]) - rule[1].max(bbox[1]))
                            / 0.0_f64.max(height)
                            >= 0.6
                })
            {
                continue;
            }
            let members = lines
                .iter()
                .enumerate()
                .filter(|(_, line)| {
                    let center_x = (line[0] + line[2]) / 2.0;
                    let center_y = (line[1] + line[3]) / 2.0;
                    bbox[0] - 0.5 * em <= center_x
                        && center_x <= bbox[2] + 0.5 * em
                        && top[3] <= center_y
                        && center_y <= bottom[1]
                })
                .map(|(i, _)| i)
                .collect();
            output.push((top_index, bottom_index, members));
        }
    }
    output
}

/// 规范字体、flags、有效字重、有效字符数与按首次出现排序的每字三份框样本。
pub type GlyphFontGroup = (
    String,
    i32,
    Option<f64>,
    usize,
    Vec<(String, Vec<[f64; 4]>)>,
);

/// 以已验证的新鲜快照分组 ASCII 字形，字号不参与字体键，重复成员和插入顺序不去重。
pub fn glyph_font_groups(
    data: &crate::text_snapshot::TextSnapshot,
    rows: &[Vec<usize>],
    bold: &[bool],
) -> Vec<GlyphFontGroup> {
    let mut groups: Vec<GlyphFontGroup> = Vec::new();
    let mut font_keys = std::collections::HashMap::new();
    let mut font_groups = vec![None; data.fonts.len()];
    let mut letters: Vec<[Option<usize>; 128]> = Vec::new();
    for &index in rows.iter().flatten() {
        let ch = &data.chars[index];
        if ch.text.len() != 1 || !ch.text.as_bytes()[0].is_ascii_alphabetic() {
            continue;
        }
        let font = &data.fonts[ch.font];
        if font.flags & (1 << 6) != 0 || bold[ch.font] {
            continue;
        }
        let Some(bbox) = ch.tight else {
            continue;
        };
        let group = if let Some(group) = font_groups[ch.font] {
            group
        } else {
            let bytes = font.name.as_bytes();
            let name = if bytes.len() >= 7
                && bytes[..6].iter().all(u8::is_ascii_uppercase)
                && bytes[6] == b'+'
            {
                font.name[7..].to_owned()
            } else {
                font.name.clone()
            };
            let weight = font.weight.max(0);
            let key = (name.clone(), font.flags, weight);
            let group = if let Some(&group) = font_keys.get(&key) {
                group
            } else {
                let group = groups.len();
                groups.push((
                    name,
                    font.flags,
                    (weight > 0).then_some(weight as f64),
                    0,
                    Vec::new(),
                ));
                letters.push([None; 128]);
                font_keys.insert(key, group);
                group
            };
            font_groups[ch.font] = Some(group);
            group
        };
        groups[group].3 += 1;
        let letter = ch.text.as_bytes()[0] as usize;
        let letter_group = if let Some(position) = letters[group][letter] {
            position
        } else {
            let position = groups[group].4.len();
            groups[group].4.push((ch.text.clone(), Vec::new()));
            letters[group][letter] = Some(position);
            position
        };
        let samples = &mut groups[group].4[letter_group].1;
        if samples.len() < 3 {
            samples.push(bbox);
        }
    }
    groups
}

/// 每个原路径只检查两个方向一次，单个矩形只贡献一次支持票。
pub fn compound_baselines(groups: &[Vec<[f64; 4]>], em: f64) -> Vec<(bool, bool)> {
    groups
        .iter()
        .map(|boxes| {
            let shared = |edge: usize| {
                boxes.len() <= 1
                    || boxes.iter().any(|seed| {
                        [seed[edge], seed[edge + 2]].iter().any(|baseline| {
                            boxes
                                .iter()
                                .filter(|b| {
                                    (b[edge] - baseline)
                                        .abs()
                                        .min((b[edge + 2] - baseline).abs())
                                        <= 0.35 * em
                                })
                                .count() as f64
                                >= 0.75 * boxes.len() as f64
                        })
                    })
            };
            (shared(0), shared(1))
        })
        .collect()
}

/// 数值刻度稳定分组并核验等差值与间距；每次追加仍采用原样本中位数作为锚点。
pub fn raster_axis_groups(lines: &[([f64; 4], f64, f64)]) -> Vec<(bool, Vec<usize>, f64)> {
    let mut output = Vec::new();
    for vertical in [true, false] {
        let coordinate = |i: usize| {
            if vertical {
                lines[i].0[2]
            } else {
                (lines[i].0[1] + lines[i].0[3]) / 2.0
            }
        };
        let position = |i: usize| {
            if vertical {
                (lines[i].0[1] + lines[i].0[3]) / 2.0
            } else {
                (lines[i].0[0] + lines[i].0[2]) / 2.0
            }
        };
        let mut order: Vec<_> = (0..lines.len()).collect();
        order.sort_by(|&a, &b| coordinate(a).partial_cmp(&coordinate(b)).unwrap());
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for i in order {
            let append = groups.last().is_some_and(|prior| {
                let anchor = crate::median(prior.iter().map(|&j| coordinate(j)).collect());
                let mut heights: Vec<_> = prior.iter().map(|&j| lines[j].1).collect();
                heights.push(lines[i].1);
                (coordinate(i) - anchor).abs() <= 0.4 * crate::median(heights)
            });
            if append {
                groups.last_mut().unwrap().push(i);
            } else {
                groups.push(vec![i]);
            }
        }
        for mut ticks in groups {
            if ticks.len() < 5 {
                continue;
            }
            ticks.sort_by(|&a, &b| position(a).partial_cmp(&position(b)).unwrap());
            let em = crate::median(ticks.iter().map(|&j| lines[j].1).collect());
            let steps: Vec<_> = ticks
                .windows(2)
                .map(|p| lines[p[1]].2 - lines[p[0]].2)
                .collect();
            let gaps: Vec<_> = ticks
                .windows(2)
                .map(|p| position(p[1]) - position(p[0]))
                .collect();
            if steps[0].abs() < 1e-8
                || steps
                    .iter()
                    .any(|s| (s - steps[0]).abs() > 0.01 * steps[0].abs())
                || gaps.iter().copied().reduce(f64::max).unwrap()
                    - gaps.iter().copied().reduce(f64::min).unwrap()
                    > 0.4 * em
                || gaps.iter().any(|&g| g < em)
                || ticks
                    .iter()
                    .any(|&j| lines[j].1 < 0.8 * em || lines[j].1 > 1.2 * em)
            {
                continue;
            }
            output.push((vertical, ticks, em));
        }
    }
    output
}

/// 新路径先放在合并组首位，其余成员按既有组顺序拼接，保留原规则遍历与平局行为。
pub fn isolated_path_groups(boxes: &[[f64; 4]], em: f64) -> Option<Vec<Vec<usize>>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, b) in boxes.iter().enumerate() {
        let mut retained = Vec::new();
        let mut merged = vec![i];
        for group in groups {
            let mut connected = false;
            for &j in &group {
                let a = boxes[j];
                let x = (b[0] - a[2]).max(a[0] - b[2]).max(0.0);
                let y = (b[1] - a[3]).max(a[1] - b[3]).max(0.0);
                let distance = x.hypot(y);
                let limit = 0.35 * em;
                // CPython 与平台 hypot 的末位舍入可能不同，临界输入整批使用参考实现。
                if (distance - limit).abs() <= 8.0 * f64::EPSILON * distance.abs().max(limit.abs())
                {
                    return None;
                }
                if distance <= limit {
                    connected = true;
                    break;
                }
            }
            if connected {
                merged.extend(group);
            } else {
                retained.push(group);
            }
        }
        retained.push(merged);
        groups = retained;
    }
    Some(groups)
}

/// 用绘制顺序和完全包含判断覆盖，返回地址本身以便 Python 保留可见性映射顺序。
pub fn overpainted_addresses(
    texts: &[(usize, usize, [f64; 4])],
    covers: &[(usize, [f64; 4])],
) -> Vec<usize> {
    texts
        .iter()
        .filter(|(order, _, b)| {
            covers.iter().any(|(later, c)| {
                later > order && c[0] <= b[0] && c[1] <= b[1] && c[2] >= b[2] && c[3] >= b[3]
            })
        })
        .map(|(_, a, _)| *a)
        .collect()
}

/// 按完整横线与成员框筛选贴邻表头，返回原成员顺序，不处理年份或最终成员认领。
pub fn year_header_groups(lines: &[[f64; 4]], rules: &[[f64; 4]], em: f64) -> Vec<Vec<usize>> {
    rules
        .iter()
        .map(|r| {
            lines
                .iter()
                .enumerate()
                .filter_map(|(i, b)| {
                    (r[0] - 0.3 * em <= b[0]
                        && b[2] <= r[2] + 0.3 * em
                        && 0.0 <= r[1] - b[3]
                        && r[1] - b[3] <= 0.8 * em)
                        .then_some(i)
                })
                .collect()
        })
        .collect()
}
