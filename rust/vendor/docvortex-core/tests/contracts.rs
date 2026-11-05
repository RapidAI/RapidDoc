//! 用独立几何结果约束内核边界，Python 随机与真实 PDF 差分另行执行。

use docvortex_core::{dedup, scripts, tables};

/// 无内部竖隔断的第一行应构成横跨两列的矩形单元格。
#[test]
fn merged_cell_is_rectangular() {
    let parents = tables::grid_parents(4, vec![(0, 1)]);
    let (_, specs) = tables::component_specs(parents, 2, 2);
    assert_eq!(specs, Some(vec![(0, 0, 1, 2), (1, 0, 1, 1), (1, 1, 1, 1)]));
}

/// L 形连通分量不能变成吞并空格的矩形表格。
#[test]
fn l_shaped_component_is_rejected() {
    let parents = tables::grid_parents(4, vec![(0, 1), (0, 2)]);
    assert_eq!(tables::component_specs(parents, 2, 2).1, None);
}

/// 反向线段与相接线段共同覆盖轨道，零长目标保持零覆盖。
#[test]
fn reversed_and_touching_intervals() {
    let values = tables::coverage_batch(
        vec![(0, 1.0, 5.0, 0.0), (0, 1.0, 5.0, 10.0)],
        vec![
            (0, vec![1.0], 0.0, 10.0, 0.0),
            (0, vec![1.0], 2.0, 2.0, 0.0),
        ],
    );
    assert_eq!(values, Some(vec![1.0, 0.0]));
}

/// 叠层达到保护上限后，后续字形不得再产生删除候选。
#[test]
fn paint_bucket_limit_preserves_later_glyphs() {
    let records = (0..80)
        .map(|i| (Some([0.0, 0.0, 5.0, 10.0]), Some([0.0, 9.0]), 0, i, true))
        .collect();
    let (pairs, offsets) = dedup::paint_pairs(records).unwrap();
    assert_eq!(pairs.len(), 64 * 63 / 2);
    assert!(pairs.iter().all(|p| p.1 < 64));
    assert!(offsets.is_empty());
}

/// 两个稳定正文字母后的小号上移数字应成为上标。
#[test]
fn raised_digit_follows_body_baseline() {
    let records = vec![
        (
            [0.0, 0.0, 5.0, 10.0],
            Some([0.0, 1.0, 5.0, 9.0]),
            Some([0.0, 9.0]),
            4 | 16 | 256,
            0,
        ),
        (
            [5.0, 0.0, 10.0, 10.0],
            Some([5.0, 1.0, 10.0, 9.0]),
            Some([5.0, 9.0]),
            4 | 16 | 256,
            0,
        ),
        (
            [10.0, -2.0, 13.0, 4.0],
            Some([10.0, -1.0, 13.0, 3.0]),
            Some([10.0, 3.0]),
            4 | 8 | 16 | 256,
            0,
        ),
    ];
    assert_eq!(scripts::classify(records), Some(vec![0, 0, 1]));
}

/// 普通框不相邻时，原始行框仍可保留连续字符路径。
#[test]
fn baseline_keeps_original_source_branch() {
    let index = docvortex_core::spatial::BaselineGeometry::new(
        vec![(0.0, 12.5), (0.0, 12.5)],
        vec![0, 0],
        vec![[0.0, 0.0, 1.0, 1.0], [50.0, 0.0, 51.0, 1.0]],
        vec![1.0, 1.0],
        vec![Some([0.0, 0.0, 10.0, 10.0]), Some([10.0, 0.0, 20.0, 10.0])],
    )
    .unwrap();
    assert_eq!(index.rows(0, 64, 8192), vec![vec![1], vec![]]);
}

/// 同分竞争必须选首个小行与首个主体，不随排序实现改变。
#[test]
fn inline_ties_are_stable() {
    let a = ([10.0, 0.0, 12.0, 4.0], 4.0, 4.0, 1, false, 0, 0);
    let b = ([0.0, 2.0, 10.0, 12.0], 10.0, 10.0, 1, false, 1, 0);
    assert_eq!(
        docvortex_core::inline_pairs::matches(vec![a, b, a, b]),
        Some(vec![(0, 1, false, false)])
    );
}

/// 聚合平局选择首坐标来源，重复来源不改变输出成员顺序。
#[test]
fn annotation_union_preserves_first_coordinate() {
    let index = docvortex_core::annotation_geometry::AnnotationGeometry::new(vec![
        (8, [-0.0, 0.0, 1.0, 1.0], [-0.0, 0.0, 1.0, 1.0]),
        (8, [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 2.0, 1.0]),
        (3, [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 2.0, 1.0]),
    ])
    .unwrap();
    let (lines, bbox) = index.aggregate(vec![0, 1, 2], vec![], None).unwrap();
    assert_eq!(lines, vec![(8, [0, 0, 1, 0], 2), (3, [2, 2, 2, 2], 1)]);
    assert_eq!(bbox, Some([0, 0, 1, 0]));
    assert!(index.aggregate(vec![3], vec![], None).is_none());
}
