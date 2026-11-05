use rapiddoc_render::wired::html;
fn p(a: f64, c: f64) -> [[f64; 2]; 4] {
    [[a, 0.], [a, 40.], [c, 40.], [c, 0.]]
}
#[test]
fn normal_empty_column_retained_and_abnormal_empty_edge_trimmed() {
    let points = [[0, 0, 0, 0], [0, 0, 1, 1]];
    let normal = html(
        &points,
        &[p(0., 100.), p(100., 200.)],
        &[vec!["A".into()], vec![]],
    )
    .unwrap();
    assert_eq!(normal.matches("<td ").count(), 2);
    let noise = html(
        &points,
        &[p(0., 10.), p(10., 110.)],
        &[vec![], vec!["<script>&".into()]],
    )
    .unwrap();
    assert_eq!(noise.matches("<td ").count(), 1);
    assert!(noise.contains("&lt;script&gt;&amp;") && !noise.contains("<script>"));
}
#[test]
fn overlapping_cells_never_overwrite_text() {
    assert!(html(
        &[[0, 0, 0, 0], [0, 0, 0, 0]],
        &[p(0., 100.), p(0., 100.)],
        &[vec!["A".into()], vec!["B".into()]]
    )
    .is_err());
}

#[test]
fn legacy_overlapping_cells_follow_python_last_write_ownership() {
    let points = [[0, 0, 0, 0], [0, 0, 0, 0]];
    let polygons = [p(0., 100.), p(0., 100.)];
    let texts = [vec!["A".into()], vec!["B".into()]];
    let result = rapiddoc_render::wired::legacy_html(&points, &polygons, &texts).unwrap();
    assert_eq!(
        result,
        "<html><body><table><tr><td rowspan=1 colspan=1>B</td></tr></table></body></html>"
    );
}
