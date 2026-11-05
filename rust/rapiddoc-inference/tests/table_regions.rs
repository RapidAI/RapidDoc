use rapiddoc_core::ErrorCode;
use rapiddoc_inference::{table_draw, table_regions};
#[test]
fn closed_grid_and_no_lines() {
    let mut lines = Vec::new();
    for y in [20., 80., 140.] {
        lines.push([20., y, 220., y]);
    }
    for x in [20., 120., 220.] {
        lines.push([x, 20., x, 140.]);
    }
    let cells = table_regions::from_lines(&lines, 240, 160, true).unwrap();
    assert_eq!(cells.polygons.len(), 4);
    assert_eq!(
        cells.topology.unwrap().logic_points,
        [[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]]
    );
    let empty = table_regions::from_lines(&[], 240, 160, true).unwrap();
    assert!(empty.polygons.is_empty() && empty.topology.is_none());
}
#[test]
fn drawing_clipping_and_resource_limits() {
    assert!(table_draw::draw(&[[-10., -10., -5., -5.]], 32, 32)
        .unwrap()
        .iter()
        .all(|v| *v == 0));
    assert_eq!(
        table_draw::draw(&vec![[0., 0., 1., 1.]; 2001], 32, 32)
            .err()
            .unwrap()
            .code,
        ErrorCode::ResourceLimit
    );
    assert!(table_draw::draw(&[[0., 0., f32::NAN, 1.]], 32, 32).is_err());
    assert!(table_regions::rotate(&[0], 1, 1, f64::NAN).is_err());
    assert_eq!(table_regions::rotate(&[123], 1, 1, 45.).unwrap(), [123]);
}
