use rapiddoc_core::ErrorCode;
use rapiddoc_inference::table_lines::{connect, Config};
#[test]
fn ordered_cross_extensions_and_work_limits() {
    let config = Config {
        more_rows: false,
        more_columns: false,
        ..Config::default()
    };
    let result = connect(
        vec![[10., 50., 90., 50.]],
        vec![[5., 10., 5., 90.], [95., 10., 95., 90.]],
        &config,
    )
    .unwrap();
    assert_eq!(result.rows, vec![[5., 50., 95., 50.]]);
    assert_eq!(
        connect(
            vec![[0., 0., 10., 0.]],
            vec![],
            &Config {
                max_work: 0,
                ..Config::default()
            }
        )
        .err()
        .unwrap()
        .code,
        ErrorCode::ResourceLimit
    );
}
