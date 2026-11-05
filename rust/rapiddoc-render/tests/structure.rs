use rapiddoc_render::structure::html;

fn tokens(inner: &[&str]) -> Vec<String> {
    [
        vec!["<html>", "<body>", "<table>"],
        inner.to_vec(),
        vec!["</table>", "</body>", "</html>"],
    ]
    .concat()
    .into_iter()
    .map(str::to_owned)
    .collect()
}

#[test]
fn merged_cells_and_escaped_plain_text() {
    let value = html(
        &tokens(&[
            "<tr>",
            "<td",
            " colspan=\"2\"",
            ">",
            "</td>",
            "</tr>",
            "<tr>",
            "<td></td>",
            "<td></td>",
            "</tr>",
        ]),
        &["<b>&\"'".into(), "中文  A".into(), "line\nnext".into()],
    )
    .unwrap();
    assert_eq!(value,"<table>\n<tr><td colspan=\"2\">&lt;b&gt;&amp;&quot;&#39;</td></tr>\n<tr><td>中文  A</td><td>line<br/>next</td></tr>\n</table>");
    assert!(html(
        &tokens(&[
            "<tr>",
            "<td",
            " rowspan=\"2\"",
            ">",
            "</td>",
            "</tr>",
            "<tr>",
            "</tr>"
        ]),
        &["text".into()]
    )
    .is_ok());
}

#[test]
fn malformed_or_untrusted_structures_fail() {
    for inner in [
        vec!["<tr>", "<td", " onclick=\"evil\"", ">", "</td>", "</tr>"],
        vec!["<tr>", "<td"],
        vec![
            "<tr>",
            "<td></td>",
            "<td></td>",
            "</tr>",
            "<tr>",
            "<td></td>",
            "</tr>",
        ],
        vec!["<script>"],
        vec!["<tr>", "<td", " rowspan=\"0\"", ">", "</td>", "</tr>"],
        vec![
            "<tr>",
            "<td",
            " colspan=\"2\"",
            " colspan=\"3\"",
            ">",
            "</td>",
            "</tr>",
        ],
    ] {
        assert!(
            html(&tokens(&inner), &vec!["x".into(); 3]).is_err(),
            "{inner:?}"
        );
    }
    assert!(html(&tokens(&["<tr>", "<td></td>", "</tr>"]), &[]).is_err());
}
