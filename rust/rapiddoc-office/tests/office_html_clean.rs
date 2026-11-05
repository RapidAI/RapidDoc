use rapiddoc_office::document_output::html;
#[path = "support/original_html_clean.rs"]
mod original;
#[test]
fn html_cleaner_preserves_capture_reference_on_long_and_malformed_attributes() {
    let names = [
        "img", "IMG", "a", "table", "td", "中文", "a\u{301}", "_x", "p",
    ];
    let keys = [
        "src",
        "alt",
        "width",
        "height",
        "colspan",
        "rowspan",
        "href",
        "data-key",
        "SRC",
        "中文",
        "a\u{200c}",
    ];
    let values = [
        "",
        "abc",
        "a\"tail",
        "a'b",
        "<x>",
        "http://x.test/?a=1&b=2",
        " javascript:evil",
        "mailto:a@b",
        "&amp;",
        "A\nB",
        "'unclosed",
        "\"unclosed",
    ];
    for name in names {
        for key in keys {
            for value in values {
                for quote in ["", "\"", "'"] {
                    let s = format!(
                        "before <{name} {key}={quote}{value}{quote} alt='Z' /></{name}> after"
                    );
                    assert_eq!(
                        html::clean(&s).unwrap(),
                        original::original_clean(&s).unwrap(),
                        "{s:?}"
                    );
                }
            }
        }
    }
    for space in [
        " ", "\n", "\t", "\r", "\u{85}", "\u{a0}", "\u{2003}", "\u{2028}",
    ] {
        let s = format!("<img{space}src{space}={space}\"data:image/png;base64,ABC==\"{space}/>");
        assert_eq!(
            html::clean(&s).unwrap(),
            original::original_clean(&s).unwrap()
        );
    }
    let payload = "ABcd019+/=".repeat(50_000);
    for s in [
        format!(
            "<table><tr><td><img src=\"data:image/jpeg;base64,{payload}\" /></td></tr></table>"
        ),
        format!("<IMG SRC='data:image/png;base64,{payload}' ALT=unquoted/>"),
    ] {
        assert_eq!(
            html::clean(&s).unwrap(),
            original::original_clean(&s).unwrap()
        );
    }
}
