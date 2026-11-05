use rapiddoc_native::text_boundary::merge;
fn join(lines: &[&str]) -> String {
    merge(&lines.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
}
#[test]
fn physical_boundaries_preserve_scripts_compounds_ranges_and_urls() {
    assert_eq!(join(&["中文", "段落"]), "中文段落");
    assert_eq!(join(&["한국어", "문장"]), "한국어 문장");
    assert_eq!(join(&["inter-", "national"]), "international");
    assert_eq!(join(&["open-", "source"]), "open-source");
    assert_eq!(join(&["GLGE-", "difficult"]), "GLGE-difficult");
    assert_eq!(join(&["1–", "³"]), "1–³");
    assert_eq!(
        join(&["https://exa", "mple.test/", "file"]),
        "https://example.test/file"
    );
    assert_eq!(join(&["http", "s://"]), "http s://");
    assert_eq!(join(&["https://a/", "https://b/"]), "https://a/ https://b/");
    assert_eq!(join(&["(hello", "world)"]), "(hello world)");
    assert_eq!(join(&["hi</b>", ","]), "hi</b>,");
}
#[test]
fn blank_styles_combining_marks_and_python_spaces_keep_reference_semantics() {
    assert_eq!(join(&["\u{1c}hello\u{1c}", "", "world"]), "hello world");
    assert_eq!(join(&["word\u{301}", "next"]), "word\u{301} next");
    assert_eq!(join(&["中</ſup>", "文"]), "中</ſup>文");
    let broken = "<b".repeat(10000);
    assert_eq!(join(&[&broken, "A"]), format!("{broken} A"));
    assert!(merge(&vec!["x".into(); 1001]).is_err());
    assert!(merge(&vec!["x".repeat(1000); 1000]).is_err());
}
