import pytest

from rapid_doc._vendor.docvortex_core.foundation._text import merge_text_line_contents, resolve_text_line_boundary


@pytest.mark.parametrize(
    ("previous_content", "next_content", "expected"),
    [
        pytest.param(
            "See https://example.test/current-research",
            "/image-processing/",
            "See https://example.test/current-research/image-processing/",
            id="url-path-continuation",
        ),
        pytest.param(
            "Download from ftp://example.test/archive",
            "?format=zip",
            "Download from ftp://example.test/archive?format=zip",
            id="url-query-continuation",
        ),
        pytest.param(
            "Visit www.example.test/docs",
            "#install",
            "Visit www.example.test/docs#install",
            id="url-fragment-continuation",
        ),
        pytest.param(
            "Use https://example.test/search?key=one",
            "&page=2",
            "Use https://example.test/search?key=one&page=2",
            id="url-query-parameter-continuation",
        ),
        pytest.param(
            "Use https://example.test/search?key",
            "=value",
            "Use https://example.test/search?key=value",
            id="url-query-value-continuation",
        ),
        pytest.param("input", "/ output", "input / output", id="ordinary-slash-text"),
        pytest.param(
            "https://blog.example.test/first",
            "https://blog.example.test/second",
            "https://blog.example.test/first https://blog.example.test/second",
            id="independent-urls",
        ),
        pytest.param(
            "DOI https",
            "://doi.org/10.37921/example",
            "DOI https://doi.org/10.37921/example",
            id="url-scheme-continuation",
        ),
        pytest.param(
            "See https://doi.o",
            "rg/10.3322/example",
            "See https://doi.org/10.3322/example",
            id="url-host-continuation",
        ),
        pytest.param(
            "See https://doi.org/10.101",
            "6/example",
            "See https://doi.org/10.1016/example",
            id="url-numeric-path-continuation",
        ),
        pytest.param(
            "See https://doi.org/10.1038/example-019",
            "-0178-8",
            "See https://doi.org/10.1038/example-019-0178-8",
            id="url-hyphen-continuation",
        ),
        pytest.param(
            "Download from https://download.docker.com/linux/",
            "ubuntu/dists/",
            "Download from https://download.docker.com/linux/ubuntu/dists/",
            id="url-alpha-path-continuation",
        ),
        pytest.param("inter-", "national", "international", id="western-hyphen"),
        pytest.param("first line", "second line", "first line second line", id="western-space"),
        pytest.param("中文", "继续", "中文继续", id="cjk-direct-join"),
    ],
)
def test_resolve_text_line_boundary_keeps_conservative_joining_rules(
    previous_content: str,
    next_content: str,
    expected: str,
) -> None:
    """验证 URL、普通西文、断词和 CJK 的物理行边界规则互不干扰。"""
    processed_previous, separator = resolve_text_line_boundary(
        previous_content,
        next_content=next_content,
    )

    assert f"{processed_previous}{separator}{next_content}" == expected


def test_merge_text_line_contents_keeps_accumulated_url_context() -> None:
    """验证三行 URL 使用完整累计前缀，而普通标题和独立 URL 保留自然空格。"""

    assert merge_text_line_contents(
        [
            "Code at https://github.",
            "com/google-research/tapas/blob/master/",
            "TABLEFORMER.md",
        ],
    ) == ("Code at https://github.com/google-research/tapas/blob/master/TABLEFORMER.md")
    assert (
        merge_text_line_contents(
            [
                "ETC: Encoding long and structured inputs",
                "in transformers",
            ],
        )
        == "ETC: Encoding long and structured inputs in transformers"
    )
    assert (
        merge_text_line_contents(
            [
                "https://example.test/first",
                "https://example.test/second",
            ],
        )
        == "https://example.test/first https://example.test/second"
    )
