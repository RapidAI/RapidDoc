//! 文字投影只处理自有 Unicode 标量和源位置；分类与连字映射继续由 Python 提供。

/// 扫描原始字符串的 Unicode 索引，跳过首个匹配闭合标记内的圆括号公式。
pub fn visible_tokens(content: &str) -> Vec<(usize, char, bool)> {
    let chars: Vec<_> = content.chars().collect();
    let mut cursor = 0;
    let mut gap = false;
    let mut tokens = Vec::new();
    while cursor < chars.len() {
        if chars[cursor] == '\\' && chars.get(cursor + 1) == Some(&'(') {
            if let Some(end) = (cursor + 2..chars.len().saturating_sub(1))
                .find(|&i| chars[i] == '\\' && chars[i + 1] == ')')
            {
                cursor = end + 2;
                gap = true;
                continue;
            }
        }
        tokens.push((cursor, chars[cursor], gap));
        gap = false;
        cursor += 1;
    }
    tokens
}

/// 空白或零宽字符不消费公式间隙；连字展开后的第一字符保留该间隙。
pub fn project_tokens(
    tokens: &[(usize, char, bool)],
    forms: &std::collections::HashMap<char, String>,
) -> Vec<(char, usize, usize, bool)> {
    let mut pending = false;
    let mut output = Vec::new();
    for &(position, original, gap) in tokens {
        pending |= gap;
        let fragment = &forms[&original];
        for (i, value) in fragment.chars().enumerate() {
            output.push((value, position, position + 1, pending && i == 0));
        }
        if !fragment.is_empty() {
            pending = false;
        }
    }
    output
}
