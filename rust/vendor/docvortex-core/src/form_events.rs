//! 扫描普通内容流的图形状态事件，特殊语法完整交回 pypdf。

/// 事件种类和原始操作数字节区间，避免为无关 PDF 对象分配内存。
pub type FormEvent = (u8, Vec<(usize, usize)>);
/// 状态程序持有数值，名称区间指向调用期间的不可变字节。
pub type FormProgramEvent = (u8, Vec<f64>, Option<(usize, usize)>);

/// PDF 白空白字符，保持与内容流读取器相同的分词边界。
fn white(b: u8) -> bool {
    matches!(b, 0 | 9 | 10 | 12 | 13 | 32)
}
/// 名称、数字和操作符的终止字符。
fn delimiter(b: u8) -> bool {
    white(b) || b"()<>[]{}/%".contains(&b)
}

/// 容器中的空白和注释不形成对象，结束符留给调用者处理。
fn skip_space(data: &[u8], cursor: &mut usize) {
    while *cursor < data.len() {
        if white(data[*cursor]) {
            *cursor += 1;
        } else if data[*cursor] == b'%' {
            while *cursor < data.len() && !matches!(data[*cursor], 10 | 13) {
                *cursor += 1;
            }
        } else {
            break;
        }
    }
}

/// 一个完整 PDF 操作数；参考内容流解析器不支持容器间接引用，遇到时整流回退。
fn object_token(data: &[u8], cursor: &mut usize, depth: usize) -> Option<(u8, usize, usize)> {
    let value = token(data, cursor, depth)?;
    if value.0 == 1 && data[value.1..value.2].iter().all(u8::is_ascii_digit) {
        let mut probe = *cursor;
        if let Some((1, a, b)) = token(data, &mut probe, depth) {
            if data[a..b].iter().all(u8::is_ascii_digit) {
                if let Some((3, start, end)) = token(data, &mut probe, depth) {
                    if &data[start..end] == b"R" {
                        return None;
                    }
                }
            }
        }
    }
    Some(value)
}

/// 返回普通操作数或操作符的字节区间；字符串和数组整体跳过，避免正文伪造 Do。
fn token(data: &[u8], cursor: &mut usize, depth: usize) -> Option<(u8, usize, usize)> {
    if depth > 64 {
        return None;
    }
    while *cursor < data.len() {
        if white(data[*cursor]) {
            *cursor += 1;
        } else if data[*cursor] == b'%' {
            while *cursor < data.len() && !matches!(data[*cursor], 10 | 13) {
                *cursor += 1;
            }
        } else {
            break;
        }
    }
    let start = *cursor;
    let first = *data.get(start)?;
    *cursor += 1;
    let kind = match first {
        b'(' => {
            let mut nesting = 1;
            while nesting > 0 {
                let b = *data.get(*cursor)?;
                *cursor += 1;
                match b {
                    b'\\' => {
                        data.get(*cursor)?;
                        *cursor += 1;
                    }
                    b'(' => nesting += 1,
                    b')' => nesting -= 1,
                    _ => {}
                }
            }
            0
        }
        b'[' => {
            loop {
                let (kind, _, _) = object_token(data, cursor, depth + 1)?;
                if kind == 4 {
                    break;
                }
                if kind == 3 {
                    return None;
                }
            }
            0
        }
        b']' => 4,
        b'<' => {
            if data.get(*cursor) == Some(&b'<') {
                *cursor += 1;
                loop {
                    skip_space(data, cursor);
                    if data.get(*cursor..*cursor + 2) == Some(b">>") {
                        *cursor += 2;
                        break;
                    }
                    let (kind, _, _) = token(data, cursor, depth + 1)?;
                    if kind != 2 {
                        return None;
                    }
                    let (kind, _, _) = object_token(data, cursor, depth + 1)?;
                    if kind >= 3 {
                        return None;
                    }
                }
                return Some((0, start, *cursor));
            }
            loop {
                let b = *data.get(*cursor)?;
                *cursor += 1;
                if b == b'>' {
                    break;
                }
                if !white(b) && !b.is_ascii_hexdigit() {
                    return None;
                }
            }
            0
        }
        b'/' => {
            while *cursor < data.len() && !delimiter(data[*cursor]) {
                *cursor += 1;
            }
            // 即便名称属于无关操作，也必须保留参考解析器对损坏转义的整流失败行为。
            let mut index = start + 1;
            while index < *cursor {
                if data[index] == b'#' {
                    if index + 2 >= *cursor
                        || !data[index + 1].is_ascii_hexdigit()
                        || !data[index + 2].is_ascii_hexdigit()
                    {
                        return None;
                    }
                    index += 3;
                } else {
                    index += 1;
                }
            }
            2
        }
        b'>' | b')' | b'{' | b'}' => return None,
        _ => {
            while *cursor < data.len() && !delimiter(data[*cursor]) {
                *cursor += 1;
            }
            let word = &data[start..*cursor];
            if [b"true".as_slice(), b"false", b"null"].contains(&word) {
                return Some((0, start, *cursor));
            }
            if first.is_ascii_digit() || matches!(first, b'+' | b'-' | b'.') {
                // pypdf 分块读取在第四个 16 字节块前触发长度限制，长数字保守交回参考解析器。
                if word.len() >= 48
                    || !word
                        .iter()
                        .all(|b| b.is_ascii_digit() || matches!(b, b'+' | b'-' | b'.'))
                    || std::str::from_utf8(word)
                        .ok()?
                        .parse::<f64>()
                        .ok()
                        .is_none_or(|v| !v.is_finite())
                {
                    return None;
                }
                1
            } else if first.is_ascii_alphabetic() || matches!(first, b'\'' | b'"') {
                3
            } else {
                return None;
            }
        }
    };
    Some((kind, start, *cursor))
}

/// 只输出 q/Q/cm/Do，操作数仍由 pypdf 在小边界转换以保留原始数值和名称行为。
pub fn form_state_events(data: &[u8]) -> Option<Vec<FormEvent>> {
    let mut cursor = 0;
    let mut operands = Vec::new();
    let mut events = Vec::new();
    while cursor < data.len() {
        // 流尾空白和注释不会形成额外操作数。
        if white(data[cursor]) {
            cursor += 1;
            continue;
        }
        if data[cursor] == b'%' {
            while cursor < data.len() && !matches!(data[cursor], 10 | 13) {
                cursor += 1;
            }
            continue;
        }
        let (kind, start, end) = token(data, &mut cursor, 0)?;
        if kind == 4 {
            return None;
        }
        if kind != 3 {
            operands.push((kind, start, end));
            continue;
        }
        let op = &data[start..end];
        if op == b"BI" {
            return None;
        }
        let event = match op {
            b"q" => Some(0),
            b"Q" => Some(1),
            b"cm" if operands.len() == 6 && operands.iter().all(|o| o.0 == 1) => Some(2),
            b"Do" if operands.len() == 1 && operands[0].0 == 2 => Some(3),
            b"cm" | b"Do" => return None,
            _ => None,
        };
        if let Some(kind) = event {
            events.push((
                kind,
                if kind < 2 {
                    vec![]
                } else {
                    operands.iter().map(|o| (o.1, o.2)).collect()
                },
            ));
        }
        operands.clear();
    }
    Some(events)
}

/// 数字操作数直接生成自有浮点值；整数负零按 NumberObject 转换成正零，名称仍由 Python 解码。
pub fn form_state_program(data: &[u8]) -> Option<Vec<FormProgramEvent>> {
    form_state_events(data)?
        .into_iter()
        .map(|(kind, intervals)| {
            if kind == 2 {
                let values = intervals
                    .into_iter()
                    .map(|(a, b)| {
                        let word = &data[a..b];
                        let value = std::str::from_utf8(word).ok()?.parse::<f64>().ok()?;
                        Some(if value == 0.0 && !word.contains(&b'.') {
                            0.0
                        } else {
                            value
                        })
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some((kind, values, None))
            } else {
                Some((kind, Vec::new(), intervals.first().copied()))
            }
        })
        .collect()
}
