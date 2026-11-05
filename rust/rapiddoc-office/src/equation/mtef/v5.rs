use super::*;
pub(super) fn parse(bytes: &[u8]) -> Result<Node> {
    if bytes.len() < 7 || bytes[0] != 5 || bytes[1] > 1 || bytes[2] > 1 || bytes[3] < 4 {
        return Err(bad());
    }
    let mut r = Reader {
        c: Cursor::new(bytes, 5),
        encodings: [
            (1, "MTCode".into()),
            (2, "Unknown".into()),
            (3, "Symbol".into()),
            (4, "MTExtra".into()),
        ]
        .into(),
        fonts: vec![None],
        styles: vec![None],
        preferences: vec![],
        colors: 0,
    };
    if r.c.cstring()?.is_empty() || r.c.u8()? & !1 != 0 {
        return Err(bad());
    }
    let nodes = r.list()?;
    if r.c.pos != bytes.len() {
        return Err(bad());
    }
    Ok(Node::Sequence(nodes))
}
type Style = Option<(usize, u8)>;
struct Reader<'a> {
    c: Cursor<'a>,
    encodings: BTreeMap<usize, String>,
    fonts: Vec<Option<usize>>,
    styles: Vec<Style>,
    preferences: Vec<Style>,
    colors: usize,
}
fn mtcode(code: u32) -> Result<Character> {
    if code < 32 || (0xd800..=0xdfff).contains(&code) {
        return Err(bad());
    }
    if (0xe000..=0xf8ff).contains(&code) {
        if let Some(value) = data().pua.get(&code) {
            return Ok(Character::Latex(value.clone()));
        }
        for (start, end, base, command) in [
            (0xf000, 0xf019, b'A', "mathfrak"),
            (0xf01a, 0xf033, b'a', "mathfrak"),
            (0xf080, 0xf099, b'A', "mathbb"),
            (0xf100, 0xf119, b'A', "mathcal"),
        ] {
            if (start..=end).contains(&code) {
                return Ok(Character::Latex(format!(
                    "\\{command}{{{}}}",
                    char::from(base + (code - start) as u8)
                )));
            }
        }
        return Err(bad());
    }
    Ok(Character::Code(code))
}
fn position(encoding: &str, position: u32) -> Result<Character> {
    match crate::ole::casefold(encoding.trim_matches(space)).as_str() {
        "mtcode" => mtcode(position),
        "symbol" => data()
            .symbol
            .get(&position)
            .copied()
            .map(Character::Code)
            .ok_or_else(bad),
        "mtextra" => data()
            .extra
            .get(&position)
            .copied()
            .map(Character::Code)
            .ok_or_else(bad),
        "winallbasiccodepages" | "windowsansi" | "windows-1252" | "cp1252" | "unicode" => {
            if position <= 255 {
                data()
                    .cp1252
                    .get(&position)
                    .copied()
                    .map(Character::Code)
                    .ok_or_else(bad)
            } else {
                mtcode(position)
            }
        }
        _ => Err(bad()),
    }
}
impl Reader<'_> {
    fn style(&self, face: i32) -> Style {
        if face < 0 {
            self.styles.get((-face) as usize).copied().flatten()
        } else if face > 0 {
            self.preferences.get(face as usize - 1).copied().flatten()
        } else {
            None
        }
    }
    fn encoding(&self, face: i32) -> &str {
        if let Some((font, _)) = self.style(face) {
            if let Some(encoding) = self
                .fonts
                .get(font)
                .copied()
                .flatten()
                .and_then(|index| self.encodings.get(&index))
            {
                return encoding;
            }
        }
        match face {
            4..=6 => "Symbol",
            11 => "MTExtra",
            _ => "WinAllBasicCodePages",
        }
    }
    fn ruler(&mut self) -> Result<()> {
        for _ in 0..self.c.u8()? {
            if self.c.u8()? > 4 {
                return Err(bad());
            }
            self.c.u16()?;
        }
        Ok(())
    }
    fn size(&mut self) -> Result<()> {
        let first = self.c.u8()?;
        if first == 100 {
            if self.c.u8()? > 7 {
                return Err(bad());
            }
            self.c.u16()?;
        } else if first == 101 {
            self.c.u16()?;
        } else {
            if first > 7 {
                return Err(bad());
            }
            self.c.u8()?;
        }
        Ok(())
    }
    fn dimension(&mut self) -> Result<()> {
        let count = self.c.u8()?;
        let mut low = false;
        let mut current = 0;
        let mut nibble = |c: &mut Cursor<'_>| -> Result<u8> {
            if !low {
                current = c.u8()?;
                low = true;
                Ok(current >> 4)
            } else {
                low = false;
                Ok(current & 15)
            }
        };
        for _ in 0..count {
            if nibble(&mut self.c)? > 4 {
                return Err(bad());
            }
            loop {
                let n = nibble(&mut self.c)?;
                if n == 15 {
                    break;
                }
                if n > 11 {
                    return Err(bad());
                }
            }
        }
        if low && current & 15 != 0 {
            return Err(bad());
        }
        Ok(())
    }
    fn embellished(&mut self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        loop {
            let kind = self.c.u8()?;
            self.c.charge()?;
            if kind == 0 {
                return Ok(out);
            }
            if kind != 6 {
                return Err(bad());
            }
            let options = self.c.u8()?;
            if options & !8 != 0 {
                return Err(bad());
            }
            if options & 8 != 0 {
                self.c.nudge()?;
            }
            let embellishment = self.c.u8()?;
            if !(2..=37).contains(&embellishment) {
                return Err(bad());
            }
            out.push(embellishment);
        }
    }
    fn character(&mut self) -> Result<Node> {
        let options = self.c.u8()?;
        if options & !0x3f != 0 || options & 4 != 0 && options & 16 != 0 {
            return Err(bad());
        }
        if options & 8 != 0 {
            self.c.nudge()?;
        }
        let face = self.c.signed()?;
        if face < 0 {
            if self
                .styles
                .get((-face) as usize)
                .copied()
                .flatten()
                .is_none()
            {
                return Err(bad());
            }
        } else if !(1..=12).contains(&face) && !matches!(face, 22..=24) {
            return Err(bad());
        }
        let code = if options & 32 != 0 {
            None
        } else {
            Some(u32::from(self.c.u16()?))
        };
        let font_position = if options & 4 != 0 {
            Some(u32::from(self.c.u8()?))
        } else if options & 16 != 0 {
            Some(u32::from(self.c.u16()?))
        } else {
            None
        };
        let value = if let Some(code) = code {
            mtcode(code)?
        } else {
            position(self.encoding(face), font_position.ok_or_else(bad)?)?
        };
        let embellishments = if options & 1 != 0 {
            self.embellished()?
        } else {
            vec![]
        };
        Ok(Node::Character(Char {
            value,
            embellishments,
            face,
            style: self.style(face).map(|s| s.1).unwrap_or(0),
            function: options & 2 != 0,
            v5: true,
        }))
    }
    fn record(&mut self, kind: u8) -> Result<Node> {
        match kind {
            1 => {
                let options = self.c.u8()?;
                if options & !15 != 0 {
                    return Err(bad());
                }
                if options & 8 != 0 {
                    self.c.nudge()?;
                }
                if options & 4 != 0 {
                    self.c.u16()?;
                }
                if options & 2 != 0 {
                    if self.c.u8()? != 7 {
                        return Err(bad());
                    }
                    self.c.charge()?;
                    self.ruler()?;
                }
                if options & 1 != 0 {
                    return Ok(Node::Sequence(vec![]));
                }
                self.c.enter()?;
                let children = self.list()?;
                self.c.depth -= 1;
                Ok(Node::Sequence(children))
            }
            2 => self.character(),
            3 => {
                let options = self.c.u8()?;
                if options & !8 != 0 {
                    return Err(bad());
                }
                if options & 8 != 0 {
                    self.c.nudge()?;
                }
                let selector = self.c.u8()?;
                let first = self.c.u8()?;
                let variation = if first & 128 != 0 {
                    u16::from(first & 127) | (u16::from(self.c.u8()?) << 8)
                } else {
                    first.into()
                };
                let options = self.c.u8()?;
                self.c.enter()?;
                let slots = self.list()?;
                self.c.depth -= 1;
                render::validate_v5(selector, variation, options)?;
                Ok(Node::Template {
                    v5: true,
                    selector,
                    variation,
                    options,
                    slots,
                })
            }
            4 => {
                let options = self.c.u8()?;
                if options & !10 != 0 {
                    return Err(bad());
                }
                if options & 8 != 0 {
                    self.c.nudge()?;
                }
                if !(1..=5).contains(&self.c.u8()?) || self.c.u8()? > 4 {
                    return Err(bad());
                }
                if options & 2 != 0 {
                    if self.c.u8()? != 7 {
                        return Err(bad());
                    }
                    self.c.charge()?;
                    self.ruler()?;
                }
                self.c.enter()?;
                let children = self.list()?;
                self.c.depth -= 1;
                if children.iter().any(|n| !matches!(n, Node::Sequence(_))) {
                    return Err(bad());
                }
                Ok(Node::Pile(children))
            }
            5 => {
                let options = self.c.u8()?;
                if options & !8 != 0 {
                    return Err(bad());
                }
                if options & 8 != 0 {
                    self.c.nudge()?;
                }
                if self.c.u8()? > 4 || !(1..=5).contains(&self.c.u8()?) || self.c.u8()? > 4 {
                    return Err(bad());
                }
                let rows = usize::from(self.c.u8()?);
                let cols = usize::from(self.c.u8()?);
                if rows == 0 || cols == 0 || rows * cols > 4096 {
                    return Err(bad());
                }
                self.c
                    .bytes((2 * (rows + 1)).div_ceil(8) + (2 * (cols + 1)).div_ceil(8))?;
                self.c.enter()?;
                let cells = self.list()?;
                self.c.depth -= 1;
                if cells.len() != rows * cols
                    || cells.iter().any(|n| !matches!(n, Node::Sequence(_)))
                {
                    return Err(bad());
                }
                Ok(Node::Matrix { rows, cols, cells })
            }
            7 => {
                self.ruler()?;
                Ok(Node::Metadata)
            }
            8 => {
                let font = usize::from(self.c.unsigned()?);
                let style = self.c.u8()?;
                if style & !3 != 0 || font == 0 || font >= self.fonts.len() {
                    return Err(bad());
                }
                self.styles.push(Some((font, style)));
                Ok(Node::Metadata)
            }
            9 => {
                self.size()?;
                Ok(Node::Metadata)
            }
            10..=14 => Ok(Node::Metadata),
            15 => {
                let color = usize::from(self.c.unsigned()?);
                if color == 0 || color > self.colors {
                    return Err(bad());
                }
                Ok(Node::Metadata)
            }
            16 => {
                let options = self.c.u8()?;
                if options & !7 != 0 {
                    return Err(bad());
                }
                for _ in 0..if options & 1 != 0 { 4 } else { 3 } {
                    if self.c.u16()? > 1000 {
                        return Err(bad());
                    }
                }
                if options & 4 != 0 {
                    self.c.cstring()?;
                }
                self.colors = self.colors.checked_add(1).ok_or_else(bad)?;
                Ok(Node::Metadata)
            }
            17 => {
                let encoding = usize::from(self.c.unsigned()?);
                if !self.encodings.contains_key(&encoding) || self.c.cstring()?.is_empty() {
                    return Err(bad());
                }
                self.fonts.push(Some(encoding));
                Ok(Node::Metadata)
            }
            18 => {
                if self.c.u8()? != 0 {
                    return Err(bad());
                }
                self.dimension()?;
                self.dimension()?;
                let mut styles = Vec::new();
                for _ in 0..self.c.u8()? {
                    let font = usize::from(self.c.unsigned()?);
                    if font == 0 {
                        styles.push(None);
                        continue;
                    }
                    if font >= self.fonts.len() {
                        return Err(bad());
                    }
                    let bits = self.c.u8()?;
                    if bits & !3 != 0 {
                        return Err(bad());
                    }
                    styles.push(Some((font, bits)));
                }
                self.preferences = styles;
                Ok(Node::Metadata)
            }
            19 => {
                let name = self.c.cstring()?;
                if name.is_empty() {
                    return Err(bad());
                }
                let index = self
                    .encodings
                    .keys()
                    .next_back()
                    .unwrap()
                    .checked_add(1)
                    .ok_or_else(bad)?;
                self.encodings.insert(index, name);
                Ok(Node::Metadata)
            }
            100.. => {
                let count = usize::from(self.c.unsigned()?);
                self.c.bytes(count)?;
                Ok(Node::Metadata)
            }
            _ => Err(bad()),
        }
    }
    fn list(&mut self) -> Result<Vec<Node>> {
        let mut out = Vec::new();
        loop {
            let kind = self.c.u8()?;
            self.c.charge()?;
            if kind == 0 {
                return group(out);
            }
            let node = self.record(kind)?;
            if !matches!(node, Node::Metadata) {
                out.push(node);
            }
        }
    }
}
fn group(nodes: Vec<Node>) -> Result<Vec<Node>> {
    let mut out = Vec::new();
    let mut nodes = nodes.into_iter().peekable();
    while let Some(node) = nodes.next() {
        if let Node::Character(c) = &node {
            if let Character::Code(code) = c.value {
                if c.embellishments.is_empty()
                    && c.style == 0
                    && (matches!(c.face, 1 | 12) || c.face == 2 && c.function)
                {
                    let mut text = char::from_u32(code).ok_or_else(bad)?.to_string();
                    while let Some(Node::Character(next)) = nodes.peek() {
                        if next.face != c.face || !next.embellishments.is_empty() || next.style != 0
                        {
                            break;
                        }
                        let Character::Code(code) = next.value else {
                            break;
                        };
                        text.push(char::from_u32(code).ok_or_else(bad)?);
                        nodes.next();
                    }
                    out.push(if c.face == 2 {
                        Node::Function(text)
                    } else {
                        Node::Text(text)
                    });
                    continue;
                }
            }
        }
        out.push(node);
    }
    Ok(out)
}
