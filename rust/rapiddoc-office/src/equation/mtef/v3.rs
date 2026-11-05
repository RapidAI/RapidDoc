use super::*;
pub(super) fn parse(bytes: &[u8]) -> Result<Node> {
    if bytes.len() < 5 || bytes[0] != 3 {
        return Err(bad());
    }
    let mut reader = Reader {
        c: Cursor::new(bytes, 5),
    };
    Ok(Node::Sequence(reader.list()?)) // v3 intentionally accepts trailing bytes.
}
struct Reader<'a> {
    c: Cursor<'a>,
}
impl Reader<'_> {
    fn ruler(&mut self) -> Result<()> {
        let count = self.c.u8()?;
        self.c.bytes(usize::from(count) * 3)?;
        Ok(())
    }
    fn font(&mut self) -> Result<()> {
        self.c.bytes(2)?;
        while self.c.u8()? != 0 {}
        Ok(())
    }
    fn size(&mut self) -> Result<()> {
        let b = self.c.u8()?;
        self.c.bytes(if b == 100 {
            3
        } else if b == 101 {
            2
        } else {
            1
        })?;
        Ok(())
    }
    fn embellished(&mut self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        loop {
            let tag = self.c.u8()?;
            self.c.charge()?;
            let kind = tag & 15;
            if kind == 0 {
                return Ok(out);
            }
            if kind != 6 {
                return Err(bad());
            }
            if tag & 128 != 0 {
                self.c.nudge()?;
            }
            out.push(self.c.u8()?);
        }
    }
    fn line(&mut self, flags: u8) -> Result<Node> {
        if flags & 16 != 0 {
            return Ok(Node::Sequence(vec![]));
        }
        if flags & 128 != 0 {
            self.c.nudge()?;
        }
        if flags & 64 != 0 {
            self.c.u8()?;
        }
        if flags & 32 != 0 {
            if self.c.u8()? & 15 != 7 {
                return Err(bad());
            }
            self.ruler()?;
        }
        self.c.enter()?;
        let children = self.list()?;
        self.c.depth -= 1;
        Ok(Node::Sequence(children))
    }
    fn record(&mut self, kind: u8, flags: u8) -> Result<Node> {
        match kind {
            1 => self.line(flags),
            2 => {
                if flags & 128 != 0 {
                    self.c.nudge()?;
                }
                let face = self.c.u8()?.into();
                let code = self.c.u16()?.into();
                let embellishments = if flags & 32 != 0 {
                    self.embellished()?
                } else {
                    vec![]
                };
                Ok(Node::Character(Char {
                    value: Character::Code(code),
                    embellishments,
                    style: 0,
                    face,
                    function: false,
                    v5: false,
                }))
            }
            3 => {
                if flags & 128 != 0 {
                    self.c.nudge()?;
                }
                let selector = self.c.u8()?;
                let variation = self.c.u8()?.into();
                let options = self.c.u8()?;
                self.c.enter()?;
                let slots = self.list()?;
                self.c.depth -= 1;
                Ok(Node::Template {
                    v5: false,
                    selector,
                    variation,
                    options,
                    slots,
                })
            }
            4 => {
                if flags & 128 != 0 {
                    self.c.nudge()?;
                }
                self.c.bytes(2)?;
                if flags & 32 != 0 {
                    if self.c.u8()? & 15 != 7 {
                        return Err(bad());
                    }
                    self.ruler()?;
                }
                self.c.enter()?;
                let slots = self.list()?;
                self.c.depth -= 1;
                Ok(Node::Pile(slots))
            }
            5 => {
                if flags & 128 != 0 {
                    self.c.nudge()?;
                }
                self.c.bytes(3)?;
                let rows = usize::from(self.c.u8()?);
                let cols = usize::from(self.c.u8()?);
                if rows == 0 || cols == 0 || rows * cols > 4096 {
                    return Err(bad());
                }
                self.c
                    .bytes((2 * (rows + 1)).div_ceil(8) + (2 * (cols + 1)).div_ceil(8))?;
                self.c.enter()?;
                let mut cells = Vec::new();
                while cells.len() < rows * cols {
                    let tag = self.c.u8()?;
                    self.c.charge()?;
                    if tag & 15 == 0 {
                        continue;
                    }
                    let node = self.record(tag & 15, tag & 240)?;
                    if !matches!(node, Node::Metadata) {
                        cells.push(node);
                    }
                }
                self.c.depth -= 1;
                if self.c.data.get(self.c.pos).is_some_and(|b| b & 15 == 0) {
                    self.c.pos += 1;
                    self.c.charge()?;
                }
                Ok(Node::Matrix { rows, cols, cells })
            }
            6 => {
                if flags & 128 != 0 {
                    self.c.nudge()?;
                }
                self.c.u8()?;
                Ok(Node::Embellishment)
            }
            7 => {
                self.ruler()?;
                Ok(Node::Metadata)
            }
            8 => {
                self.font()?;
                Ok(Node::Metadata)
            }
            9 => {
                self.size()?;
                Ok(Node::Metadata)
            }
            10..=14 => Ok(Node::Metadata),
            _ => Err(bad()),
        }
    }
    fn list(&mut self) -> Result<Vec<Node>> {
        let mut out = Vec::new();
        loop {
            let tag = self.c.u8()?;
            self.c.charge()?;
            if tag & 15 == 0 {
                return Ok(out);
            }
            let node = self.record(tag & 15, tag & 240)?;
            if !matches!(node, Node::Metadata) {
                out.push(node);
            }
        }
    }
}
