//! Paragraph SPRMs, including huge PAPX and TDefTable merge flags.
use super::formatting::sprms;
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Cell {
    pub right: i16,
    pub horizontal_first: bool,
    pub horizontal_continue: bool,
    pub vertical_first: bool,
    pub vertical_continue: bool,
}
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Table {
    pub boundaries: Vec<i16>,
    pub cells: Vec<Cell>,
    pub header: bool,
}
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Paragraph {
    pub in_table: Option<bool>,
    pub row_mark: Option<bool>,
    pub outline_level: Option<i16>,
    pub ilfo: Option<u16>,
    pub ilvl: Option<u8>,
    pub table_depth: Option<i64>,
    pub inner_cell: Option<bool>,
    pub inner_row: Option<bool>,
    pub table: Option<Table>,
}
fn u16at(d: &[u8], p: usize) -> Option<u16> {
    Some(u16::from_le_bytes(d.get(p..p + 2)?.try_into().ok()?))
}
fn u32at(d: &[u8], p: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(p..p + 4)?.try_into().ok()?))
}
fn table(d: &[u8]) -> Option<Table> {
    let n = *d.get(2)? as usize;
    if n > 63 {
        return None;
    }
    let mut boundaries = vec![];
    for i in 0..=n {
        boundaries.push(u16at(d, 3 + i * 2)? as i16)
    }
    let base = 3 + (n + 1) * 2;
    let mut cells = vec![];
    for i in 0..n {
        let flags = u16at(d, base + i * 20).unwrap_or(0);
        let h = flags & 3;
        let v = (flags >> 5) & 3;
        cells.push(Cell {
            right: boundaries[i + 1],
            horizontal_first: h >= 2,
            horizontal_continue: h == 1,
            vertical_first: v == 3,
            vertical_continue: v == 1,
        });
    }
    Some(Table {
        boundaries,
        cells,
        header: false,
    })
}
impl Paragraph {
    pub fn resolve(style: &super::styles::Style, papx: &[u8], prm: &[u8], data: &[u8]) -> Self {
        let mut out = Self::default();
        for bytes in &style.paragraph {
            out.apply(bytes, &[]);
        }
        let mut direct = Self::default();
        direct.apply(papx, data);
        out.merge(&direct);
        out.apply(prm, &[]);
        out
    }
    pub fn merge(&mut self, over: &Self) {
        macro_rules! copy {($($f:ident),*)=>{$(if over.$f.is_some(){self.$f=over.$f.clone();})*}}
        copy!(
            in_table,
            row_mark,
            outline_level,
            ilfo,
            ilvl,
            table_depth,
            inner_cell,
            inner_row,
            table
        );
    }
    pub fn apply(&mut self, grpprl: &[u8], data: &[u8]) {
        for (op, v) in sprms(grpprl) {
            match op {
                0x2416 => self.in_table = Some(v[0] != 0),
                0x2417 => self.row_mark = Some(v[0] != 0),
                0x2640 => self.outline_level = Some(if v[0] < 9 { v[0] as i16 + 1 } else { -1 }),
                0x260a => self.ilvl = Some(v[0]),
                0x460b => self.ilfo = u16at(v, 0),
                0x6649 => self.table_depth = u32at(v, 0).map(i64::from),
                0x664a => {
                    if let Some(n) = u32at(v, 0) {
                        self.table_depth =
                            Some((self.table_depth.unwrap_or(0) + n as i32 as i64).max(0));
                    }
                }
                0x244b => self.inner_cell = Some(v[0] != 0),
                0x244c => self.inner_row = Some(v[0] != 0),
                0x6646 => {
                    if let Some(offset) = u32at(v, 0) {
                        let offset = offset as usize;
                        if let Some(n) = u16at(data, offset) {
                            if let Some(bytes) = data.get(offset + 2..offset + 2 + n as usize) {
                                self.apply(bytes, &[]);
                            }
                        }
                    }
                }
                0xd608 => {
                    if let Some(mut t) = table(v) {
                        t.header = self.table.as_ref().is_some_and(|t| t.header);
                        self.table = Some(t);
                    }
                }
                0x3404 => self.table.get_or_insert_with(Table::default).header = v[0] != 0,
                0xd62b => {
                    if v.len() >= 3 {
                        if let Some(c) = self
                            .table
                            .as_mut()
                            .and_then(|t| t.cells.get_mut(v[1] as usize))
                        {
                            c.vertical_continue = v[2] == 1;
                            c.vertical_first = v[2] == 3;
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
