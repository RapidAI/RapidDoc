//! Minimal lossless ONNX protobuf wire editor. Unknown fields and tensor bytes
//! are retained. Arc-backed slices avoid cloning weights while editing metadata.
use rapiddoc_core::{Error, ErrorCode};
use std::{ops::Range, sync::Arc};
#[derive(Clone)]
pub struct Bytes {
    data: Arc<Vec<u8>>,
    range: Range<usize>,
}
impl Bytes {
    pub fn new(v: Vec<u8>) -> Self {
        let end = v.len();
        Self {
            data: Arc::new(v),
            range: 0..end,
        }
    }
    pub fn as_ref(&self) -> &[u8] {
        &self.data[self.range.clone()]
    }
    fn sub(&self, r: Range<usize>) -> Self {
        Self {
            data: self.data.clone(),
            range: self.range.start + r.start..self.range.start + r.end,
        }
    }
}
#[derive(Clone)]
pub struct Field {
    pub tag: u32,
    pub wire: u8,
    pub data: Bytes,
    nested: Option<Box<Msg>>,
}
#[derive(Clone, Default)]
pub struct Msg {
    pub fields: Vec<Field>,
}
pub fn invalid(s: impl ToString) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: s.to_string(),
    }
}
fn take_varint(bytes: &[u8], cursor: &mut usize) -> Result<u64, Error> {
    let mut result = 0u64;
    for shift in (0..70).step_by(7) {
        let b = *bytes
            .get(*cursor)
            .ok_or_else(|| invalid("Truncated ONNX varint"))?;
        *cursor += 1;
        if shift == 63 && b > 1 {
            return Err(invalid("ONNX varint overflow"));
        }
        result |= ((b & 127) as u64) << shift;
        if b < 128 {
            return Ok(result);
        }
    }
    Err(invalid("Invalid ONNX varint"))
}
fn varint(mut n: u64) -> Vec<u8> {
    let mut v = vec![];
    while n >= 128 {
        v.push(n as u8 | 128);
        n >>= 7;
    }
    v.push(n as u8);
    v
}
impl Field {
    pub fn text(tag: u32, s: &str) -> Self {
        Self {
            tag,
            wire: 2,
            data: Bytes::new(s.as_bytes().to_vec()),
            nested: None,
        }
    }
    pub fn integer(tag: u32, n: i64) -> Self {
        Self {
            tag,
            wire: 0,
            data: Bytes::new(varint(n as u64)),
            nested: None,
        }
    }
    pub fn nested(tag: u32, m: &Msg) -> Self {
        Self {
            tag,
            wire: 2,
            data: Bytes::new(Vec::new()),
            nested: Some(Box::new(m.clone())),
        }
    }
    pub fn int(&self) -> Result<i64, Error> {
        let mut p = 0;
        Ok(take_varint(self.data.as_ref(), &mut p)? as i64)
    }
    pub fn string(&self) -> Result<&str, Error> {
        std::str::from_utf8(self.data.as_ref()).map_err(invalid)
    }
    pub fn msg(&self) -> Result<Msg, Error> {
        if self.wire != 2 {
            return Err(invalid("Expected ONNX message"));
        }
        if let Some(message) = self.nested.as_ref() {
            return Ok((**message).clone());
        }
        Msg::parse(self.data.clone())
    }
}
impl Msg {
    pub fn parse(bytes: Bytes) -> Result<Self, Error> {
        let b = bytes.as_ref();
        let mut p = 0;
        let mut fields = vec![];
        while p < b.len() {
            let key = take_varint(b, &mut p)?;
            let tag = u32::try_from(key >> 3).map_err(invalid)?;
            if tag == 0 {
                return Err(invalid("Zero protobuf field"));
            }
            let wire = (key & 7) as u8;
            let (start, end) = match wire {
                0 => {
                    let start = p;
                    take_varint(b, &mut p)?;
                    (start, p)
                }
                1 | 5 => {
                    let start = p;
                    p = p
                        .checked_add(if wire == 1 { 8 } else { 4 })
                        .ok_or_else(|| invalid("ONNX length overflow"))?;
                    (start, p)
                }
                2 => {
                    let n = usize::try_from(take_varint(b, &mut p)?).map_err(invalid)?;
                    let start = p;
                    p = p
                        .checked_add(n)
                        .ok_or_else(|| invalid("ONNX length overflow"))?;
                    (start, p)
                }
                _ => return Err(invalid("Unsupported ONNX protobuf wire type")),
            };
            if end > b.len() {
                return Err(invalid("Truncated ONNX field"));
            }
            fields.push(Field {
                tag,
                wire,
                data: bytes.sub(start..end),
                nested: None,
            });
        }
        Ok(Self { fields })
    }
    pub fn decode(v: Vec<u8>) -> Result<Self, Error> {
        Self::parse(Bytes::new(v))
    }
    fn encoded_len(&self) -> usize {
        fn varlen(mut value: u64) -> usize {
            let mut n = 1;
            while value >= 128 {
                value >>= 7;
                n += 1;
            }
            n
        }
        self.fields
            .iter()
            .map(|f| {
                let length = f
                    .nested
                    .as_ref()
                    .map(|m| m.encoded_len())
                    .unwrap_or_else(|| f.data.as_ref().len());
                varlen(((f.tag as u64) << 3) | f.wire as u64)
                    + if f.wire == 2 {
                        varlen(length as u64)
                    } else {
                        0
                    }
                    + length
            })
            .sum()
    }
    fn encode_into(&self, output: &mut Vec<u8>) {
        for f in &self.fields {
            output.extend(varint(((f.tag as u64) << 3) | f.wire as u64));
            if f.wire == 2 {
                let length = f
                    .nested
                    .as_ref()
                    .map(|m| m.encoded_len())
                    .unwrap_or_else(|| f.data.as_ref().len());
                output.extend(varint(length as u64));
            }
            if let Some(message) = f.nested.as_ref() {
                message.encode_into(output);
            } else {
                output.extend_from_slice(f.data.as_ref());
            }
        }
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(self.encoded_len());
        self.encode_into(&mut output);
        output
    }
    pub fn all(&self, tag: u32) -> Vec<Field> {
        self.fields
            .iter()
            .filter(|f| f.tag == tag)
            .cloned()
            .collect()
    }
    pub fn one(&self, tag: u32) -> Option<&Field> {
        self.fields.iter().find(|f| f.tag == tag)
    }
    pub fn text(&self, tag: u32) -> Result<String, Error> {
        self.one(tag)
            .map(|f| f.string().map(str::to_owned))
            .unwrap_or_else(|| Ok(String::new()))
    }
    pub fn int(&self, tag: u32, default: i64) -> Result<i64, Error> {
        self.one(tag).map(Field::int).unwrap_or(Ok(default))
    }
    pub fn sub(&self, tag: u32) -> Result<Self, Error> {
        self.one(tag)
            .ok_or_else(|| invalid(format!("Missing ONNX message {tag}")))?
            .msg()
    }
    pub fn replace(&mut self, tag: u32, fields: Vec<Field>) {
        let at = self
            .fields
            .iter()
            .position(|f| f.tag == tag)
            .unwrap_or(self.fields.len());
        self.fields.retain(|f| f.tag != tag);
        self.fields.splice(at..at, fields);
    }
    pub fn set_text(&mut self, tag: u32, value: &str) {
        self.replace(tag, vec![Field::text(tag, value)]);
    }
    pub fn set_int(&mut self, tag: u32, value: i64) {
        self.replace(tag, vec![Field::integer(tag, value)]);
    }
    pub fn set_sub(&mut self, tag: u32, value: &Msg) {
        self.replace(tag, vec![Field::nested(tag, value)]);
    }
    pub fn texts(&self, tag: u32) -> Result<Vec<String>, Error> {
        self.all(tag)
            .iter()
            .map(|f| f.string().map(str::to_owned))
            .collect()
    }
    pub fn ints(&self, tag: u32) -> Result<Vec<i64>, Error> {
        let mut out = vec![];
        for f in self.all(tag) {
            if f.wire == 0 {
                out.push(f.int()?);
            } else if f.wire == 2 {
                let mut p = 0;
                while p < f.data.as_ref().len() {
                    out.push(take_varint(f.data.as_ref(), &mut p)? as i64);
                }
            } else {
                return Err(invalid("Invalid packed integer"));
            }
        }
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wire_roundtrip_preserves_unknown_fields_and_checks_lengths() {
        let bytes = vec![8, 1, 18, 3, 1, 2, 3, 29, 0, 0, 0, 0];
        let m = Msg::decode(bytes.clone()).unwrap();
        assert_eq!(m.encode(), bytes);
        assert_eq!(m.int(1, 0).unwrap(), 1);
        assert!(Msg::decode(vec![18, 9, 1]).is_err());
        assert!(Msg::decode(vec![0]).is_err());
        assert!(Msg::decode(vec![8, 255]).is_err());
    }
}

#[cfg(test)]
mod nested_tests {
    use super::*;
    #[test]
    fn nested_edits_preserve_shared_weight_storage_and_exact_wire_output() {
        let mut tensor = Msg::default();
        tensor.set_text(8, "weights");
        tensor.fields.push(Field {
            tag: 9,
            wire: 2,
            data: Bytes::new(vec![17; 1024 * 1024]),
            nested: None,
        });
        let data = tensor.one(9).unwrap().data.data.clone();
        let mut graph = Msg::default();
        graph.set_sub(5, &tensor);
        let mut model = Msg::default();
        model.set_int(1, 8);
        model.set_sub(7, &graph);
        assert!(Arc::ptr_eq(
            &data,
            &model
                .sub(7)
                .unwrap()
                .sub(5)
                .unwrap()
                .one(9)
                .unwrap()
                .data
                .data
        ));
        let encoded = model.encode();
        assert_eq!(encoded.len(), model.encoded_len());
        let decoded = Msg::decode(encoded.clone()).unwrap();
        assert_eq!(decoded.encode(), encoded);
        let mut changed = decoded.sub(7).unwrap();
        changed.set_text(2, "renamed graph");
        let mut updated = decoded.clone();
        updated.set_sub(7, &changed);
        assert_eq!(
            updated
                .sub(7)
                .unwrap()
                .sub(5)
                .unwrap()
                .one(9)
                .unwrap()
                .data
                .as_ref(),
            &[17; 1024 * 1024]
        );
        assert_eq!(
            Msg::decode(updated.encode())
                .unwrap()
                .sub(7)
                .unwrap()
                .text(2)
                .unwrap(),
            "renamed graph"
        );
    }
}
