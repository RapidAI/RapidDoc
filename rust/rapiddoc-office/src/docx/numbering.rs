use super::*;
use num_bigint::BigInt;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq)]
pub struct NumberingProperties {
    pub num_id: Option<i64>,
    pub level: Option<i64>,
}
impl NumberingProperties {
    pub(super) fn read(node: Node<'_, '_>) -> Self {
        Self {
            num_id: child(node, "numId").and_then(|n| integer(attr(n, "val"))),
            level: child(node, "ilvl").and_then(|n| integer(attr(n, "val"))),
        }
    }
}
#[derive(Clone)]
struct Level {
    raw_id: String,
    style: Option<String>,
    start: Option<BigInt>,
    format: Option<String>,
}
#[derive(Clone)]
struct Definition {
    abstract_id: Option<String>,
    overrides: BTreeMap<String, Option<BigInt>>,
}
#[derive(Default, Clone)]
pub struct Numbering {
    abstracts: BTreeMap<String, Vec<Level>>,
    definitions: BTreeMap<String, Definition>,
    counters: BTreeMap<(i64, i64), Option<BigInt>>,
}
impl Numbering {
    pub fn parse(xml: Option<&str>) -> Result<Self> {
        let mut output = Self::default();
        let Some(xml) = xml else {
            return Ok(output);
        };
        let doc = crate::xml::parse(xml)?;
        for node in doc.descendants().filter(|n| is_word(*n, "abstractNum")) {
            let Some(id) = attr(node, "abstractNumId") else {
                continue;
            };
            let mut levels = Vec::new();
            for level in node.descendants().filter(|n| is_word(*n, "lvl")) {
                let Some(raw_id) = attr(level, "ilvl") else {
                    continue;
                };
                levels.push(Level {
                    raw_id: raw_id.to_owned(),
                    style: child(level, "pStyle")
                        .and_then(|n| attr(n, "val"))
                        .map(str::to_owned),
                    start: match child(level, "start") {
                        Some(n) => attr(n, "val")
                            .map(|s| big_integer(s).unwrap_or_else(|| BigInt::from(1))),
                        None => Some(BigInt::from(1)),
                    },
                    format: level
                        .descendants()
                        .find(|n| is_word(*n, "numFmt"))
                        .and_then(|n| attr(n, "val"))
                        .map(str::to_owned),
                });
            }
            output.abstracts.entry(id.to_owned()).or_insert(levels);
        }
        for node in doc.descendants().filter(|n| is_word(*n, "num")) {
            let Some(id) = attr(node, "numId") else {
                continue;
            };
            let abstract_id = node
                .descendants()
                .find(|n| is_word(*n, "abstractNumId"))
                .and_then(|n| attr(n, "val"))
                .map(str::to_owned);
            let mut overrides = BTreeMap::new();
            for level in node.children().filter(|n| is_word(*n, "lvlOverride")) {
                if let Some((id, start)) = attr(level, "ilvl").zip(child(level, "startOverride")) {
                    overrides.entry(id.to_owned()).or_insert(
                        attr(start, "val")
                            .map(|s| big_integer(s).unwrap_or_else(|| BigInt::from(1))),
                    );
                }
            }
            output
                .definitions
                .entry(id.to_owned())
                .or_insert(Definition {
                    abstract_id,
                    overrides,
                });
        }
        Ok(output)
    }
    fn levels(&self, id: i64) -> Option<&[Level]> {
        let definition = self.definitions.get(&id.to_string())?;
        self.abstracts
            .get(definition.abstract_id.as_ref()?)
            .map(Vec::as_slice)
    }
    pub fn infer_level(&self, id: i64, styles: &BTreeSet<String>) -> Option<i64> {
        let level = self
            .levels(id)?
            .iter()
            .find(|level| level.style.as_ref().is_some_and(|s| styles.contains(s)))?;
        integer(Some(&level.raw_id))
    }
    pub fn is_numbered(&self, id: i64, level: i64) -> bool {
        self.levels(id)
            .and_then(|levels| levels.iter().find(|l| l.raw_id == level.to_string()))
            .and_then(|l| l.format.as_deref())
            .is_some_and(|format| {
                matches!(
                    format,
                    "decimal"
                        | "lowerRoman"
                        | "upperRoman"
                        | "lowerLetter"
                        | "upperLetter"
                        | "decimalZero"
                )
            })
    }
    pub fn start(&self, id: i64, level: i64) -> Option<BigInt> {
        if let Some(value) = self
            .definitions
            .get(&id.to_string())
            .and_then(|d| d.overrides.get(&level.to_string()))
        {
            return value.clone();
        }
        self.levels(id)
            .and_then(|levels| levels.iter().find(|l| l.raw_id == level.to_string()))
            .map_or(Some(BigInt::from(1)), |l| l.start.clone())
    }
    pub fn advance(&mut self, id: i64, level: i64) -> Result<Option<BigInt>> {
        let current = match self.counters.get(&(id, level)) {
            Some(Some(value)) => Some(value + 1),
            Some(None) => return Err(Error::malformed("cannot increment missing numbering start")),
            None => self.start(id, level),
        };
        self.counters.insert((id, level), current.clone());
        self.counters
            .retain(|(other_id, other_level), _| *other_id != id || *other_level <= level);
        Ok(current)
    }
}
fn big_integer(text: &str) -> Option<BigInt> {
    integer_text(Some(text))?.parse().ok()
}
