//! Candidate membership stays compact until the final output boundary.
use serde::{ser::SerializeSeq, Serialize, Serializer};
use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

pub(super) trait Lookup {
    fn contains(&self, id: &i64) -> bool;
}
impl Lookup for BTreeSet<i64> {
    fn contains(&self, id: &i64) -> bool {
        BTreeSet::contains(self, id)
    }
}

pub(super) struct Domain {
    pub(super) ids: Vec<i64>,
    pub(super) ranks: HashMap<i64, usize>,
}
impl Domain {
    pub(super) fn new(ids: BTreeSet<i64>) -> Arc<Self> {
        let ids: Vec<_> = ids.into_iter().collect();
        let ranks = ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        Arc::new(Self { ids, ranks })
    }
    fn pack(&self, ids: impl Iterator<Item = i64>) -> Option<Vec<u64>> {
        let mut words = vec![0; self.ids.len().div_ceil(64)];
        for id in ids {
            let rank = *self.ranks.get(&id)?;
            words[rank / 64] |= 1u64 << (rank % 64);
        }
        Some(words)
    }
}
#[derive(Clone)]
pub(super) enum Members {
    Tree(Arc<BTreeSet<i64>>),
    Packed {
        domain: Arc<Domain>,
        words: Arc<Vec<u64>>,
        count: usize,
    },
}
impl From<BTreeSet<i64>> for Members {
    fn from(ids: BTreeSet<i64>) -> Self {
        Self::Tree(Arc::new(ids))
    }
}
impl Lookup for Members {
    fn contains(&self, id: &i64) -> bool {
        match self {
            Self::Tree(ids) => ids.contains(id),
            Self::Packed { domain, words, .. } => domain
                .ranks
                .get(id)
                .is_some_and(|rank| words[rank / 64] & (1u64 << (rank % 64)) != 0),
        }
    }
}
impl Members {
    pub(super) fn packed(domain: Arc<Domain>, words: Vec<u64>) -> Self {
        let count = words.iter().map(|w| w.count_ones() as usize).sum();
        Self::Packed {
            domain,
            words: Arc::new(words),
            count,
        }
    }
    pub(super) fn from_domain(domain: Arc<Domain>, ids: &BTreeSet<i64>) -> Self {
        match domain.pack(ids.iter().copied()) {
            Some(words) => Self::packed(domain, words),
            None => ids.clone().into(),
        }
    }
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Tree(ids) => ids.len(),
            Self::Packed { count, .. } => *count,
        }
    }
    pub(super) fn iter(&self) -> Box<dyn Iterator<Item = i64> + '_> {
        match self {
            Self::Tree(ids) => Box::new(ids.iter().copied()),
            Self::Packed { domain, words, .. } => {
                Box::new(words.iter().enumerate().flat_map(move |(word, bits)| {
                    let mut bits = *bits;
                    std::iter::from_fn(move || {
                        if bits == 0 {
                            return None;
                        }
                        let rank = word * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        Some(domain.ids[rank])
                    })
                }))
            }
        }
    }
    pub(super) fn remove(&mut self, id: &i64) {
        match self {
            Self::Tree(ids) => {
                Arc::make_mut(ids).remove(id);
            }
            Self::Packed {
                domain,
                words,
                count,
            } => {
                if let Some(rank) = domain.ranks.get(id) {
                    let bit = 1u64 << (rank % 64);
                    if words[rank / 64] & bit != 0 {
                        Arc::make_mut(words)[rank / 64] &= !bit;
                        *count -= 1;
                    }
                }
            }
        }
    }
    pub(super) fn union_with(&mut self, other: &Self) {
        // Map a legacy/plain candidate into the first shared page domain once.
        if let (Self::Tree(ids), Self::Packed { domain, .. }) = (&*self, other) {
            if let Some(words) = domain.pack(ids.iter().copied()) {
                *self = Self::packed(Arc::clone(domain), words);
            }
        }
        match (&mut *self, other) {
            (
                Self::Packed {
                    domain,
                    words,
                    count,
                },
                Self::Packed {
                    domain: right_domain,
                    words: right,
                    ..
                },
            ) if Arc::ptr_eq(domain, right_domain) => {
                if !Arc::ptr_eq(words, right) {
                    let data = Arc::make_mut(words);
                    for (left, right) in data.iter_mut().zip(right.iter()) {
                        *left |= right;
                    }
                    *count = data.iter().map(|w| w.count_ones() as usize).sum();
                }
            }
            (
                Self::Packed {
                    domain,
                    words,
                    count,
                },
                Self::Tree(right),
            ) if right.iter().all(|id| domain.ranks.contains_key(id)) => {
                for id in right.iter() {
                    let rank = domain.ranks[id];
                    let bit = 1u64 << (rank % 64);
                    if words[rank / 64] & bit == 0 {
                        Arc::make_mut(words)[rank / 64] |= bit;
                        *count += 1;
                    }
                }
            }
            (Self::Tree(left), Self::Tree(right)) => {
                if !Arc::ptr_eq(left, right) {
                    if left.is_subset(right) {
                        *left = Arc::clone(right);
                    } else if !right.is_subset(left) {
                        Arc::make_mut(left).extend(right.iter().copied());
                    }
                }
            }
            _ => {
                *self = self
                    .iter()
                    .chain(other.iter())
                    .collect::<BTreeSet<_>>()
                    .into();
            }
        }
    }
}
impl Serialize for Members {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for id in self.iter() {
            seq.serialize_element(&id)?;
        }
        seq.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check(actual: &Members, expected: &BTreeSet<i64>) {
        assert_eq!(actual.len(), expected.len());
        assert_eq!(
            actual.iter().collect::<Vec<_>>(),
            expected.iter().copied().collect::<Vec<_>>()
        );
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
        for id in [-999, -65, -1, 0, 1, 63, 64, 65, 129, 999] {
            assert_eq!(actual.contains(&id), expected.contains(&id));
        }
    }
    #[test]
    fn compact_members_preserve_union_removal_and_shared_ownership() {
        let domain = Domain::new((-80..150).collect());
        let other_domain = Domain::new((-100..200).collect());
        let a = (-70..140).filter(|i| i % 3 != 0).collect::<BTreeSet<_>>();
        let b = (-80..150).filter(|i| i % 5 == 0).collect::<BTreeSet<_>>();
        let lefts = [
            Members::from(a.clone()),
            Members::from_domain(Arc::clone(&domain), &a),
        ];
        let rights = [
            Members::from(b.clone()),
            Members::from_domain(Arc::clone(&domain), &b),
            Members::from_domain(other_domain, &b),
            Members::from(BTreeSet::from([i64::MIN, 999])),
        ];
        for left in &lefts {
            for right in &rights {
                let mut actual = left.clone();
                let mut expected = a
                    .iter()
                    .copied()
                    .chain(right.iter())
                    .collect::<BTreeSet<_>>();
                actual.union_with(right);
                actual.union_with(right);
                check(&actual, &expected);
                for id in [-80, -65, 0, 64, 128, 999, i64::MIN] {
                    actual.remove(&id);
                    actual.remove(&id);
                    expected.remove(&id);
                    check(&actual, &expected);
                }
                check(left, &a);
            }
        }
        let empty = Members::from_domain(Domain::new(BTreeSet::new()), &BTreeSet::new());
        check(&empty, &BTreeSet::new());
        let mut same = lefts[1].clone();
        same.union_with(&lefts[1]);
        check(&same, &a);
    }
}
