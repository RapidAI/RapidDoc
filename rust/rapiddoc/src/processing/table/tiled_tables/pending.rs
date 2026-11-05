//! CPython 64-bit set(range(n)), pop and difference_update(list) for n<=5000.
//! PSF-derived probing/deletion/resize policy, see PYTHON-SET.txt.
pub(in crate::processing) struct Pending {
    slots: Vec<Option<usize>>,
    used: usize,
    fill: usize,
    finger: usize,
}
impl Pending {
    /// CPython set comprehension over unique nonnegative indices, preserving sparse hashes.
    pub(in crate::processing) fn from_indices(keys: &[usize]) -> Self {
        let mut s = Self {
            slots: vec![None; 8],
            used: 0,
            fill: 0,
            finger: 0,
        };
        for &key in keys {
            Self::insert(&mut s.slots, key);
            s.used += 1;
            s.fill += 1;
            if s.fill * 5 >= (s.slots.len() - 1) * 3 {
                s.resize(s.used * 4)
            }
        }
        s
    }
    /// `set.remove`, without difference_update's separate compaction policy.
    pub(in crate::processing) fn remove(&mut self, key: usize) {
        let mask = self.slots.len() - 1;
        let (mut at, mut perturb) = (key & mask, key);
        loop {
            let probes = if at + 9 <= mask { 9 } else { 0 };
            for i in at..=at + probes {
                if self.slots[i] == Some(key) {
                    self.slots[i] = Some(usize::MAX);
                    self.used -= 1;
                    return;
                }
                if self.slots[i].is_none() {
                    return;
                }
            }
            perturb >>= 5;
            at = (at * 5 + 1 + perturb) & mask;
        }
    }
    pub(in crate::processing) fn capacity(&self) -> usize {
        self.slots.len()
    }
    fn insert(slots: &mut [Option<usize>], key: usize) {
        let mask = slots.len() - 1;
        let (mut at, mut perturb) = (key & mask, key);
        loop {
            let probes = if at + 9 <= mask { 9 } else { 0 };
            for slot in slots.iter_mut().skip(at).take(probes + 1) {
                if slot.is_none() {
                    *slot = Some(key);
                    return;
                }
            }
            perturb >>= 5;
            at = (at * 5 + 1 + perturb) & mask;
        }
    }
    fn resize(&mut self, min: usize) {
        let mut size = 8;
        while size <= min {
            size *= 2;
        }
        let old = std::mem::replace(&mut self.slots, vec![None; size]);
        for key in old.into_iter().flatten().filter(|k| *k != usize::MAX) {
            Self::insert(&mut self.slots, key);
        }
        self.fill = self.used;
    }
    pub(in crate::processing) fn new(n: usize) -> Self {
        let mut s = Self {
            slots: vec![None; 8],
            used: 0,
            fill: 0,
            finger: 0,
        };
        for key in 0..n {
            Self::insert(&mut s.slots, key);
            s.used += 1;
            s.fill += 1;
            if s.fill * 5 >= (s.slots.len() - 1) * 3 {
                s.resize(s.used * 4);
            }
        }
        s
    }
    pub(in crate::processing) fn ordered(&self) -> Vec<usize> {
        self.slots
            .iter()
            .flatten()
            .filter(|k| **k != usize::MAX)
            .copied()
            .collect()
    }
    pub(in crate::processing) fn pop(&mut self) -> Option<usize> {
        if self.used == 0 {
            return None;
        }
        let mask = self.slots.len() - 1;
        let mut at = self.finger & mask;
        loop {
            if let Some(k) = self.slots[at].filter(|k| *k != usize::MAX) {
                self.slots[at] = Some(usize::MAX);
                self.used -= 1;
                self.finger = at + 1;
                return Some(k);
            }
            at = (at + 1) & mask;
        }
    }
    pub(in crate::processing) fn difference(&mut self, keys: &[usize]) {
        for key in keys {
            let mask = self.slots.len() - 1;
            let (mut at, mut perturb) = (*key & mask, *key);
            'search: loop {
                let probes = if at + 9 <= mask { 9 } else { 0 };
                for i in at..=at + probes {
                    if self.slots[i] == Some(*key) {
                        self.slots[i] = Some(usize::MAX);
                        self.used -= 1;
                        break 'search;
                    }
                    if self.slots[i].is_none() {
                        break 'search;
                    }
                }
                perturb >>= 5;
                at = (at * 5 + 1 + perturb) & mask;
            }
        }
        if self.fill - self.used > (self.slots.len() - 1) / 4 {
            self.resize(self.used * 4);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn old_pop_difference_traces() {
        let f: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-tiled-tables.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for trace in f["set_traces"].as_array().unwrap() {
            let mut s = super::Pending::new(trace["count"].as_u64().unwrap() as usize);
            for step in trace["steps"].as_array().unwrap() {
                assert_eq!(s.pop(), Some(step["seed"].as_u64().unwrap() as usize));
                assert_eq!(serde_json::json!(s.ordered()), step["order"]);
                let remove: Vec<usize> = serde_json::from_value(step["remove"].clone()).unwrap();
                s.difference(&remove);
                assert_eq!(serde_json::json!(s.ordered()), step["remaining"]);
            }
            assert_eq!(s.pop(), None);
        }
    }
}
