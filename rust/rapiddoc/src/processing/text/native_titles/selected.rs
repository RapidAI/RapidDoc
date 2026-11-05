//! Bounded CPython integer-set add/discard/list order for lane expansion anchors.
//! Uses the same PSF-derived probing policy as table_native_vector/python_set.rs.
//! Only nonnegative row positions below 4096 are admitted by the parent adapter.
pub(super) struct Selected {
    slots: Vec<Option<usize>>,
    used: usize,
    fill: usize,
}
impl Selected {
    pub(super) fn new() -> Self {
        Self {
            slots: vec![None; 8],
            used: 0,
            fill: 0,
        }
    }
    fn locate(&self, key: usize) -> (bool, usize) {
        let mask = self.slots.len() - 1;
        let (mut at, mut perturb) = (key & mask, key);
        let mut free = None;
        loop {
            let probes = if at + 9 <= mask { 9 } else { 0 };
            for i in at..=at + probes {
                match self.slots[i] {
                    Some(k) if k == key => return (true, i),
                    Some(usize::MAX) => free = Some(i),
                    None => return (false, free.unwrap_or(i)),
                    _ => {}
                }
            }
            perturb >>= 5;
            at = (at * 5 + 1 + perturb) & mask;
        }
    }
    pub(super) fn insert(&mut self, key: usize) {
        let (found, i) = self.locate(key);
        if found {
            return;
        }
        let empty = self.slots[i].is_none();
        self.slots[i] = Some(key);
        self.used += 1;
        if !empty {
            return;
        }
        self.fill += 1;
        if self.fill * 5 >= (self.slots.len() - 1) * 3 {
            let mut size = 8;
            while size <= self.used * 4 {
                size *= 2
            }
            let old = std::mem::replace(&mut self.slots, vec![None; size]);
            for key in old.into_iter().flatten().filter(|k| *k != usize::MAX) {
                let (_, i) = self.locate(key);
                self.slots[i] = Some(key)
            }
            self.fill = self.used
        }
    }
    pub(super) fn remove(&mut self, key: &usize) {
        let (found, i) = self.locate(*key);
        if found {
            self.slots[i] = Some(usize::MAX);
            self.used -= 1
        }
    }
    pub(super) fn contains(&self, key: &usize) -> bool {
        self.locate(*key).0
    }
    pub(super) fn ordered(&self) -> Vec<usize> {
        self.slots
            .iter()
            .flatten()
            .copied()
            .filter(|k| *k != usize::MAX)
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_title_anchor_set_order() {
        let data: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-title-contracts.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in data["set_traces"].as_array().unwrap() {
            let mut s = Selected::new();
            for step in c["steps"].as_array().unwrap() {
                let key = step["key"].as_u64().unwrap() as usize;
                if step["operation"] == "add" {
                    s.insert(key)
                } else {
                    s.remove(&key)
                }
                assert_eq!(
                    serde_json::json!(s.ordered()),
                    step["order"],
                    "{}",
                    c["label"]
                );
            }
        }
    }
}
