//! CPython3.14整数set的有界插入/遍历语义，用于物理线等距与原始字符来源顺序。
//! 参考Objects/setobject.c（PSF license），见licenses/PYTHON-SET.txt。
//! 原索引入口0..5000且互异；有符号来源入口abs<=1e12且允许重复，均无删除。
//! 64位整数hash除-1改为-2外即自身；不能替代通用Python集合。
fn insert(table: &mut [Option<usize>], key: usize) {
    let mask = table.len() - 1;
    let (mut at, mut perturb) = (key & mask, key);
    loop {
        let probes = if at + 9 <= mask { 9 } else { 0 };
        for slot in table.iter_mut().skip(at).take(probes + 1) {
            if slot.is_none() {
                *slot = Some(key);
                return;
            }
        }
        perturb >>= 5;
        at = (at * 5 + 1 + perturb) & mask;
    }
}
pub(super) fn ordered(indices: &[usize]) -> Vec<usize> {
    let mut table = vec![None; 8];
    for (i, key) in indices.iter().enumerate() {
        insert(&mut table, *key);
        let used = i + 1;
        if used * 5 >= (table.len() - 1) * 3 {
            let mut size = 8;
            while size <= used * 4 {
                size *= 2
            }
            let old = std::mem::replace(&mut table, vec![None; size]);
            for key in old.into_iter().flatten() {
                insert(&mut table, key)
            }
        }
    }
    table.into_iter().flatten().collect()
}
/// 有界有符号整数来源ID（abs<=1e12），允许重复输入；保留CPython64位hash与首次插入顺序。
pub(in crate::processing) fn ordered_signed(values: &[i64]) -> Vec<i64> {
    fn insert(table: &mut [Option<i64>], key: i64) -> bool {
        let hash = if key == -1 { -2 } else { key } as usize;
        let mask = table.len() - 1;
        let (mut at, mut perturb) = (hash & mask, hash);
        loop {
            let probes = if at + 9 <= mask { 9 } else { 0 };
            for slot in table.iter_mut().skip(at).take(probes + 1) {
                if *slot == Some(key) {
                    return false;
                }
                if slot.is_none() {
                    *slot = Some(key);
                    return true;
                }
            }
            perturb >>= 5;
            at = (at.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
    }
    let mut table = vec![None; 8];
    let mut used = 0;
    for key in values {
        if !insert(&mut table, *key) {
            continue;
        }
        used += 1;
        if used * 5 >= (table.len() - 1) * 3 {
            let mut size = 8;
            while size <= used * 4 {
                size *= 2;
            }
            let old = std::mem::replace(&mut table, vec![None; size]);
            for key in old.into_iter().flatten() {
                insert(&mut table, key);
            }
        }
    }
    table.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn frozen_cpython_integer_set_iteration() {
        let f: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-table-track-ties.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in f["sets"].as_array().unwrap() {
            let input: Vec<usize> = serde_json::from_value(c["input"].clone()).unwrap();
            let expected: Vec<usize> = serde_json::from_value(c["expected"].clone()).unwrap();
            assert_eq!(super::ordered(&input), expected, "{input:?}");
        }
    }
}
