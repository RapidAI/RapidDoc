//! Explicit opt-in document budgets; production defaults match Python without artificial caps.
use std::{cell::Cell, marker::PhantomData, rc::Rc};
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Limits {
    pub work: usize,
    pub nodes: usize,
    pub bytes: usize,
    pub output: usize,
    pub candidate_members: usize,
    pub host: bool,
}
const DEFAULT: Limits = Limits {
    work: 10_000_000,
    nodes: 200_000,
    bytes: 2_000_000,
    output: 8_000_000,
    candidate_members: 20000,
    host: false,
};
thread_local! {static ACTIVE:Cell<Limits>=const{Cell::new(DEFAULT)};}
#[inline]
pub(crate) fn current() -> Limits {
    ACTIVE.with(Cell::get)
}
/// Host pages carry both the flat character list and the original grouped tree.
/// Keep diagnostic limits fixed; never truncate those two representations.
pub(crate) fn host_page() -> Scope {
    let compatibility = rapiddoc_core::resources::python_compatible();
    let previous = ACTIVE.with(|v| {
        v.replace(apply(Limits {
            work: usize::MAX,
            nodes: usize::MAX,
            bytes: usize::MAX,
            output: usize::MAX,
            candidate_members: usize::MAX,
            host: true,
        }))
    });
    Scope {
        previous,
        _thread: PhantomData,
        _compatibility: compatibility,
    }
}
pub(crate) struct Scope {
    previous: Limits,
    _thread: PhantomData<Rc<()>>,
    _compatibility: rapiddoc_core::resources::Scope,
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|v| v.set(self.previous))
    }
}
pub(crate) fn document(_pages: usize) -> Scope {
    host_page()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_document_budget_scope() {
        assert_eq!(current(), DEFAULT);
        {
            let _outer = document(14);
            assert_eq!(current().work, usize::MAX);
            assert_eq!(current().candidate_members, usize::MAX);
            {
                let _inner = document(2);
                assert_eq!(current().output, usize::MAX)
            }
            assert_eq!(current().output, usize::MAX);
            assert_eq!(std::thread::spawn(current).join().unwrap(), DEFAULT);
        }
        assert_eq!(current(), DEFAULT);
        let _ = std::panic::catch_unwind(|| {
            let _guard = document(999);
            assert_eq!(current().work, usize::MAX);
            panic!("deliberate scope unwind")
        });
        assert_eq!(current(), DEFAULT);
        assert!(crate::legacy_page::Work(9_999_999).charge(2).is_err());
    }
}

pub(crate) fn records(diagnostic: usize) -> usize {
    if current().host {
        usize::MAX
    } else {
        diagnostic
    }
}

/// Count encoded bytes only when an output budget exists. Never allocate a
/// second serialized copy of a document just to check an unlimited budget.
pub(crate) fn output_exceeds(value: &serde_json::Value) -> bool {
    output_exceeds_limit(value, current().output)
}
pub(crate) fn output_exceeds_limit(value: &serde_json::Value, maximum: usize) -> bool {
    if maximum == usize::MAX {
        return false;
    }
    struct Counter {
        bytes: usize,
        maximum: usize,
    }
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.bytes = self.bytes.saturating_add(bytes.len());
            if self.bytes > self.maximum {
                return Err(std::io::Error::other("output budget"));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer(Counter { bytes: 0, maximum }, value).is_err()
}

/// Optional limits on actual resources, never on extracted character counts.
#[derive(Clone, Copy, Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostResources {
    pub max_work: Option<usize>,
    pub max_json_nodes: Option<usize>,
    pub max_json_bytes: Option<usize>,
    pub max_output_bytes: Option<usize>,
    pub max_candidate_members: Option<usize>,
}
thread_local! {static RESOURCES:Cell<HostResources>=const{Cell::new(HostResources{max_work:None,max_json_nodes:None,max_json_bytes:None,max_output_bytes:None,max_candidate_members:None})};}
fn apply(mut limits: Limits) -> Limits {
    RESOURCES.with(|v| {
        let r = v.get();
        if let Some(n) = r.max_work {
            limits.work = n
        }
        if let Some(n) = r.max_json_nodes {
            limits.nodes = n
        }
        if let Some(n) = r.max_json_bytes {
            limits.bytes = n
        }
        if let Some(n) = r.max_output_bytes {
            limits.output = n
        }
        if let Some(n) = r.max_candidate_members {
            limits.candidate_members = n
        }
    });
    limits
}
pub fn with_resources<T>(resources: HostResources, operation: impl FnOnce() -> T) -> T {
    struct Restore(HostResources);
    impl Drop for Restore {
        fn drop(&mut self) {
            RESOURCES.with(|v| v.set(self.0));
        }
    }
    let _restore = Restore(RESOURCES.with(|v| v.replace(resources)));
    operation()
}

#[cfg(test)]
mod resource_tests {
    use super::*;
    #[test]
    fn borrowed_request_has_identical_finite_tree_budget() {
        let source = serde_json::json!({"a":["汉字",2]});
        let options = serde_json::json!({"em":4,"extra":["x"]});
        let mut original = options.clone();
        original["source"] = source.clone();
        for nodes in 0..12 {
            for bytes in 0..32 {
                with_resources(
                    HostResources {
                        max_json_nodes: Some(nodes),
                        max_json_bytes: Some(bytes),
                        ..Default::default()
                    },
                    || {
                        let _scope = host_page();
                        assert_eq!(
                            crate::legacy_page::validate_tree(&original).is_err(),
                            crate::legacy_page::validate_source_request(&source, &options).is_err(),
                            "nodes={nodes} bytes={bytes}"
                        );
                    },
                );
            }
        }
    }
    #[test]
    fn tree_scan_preserves_explicit_node_and_byte_bounds() {
        let value = serde_json::json!({"s":"汉字"}); // two nodes, seven UTF-8 bytes
        for (nodes, bytes, rejects) in [(2, 7, false), (1, 7, true), (2, 6, true)] {
            with_resources(
                HostResources {
                    max_json_nodes: Some(nodes),
                    max_json_bytes: Some(bytes),
                    ..Default::default()
                },
                || {
                    let _scope = host_page();
                    assert_eq!(crate::legacy_page::validate_tree(&value).is_err(), rejects);
                },
            );
        }
        let mut nested = serde_json::Value::Null;
        for _ in 0..66 {
            nested = serde_json::json!([nested]);
        }
        assert!(crate::legacy_page::validate_tree(&nested).is_err());
        let _scope = host_page();
        assert!(crate::legacy_page::validate_tree(&nested).is_ok());
    }
    #[test]
    fn output_byte_count_preserves_exact_finite_limits() {
        let value = serde_json::json!({"s":"汉字\n\"\\", "a":[1, 2.5, null, true]});
        let bytes = value.to_string().len();
        for (maximum, exceeds) in [(bytes - 1, true), (bytes, false), (bytes + 1, false)] {
            assert_eq!(output_exceeds_limit(&value, maximum), exceeds);
            with_resources(
                HostResources {
                    max_output_bytes: Some(maximum),
                    ..Default::default()
                },
                || {
                    let _scope = host_page();
                    assert_eq!(output_exceeds(&value), exceeds);
                },
            );
        }
        let _scope = host_page();
        assert!(!output_exceeds(&value));
        assert!(!output_exceeds_limit(&value, usize::MAX));
    }
    #[test]
    fn character_count_is_not_a_host_budget() {
        assert_eq!(records(3000), 3000);
        {
            let _scope = host_page();
            assert_eq!(records(3000), usize::MAX);
            assert_eq!(records(20000), usize::MAX);
        }
        with_resources(
            HostResources {
                max_json_nodes: Some(20),
                max_json_bytes: Some(1024),
                max_work: Some(5),
                ..Default::default()
            },
            || {
                let _scope = host_page();
                assert!(current().host);
                assert_eq!(records(3000), usize::MAX);
                assert_eq!(current().nodes, 20);
                assert!(crate::legacy_page::Work(4).charge(2).is_err());
                {
                    let _nested = document(10);
                    assert_eq!(current().bytes, 1024);
                    assert_eq!(current().work, 5);
                }
                assert_eq!(
                    std::thread::spawn(|| {
                        let _scope = host_page();
                        current().nodes
                    })
                    .join()
                    .unwrap(),
                    usize::MAX
                );
            },
        );
        assert_eq!(current(), DEFAULT);
        let _ = std::panic::catch_unwind(|| {
            with_resources(
                HostResources {
                    max_work: Some(1),
                    ..Default::default()
                },
                || {
                    let _scope = host_page();
                    panic!("resource scope unwind")
                },
            )
        });
        {
            let _scope = host_page();
            assert_eq!(current().work, usize::MAX);
        }
    }
}

#[cfg(test)]
mod host_tests {
    use super::*;
    #[test]
    fn actual_dense_host_page_and_restore() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let q: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("build/dense-native-ocr-failure.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(q["native_page"]["chars"].as_array().unwrap().len(), 20198);
        assert!(crate::legacy_model::native_ocr::plan(&q).is_err());
        {
            let _guard = host_page();
            let got = crate::legacy_model::native_ocr::plan(&q).unwrap();
            assert!(got["detect"].as_array().unwrap().is_empty());
            assert_eq!(std::thread::spawn(current).join().unwrap(), DEFAULT);
        }
        assert_eq!(current(), DEFAULT);
        let _ = std::panic::catch_unwind(|| {
            let _scope = host_page();
            panic!("intentional host unwind")
        });
        assert_eq!(current(), DEFAULT);
    }
}
