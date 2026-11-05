//! Development-only: measure existing Value -> Value serialization vs ownership.
//! This is a microbenchmark, not a parser performance result.
use serde_json::{json, Value};
use std::{hint::black_box, time::Instant};

fn main() {
    let path = std::env::args().nth(1).expect("input JSON");
    let source: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let expected = json!({"source": &source});
    let mut direct = json!({});
    direct["source"] = source.clone();
    assert_eq!(
        serde_json::to_vec(&expected).unwrap(),
        serde_json::to_vec(&direct).unwrap()
    );
    let mut rows = Vec::new();
    for trial in 0..4 {
        let methods = if trial % 2 == 0 {
            ["serialize", "clone"]
        } else {
            ["clone", "serialize"]
        };
        for method in methods {
            let start = Instant::now();
            for _ in 0..1000 {
                let output = if method == "serialize" {
                    json!({"source": black_box(&source)})
                } else {
                    let mut output = json!({});
                    output["source"] = black_box(&source).clone();
                    output
                };
                black_box(output);
            }
            rows.push(json!({"trial":trial,"method":method,"iterations":1000,"seconds":start.elapsed().as_secs_f64()}));
        }
    }
    println!(
        "{}",
        json!({"exact_serialized_output_equal":true,"rows":rows})
    );
}
