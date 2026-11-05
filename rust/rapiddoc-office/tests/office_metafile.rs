use rapiddoc_office::metafile::{
    self,
    binary::Records,
    geometry,
    header::{self, Source},
    model::*,
    playback::Playback,
};
use serde_json::{json, Value};
use std::io::Read;
fn fixture() -> Value {
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(
        &include_bytes!(
            "../../../tests/fixtures/office_rust_reference_068/metafile_geometry.json.gz"
        )[..],
    )
    .read_to_end(&mut bytes)
    .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
fn equal(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let a = a.as_f64().unwrap();
            let e = e.as_f64().unwrap();
            assert!(
                (a - e).abs() <= 1e-12 * e.abs().max(1.),
                "{path}: {a} != {e}"
            );
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{path} length");
            for (i, (a, e)) in a.iter().zip(e).enumerate() {
                equal(a, e, &format!("{path}[{i}]"));
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            assert_eq!(a.len(), e.len(), "{path} keys");
            for (key, e) in e {
                equal(
                    a.get(key).unwrap_or_else(|| panic!("{path} missing {key}")),
                    e,
                    &format!("{path}.{key}"),
                );
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}
#[test]
fn original_metafile_headers_records_and_gdi_replay() {
    for case in fixture()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let bytes: Vec<_> = case["hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
            .collect();
        let size = case["size"]
            .as_array()
            .map(|s| [s[0].as_u64().unwrap(), s[1].as_u64().unwrap()]);
        let prepared = header::prepare(&bytes, case["dpi"].as_u64().unwrap() as u32, size);
        if let Some(error) = case.get("header_error") {
            assert_eq!(
                prepared.unwrap_err().kind.code(),
                error.as_str().unwrap(),
                "{name}"
            );
            continue;
        }
        let header = prepared.unwrap_or_else(|e| panic!("{name}: {e}"));
        equal(
            &serde_json::to_value(&header).unwrap(),
            &case["header"],
            name,
        );
        let records = Records::new(
            &bytes,
            header.record_start,
            header.record_end,
            header.source_format,
        )
        .unwrap()
        .collect::<rapiddoc_office::Result<Vec<_>>>();
        if let Some(error) = case.get("records_error") {
            assert_eq!(
                records.unwrap_err().kind.code(),
                error.as_str().unwrap(),
                "{name} records"
            );
        } else {
            let records = records.unwrap();
            equal(&json!(records.iter().map(|r| json!({"kind":r.kind,"index":r.index,"offset":r.offset,"size":r.data.data.len()})).collect::<Vec<_>>()),&case["records"],name);
        }
        if case["emfplus"] == true {
            let result = metafile::emf::replay(&bytes, header, size);
            if let Some(error) = case.get("replay_error") {
                assert_eq!(
                    result.err().map(|e| e.kind.code()),
                    error.as_str(),
                    "{name}"
                );
            } else {
                equal(
                    &serde_json::to_value(result.unwrap()).unwrap(),
                    &case["document"],
                    name,
                );
            }
            continue;
        }
        let mut p = Playback::new(header.clone());
        let mut trace = Vec::new();
        let mut cursor = 0;
        let result = (|| {
            for record in Records::new(
                &bytes,
                header.record_start,
                header.record_end,
                header.source_format,
            )? {
                let record = record?;
                p.set_record(record.kind, record.index, record.offset);
                match header.source_format {
                    Source::Emf => metafile::emf::handle(record.kind, record.data, &mut p)?,
                    Source::Wmf => metafile::wmf::handle(
                        record.kind,
                        record.data.sub(6, record.data.data.len() - 6)?,
                        &mut p,
                    )?,
                }
                trace.push(json!({"state":p.state,"path":p.path_builder.build(),"current":p.path_builder.current,"start":p.path_builder.start,"open":p.path_builder.figure_open(),"active":p.path_active,"ready":p.path_ready,"budget":p.budget,"new_commands":p.commands[cursor..],"diagnostics":p.diagnostics(),"partial":p.partial}));
                cursor = p.commands.len();
            }
            p.finish(size)
        })();
        equal(&json!(trace), &case["trace"], name);
        if let Some(error) = case.get("replay_error") {
            assert!(
                matches!(result,Err(ref e) if e.kind.code()==error.as_str().unwrap()),
                "{name} replay error"
            );
        } else {
            equal(
                &serde_json::to_value(result.unwrap_or_else(|e| panic!("{name}: {e}"))).unwrap(),
                &case["document"],
                name,
            );
        }
    }
}
#[test]
fn original_paths_and_flattening() {
    for case in fixture()["geometry"].as_array().unwrap() {
        let c = case["rect"].as_array().unwrap();
        let r = Rect {
            left: c[0].as_f64().unwrap(),
            top: c[1].as_f64().unwrap(),
            right: c[2].as_f64().unwrap(),
            bottom: c[3].as_f64().unwrap(),
        };
        let path = match case["kind"].as_str().unwrap() {
            "rectangle" => geometry::rectangle(r),
            "ellipse" => geometry::ellipse(r),
            "round_rectangle" => geometry::round_rectangle(r, 40., 15.),
            "arc" => geometry::arc(
                r,
                [100., 0.],
                [0., 60.],
                case["direction"].as_u64().unwrap() as u32,
                "pie",
            ),
            _ => unreachable!(),
        };
        let name = case["kind"].as_str().unwrap();
        equal(&json!(path), &case["path"], name);
        let mut budget = geometry::FlattenBudget::default();
        equal(
            &json!(
                geometry::flatten(&path, case["flatness"].as_f64().unwrap(), 12, &mut budget)
                    .unwrap()
            ),
            &case["flat"],
            name,
        );
        assert_eq!(budget.used as u64, case["used"].as_u64().unwrap());
        equal(&json!(geometry::bounds(&path)), &case["bounds"], name);
        equal(&json!(geometry::close_open(&path)), &case["closed"], name);
        equal(
            &json!(geometry::transform(
                &path,
                Matrix {
                    a: 1.,
                    b: 0.25,
                    c: -0.5,
                    d: 2.,
                    e: 7.,
                    f: -9.
                }
            )),
            &case["transformed"],
            name,
        );
    }
}
#[test]
fn original_canvas_sizing() {
    for case in fixture()["sizes"].as_array().unwrap() {
        let size = [
            case["size"][0].as_u64().unwrap(),
            case["size"][1].as_u64().unwrap(),
        ];
        let actual = header::bounded_canvas(size);
        if let Some(error) = case.get("error") {
            assert_eq!(actual.unwrap_err().kind.code(), error.as_str().unwrap());
        } else {
            let (size, scaled) = actual.unwrap();
            equal(
                &json!([size[0], size[1], scaled]),
                &case["bounded"],
                "canvas",
            );
        }
    }
}
