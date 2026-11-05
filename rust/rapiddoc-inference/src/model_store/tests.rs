use super::*;

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let dir = env::temp_dir().join(format!(
            "rapiddoc-model-store-{}-{}",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn catalog() -> Catalog {
    let mut c = Catalog::bundled().unwrap();
    let model = c.models.get_mut("rapid_orientation").unwrap();
    model.sha256 = format!("{:x}", Sha256::digest(b"weights"));
    c
}

#[test]
fn defaults_and_variants_are_separate() {
    let c = Catalog::bundled().unwrap();
    assert_eq!(c.models.len(), 15);
    assert_eq!(c.model("layout").unwrap().0, "pp_doclayoutv3");
    assert_eq!(c.model("formula").unwrap().0, "pp_formulanet_plus_m");
    assert_eq!(c.model("table").unwrap().0, "slanet_plus");
    assert!(c.model("table_transformer_v1_1_all").is_ok());
    for model in c.models.values() {
        assert!(model.url.starts_with("https://www.modelscope.cn/"));
    }
}

#[test]
fn catalog_rejects_invalid_entries() {
    let c = catalog();
    for filename in ["../model", "a/b", "a\\b", "C:model", ".."] {
        let mut invalid = c.clone();
        invalid
            .models
            .get_mut("rapid_orientation")
            .unwrap()
            .filename = filename.into();
        assert!(Catalog::from_json(&serde_json::to_string(&invalid).unwrap()).is_err());
    }
    let mut invalid = c.clone();
    invalid.defaults.insert("layout".into(), "missing".into());
    assert!(Catalog::from_json(&serde_json::to_string(&invalid).unwrap()).is_err());
}

#[test]
fn download_is_verified_published_and_reused() {
    let d = Directory::new();
    let c = catalog();
    let first = c
        .resolve_with("orientation", Some(&d.0), None, true, |_, out| {
            out.write_all(b"weights")?;
            Ok(())
        })
        .unwrap();
    assert!(first.downloaded);
    assert_eq!(fs::read(&first.path).unwrap(), b"weights");
    let second = c
        .resolve_with("rapid_orientation", Some(&d.0), None, true, |_, _| {
            panic!("must reuse")
        })
        .unwrap();
    assert!(!second.downloaded);
    assert_eq!(first.path, second.path);
    assert_eq!(fs::read_dir(&d.0).unwrap().count(), 1);
}

#[test]
fn streaming_download_events_include_size_verification_and_cache_reuse() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0; 4096];
        stream.read(&mut request).unwrap();
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\nweights")
            .unwrap();
    });
    let d = Directory::new();
    let mut c = catalog();
    c.models.get_mut("rapid_orientation").unwrap().url = format!("http://{address}/model");
    let mut events = vec![];
    let model = c
        .resolve_progress("orientation", Some(&d.0), None, true, &mut |event| {
            events.push(serde_json::to_value(event).unwrap());
            Ok(())
        })
        .unwrap();
    server.join().unwrap();
    assert_eq!(
        events
            .iter()
            .map(|e| e["stage"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "download_started",
            "download_progress",
            "download_verifying",
            "download_completed"
        ]
    );
    assert_eq!(events[1]["total"], 7);
    assert_eq!(events.last().unwrap()["completed"], 7);
    assert_eq!(events.last().unwrap()["path"], model.path.to_str().unwrap());
    c.resolve_progress("orientation", Some(&d.0), None, true, &mut |_| {
        panic!("Cached model must not emit download events")
    })
    .unwrap();
}

#[test]
fn failed_download_emits_error_and_cancellation_cleans_partial() {
    for cancel in [false, true] {
        let d = Directory::new();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener); // refused connection, no dependency on external services
        let mut c = catalog();
        c.models.get_mut("rapid_orientation").unwrap().url = format!("http://{address}/model");
        let mut events = vec![];
        let result = c.resolve_progress("orientation", Some(&d.0), None, true, &mut |event| {
            events.push(event.stage);
            if cancel {
                Err(StoreError::new(StoreErrorKind::Io, "cancelled"))
            } else {
                Ok(())
            }
        });
        assert!(result.is_err());
        assert_eq!(fs::read_dir(&d.0).unwrap().count(), 0);
        if cancel {
            assert_eq!(events, ["download_started"]);
        } else {
            assert_eq!(events, ["download_started", "download_failed"]);
        }
    }
}

#[test]
fn unknown_download_size_and_failed_checksum_do_not_report_success() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0; 4096];
        stream.read(&mut request).unwrap();
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nbroken")
            .unwrap();
    });
    let d = Directory::new();
    let mut c = catalog();
    c.models.get_mut("rapid_orientation").unwrap().url = format!("http://{address}/model");
    let mut events = vec![];
    let result = c.resolve_progress("orientation", Some(&d.0), None, true, &mut |event| {
        events.push(serde_json::to_value(event).unwrap());
        Ok(())
    });
    server.join().unwrap();
    assert_eq!(result.unwrap_err().kind, StoreErrorKind::Integrity);
    assert_eq!(events[1]["total"], serde_json::Value::Null);
    assert_eq!(events.last().unwrap()["stage"], "download_failed");
    assert!(events.last().unwrap()["error"].is_string());
    assert!(!events
        .iter()
        .any(|event| event["stage"] == "download_completed"));
    assert_eq!(fs::read_dir(&d.0).unwrap().count(), 0);
}

#[test]
fn failures_leave_no_partial_or_final_model() {
    for interrupted in [false, true] {
        let d = Directory::new();
        let c = catalog();
        let result = c.resolve_with("orientation", Some(&d.0), None, true, |_, out| {
            out.write_all(b"broken")?;
            if interrupted {
                return Err(StoreError::new(StoreErrorKind::Http, "interrupted"));
            }
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(fs::read_dir(&d.0).unwrap().count(), 0);
    }
}

#[test]
fn nested_model_custom_override_and_corruption() {
    let d = Directory::new();
    let c = catalog();
    let nested = d.0.join("orientation");
    fs::create_dir(&nested).unwrap();
    let path = nested.join("rapid_orientation.onnx");
    fs::write(&path, b"weights").unwrap();
    assert_eq!(
        c.resolve("orientation", Some(&d.0), None, false)
            .unwrap()
            .path,
        path
    );
    fs::write(&path, b"custom").unwrap();
    assert_eq!(
        c.resolve("orientation", Some(&d.0), None, false)
            .unwrap_err()
            .kind,
        StoreErrorKind::Integrity
    );
    let custom = c
        .resolve("orientation", Some(&d.0), Some(&path), false)
        .unwrap();
    assert_eq!(custom.sha256, digest(&path).unwrap());
    assert!(!custom.downloaded);
}

#[test]
fn offline_missing_and_unknown_model() {
    let d = Directory::new();
    let c = catalog();
    assert_eq!(
        c.resolve("orientation", Some(&d.0), None, false)
            .unwrap_err()
            .kind,
        StoreErrorKind::NotFound
    );
    assert_eq!(
        c.resolve("unknown", Some(&d.0), None, true)
            .unwrap_err()
            .kind,
        StoreErrorKind::InvalidInput
    );
}

#[test]
fn concurrent_valid_publication_is_reused() {
    let d = Directory::new();
    let c = catalog();
    let result = c
        .resolve_with("orientation", Some(&d.0), None, true, |_, out| {
            out.write_all(b"weights")?;
            fs::write(d.0.join("rapid_orientation.onnx"), b"weights")?;
            Ok(())
        })
        .unwrap();
    assert_eq!(fs::read(result.path).unwrap(), b"weights");
    assert_eq!(fs::read_dir(&d.0).unwrap().count(), 1);
}
