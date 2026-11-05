use rapiddoc::{sdk::input, ParseOptions, RapidDoc};
use std::{
    io::{Read, Write},
    net::TcpListener,
    thread,
};

fn serve(bodies: Vec<(String, Vec<u8>)>) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let task = thread::spawn(move || {
        for (status, body) in bodies {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                if stream.read(&mut byte).unwrap() == 0 {
                    break;
                }
                request.push(byte[0]);
            }
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(&body).unwrap();
        }
    });
    (address, task)
}

#[test]
fn url_filename_is_decoded_and_never_becomes_a_directory() {
    assert_eq!(
        input::source_name("https://host/download?filename=%E8%AE%BA%E6%96%87.pdf#page=1"),
        "论文.pdf"
    );
    assert_eq!(
        input::source_name("https://host/download?filename=../../report.docx"),
        "report.docx"
    );
    assert_eq!(
        input::source_name("https://host/a%20b.pdf?token=secret"),
        "a b.pdf"
    );
    assert_eq!(input::source_name("https://host/"), "document");
}

#[test]
fn redirect_url_and_local_office_have_identical_content() {
    let body = include_bytes!("../../../tests/migration/inputs/synthetic.docx").to_vec();
    let (url, task) = serve(vec![
        ("302 Found\r\nLocation: /data".into(), vec![]),
        ("200 OK".into(), body.clone()),
    ]);
    let engine = RapidDoc::default();
    let local = engine
        .parse_bytes(&body, &ParseOptions::default(), &mut |_| Ok(()))
        .unwrap();
    let mut events = vec![];
    let remote = engine
        .parse_with_progress(
            format!("{url}/download?filename=report.docx"),
            &ParseOptions::default(),
            &mut |event| {
                events.push(event.stage.to_owned());
                Ok(())
            },
        )
        .unwrap();
    task.join().unwrap();
    assert_eq!(remote.markdown, local.markdown);
    assert_eq!(remote.content_list_json, local.content_list_json);
    assert_eq!(remote.metadata["source_name"], "report.docx");
    assert_eq!(events.first().unwrap(), "source_download_started");
    assert_eq!(events.last().unwrap(), "source_download_completed");
}

#[test]
fn downloads_over_ten_megabytes_are_not_subject_to_ureq_convenience_limit() {
    let size = 11 * 1024 * 1024;
    let (url, task) = serve(vec![("200 OK".into(), vec![42; size])]);
    let (_, data) = input::read_url(&format!("{url}/large.pdf"), &mut |_| Ok(())).unwrap();
    task.join().unwrap();
    assert_eq!(data.len(), size);
    assert!(data.iter().all(|v| *v == 42));
}

#[test]
fn http_errors_and_callback_cancellation_are_reported() {
    let (url, task) = serve(vec![("404 Not Found".into(), vec![])]);
    let error = input::read_url(&url, &mut |_| Ok(())).unwrap_err();
    task.join().unwrap();
    assert!(error.message.contains("404"));
    let cancelled = input::read_url("http://127.0.0.1:9/no-connection", &mut |_| {
        Err(rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: "cancelled by caller".into(),
        })
    })
    .unwrap_err();
    assert_eq!(cancelled.message, "cancelled by caller");
}
