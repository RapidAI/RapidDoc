use serde_json::{json, Value};
use std::io::Read;
#[test]
fn original_office_raster_modes_and_pixels() {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/office_raster.json.gz")
            [..],
    )
    .read_to_end(&mut raw)
    .unwrap();
    let f: Value = serde_json::from_slice(&raw).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let bytes = c["hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let result = rapiddoc_office::image::serialize(&bytes, None, None);
        if c["expected"].get("error").is_some() {
            assert!(result.is_err(), "{}", c["name"]);
            continue;
        }
        let source = result.unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        let actual=source.map(|s|{let(mime,data)=s.split_once(',').unwrap();let data=base64::decode(data).unwrap();let (size,pixels)=if mime.starts_with("data:image/png;"){let mut reader=png::Decoder::new(std::io::Cursor::new(data)).read_info().unwrap();let mut raw=vec![0;reader.output_buffer_size().unwrap()];let info=reader.next_frame(&mut raw).unwrap();raw.truncate(info.buffer_size());([info.width as usize,info.height as usize],raw)}else{let i=turbojpeg::decompress(&data,turbojpeg::PixelFormat::RGBA).unwrap();([i.width,i.height],i.pixels)};json!({"mime":mime.trim_start_matches("data:").split(';').next().unwrap(),"size":size,"rgba":pixels})});
        if json!(actual) != c["expected"] {
            let a = json!(actual);
            panic!("{} mismatch: actual mime={} size={}, expected mime={} size={}; first differing pixel {:?}",c["name"],a["mime"],a["size"],c["expected"]["mime"],c["expected"]["size"],a["rgba"].as_array().zip(c["expected"]["rgba"].as_array()).and_then(|(a,b)|a.iter().zip(b).position(|(a,b)|a!=b)));
        }
    }
}
