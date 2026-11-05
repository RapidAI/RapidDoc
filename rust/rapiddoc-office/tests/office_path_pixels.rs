use rapiddoc_office::metafile::{
    geometry::FlattenBudget, image_session::ImageSession, mask::Mask, model::*, paths,
};
use serde_json::Value;
#[test]
fn original_docx_vector_parts_to_table_html() {
    use rapiddoc_office::docx::{table_html::Reader, table_images::Images, Source};
    let mut data = vec![];
    flate2::read::GzDecoder::new(&include_bytes!("../../../tests/fixtures/office_rust_reference_068/docx_vectors.json.gz")[..]).read_to_end(&mut data).unwrap();
    let f: Value = serde_json::from_slice(&data).unwrap();
    let pattern = regex::Regex::new(r"data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)").unwrap();
    for c in f["cases"].as_array().unwrap() {
        let bytes: Vec<_> = c["package_hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
            .collect();
        let mut source = Source::open(bytes.into()).unwrap();
        let tree = rapiddoc_office::xml::parse(&source.document).unwrap();
        let table = tree
            .descendants()
            .find(|n| n.is_element() && n.tag_name().name() == "tbl")
            .unwrap();
        let mut images = Images::new(&mut source.package, &source.relationships).unwrap();
        let image_node = table
            .descendants()
            .find(|n| n.is_element() && n.tag_name().name() == "imagedata")
            .unwrap();
        for _ in 0..2 {
            rapiddoc_office::docx::table_html::Resources::render(&mut images, image_node).unwrap();
        }
        if c["report"].is_null() {
            assert_eq!(images.vector_reports().len(), 0, "{}", c["name"]);
            assert_eq!(
                images.fallbacks().len(),
                1,
                "repeated corrupt part must reuse fallback"
            );
        } else {
            assert_eq!(
                images.vector_reports().len(),
                1,
                "repeated part must reuse rendering"
            );
            assert!(images.fallbacks().is_empty());
            let report = &images.vector_reports()[0];
            assert_eq!(
                serde_json::json!({"width":report.width,"height":report.height,"partial":report.partial,"diagnostics":report.diagnostics}),
                c["report"]
            );
        }
        let mut reader = Reader::new(
            source.styles.as_deref(),
            source.numbering.as_deref(),
            &source.relationships,
            images,
        )
        .unwrap();
        let html = reader
            .table_html(table)
            .unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        let mut images = vec![];
        let actual = pattern
            .replace_all(&html, |caps: &regex::Captures<'_>| {
                images.push(normalized_svg(base64::decode(&caps[1]).unwrap()));
                format!("data:image/svg+xml;base64,SVG[{}]", images.len() - 1)
            })
            .into_owned();
        assert_eq!(actual, c["html"].as_str().unwrap(), "{}", c["name"]);
        assert_eq!(
            serde_json::to_value(images).unwrap(),
            c["images"],
            "{}",
            c["name"]
        );
    }
}
fn normalized_svg(raw: Vec<u8>) -> Value {
    use sha2::{Digest, Sha256};
    let patterns = [
        regex::Regex::new(r"data:image/png;base64,([A-Za-z0-9+/=]+)").unwrap(),
        regex::Regex::new(r#"data-mime="image/png">([A-Za-z0-9+/=]+)"#).unwrap(),
    ];
    let mut markup = String::from_utf8(raw).unwrap();
    let mut images = vec![];
    for pattern in &patterns {
        markup=pattern.replace_all(&markup,|caps:&regex::Captures<'_>|{
                let encoded=&caps[1];let raw=base64::decode(encoded).unwrap();
                let mut decoder=png::Decoder::new(std::io::Cursor::new(raw));decoder.set_transformations(png::Transformations::EXPAND|png::Transformations::STRIP_16);
                let mut reader=decoder.read_info().unwrap();let mut pixels=vec![0;reader.output_buffer_size().unwrap()];let info=reader.next_frame(&mut pixels).unwrap();pixels.truncate(info.buffer_size());assert_eq!(info.color_type,png::ColorType::Rgba);
                let ppm=reader.info().pixel_dims.map(|d|vec![d.xppu,d.yppu]);
                images.push(serde_json::json!({"size":[info.width,info.height],"pixels_sha256":format!("{:x}",Sha256::digest(&pixels)),"ppm":ppm}));
                let prefix=&caps[0][..caps[0].len()-encoded.len()];format!("{prefix}PNG[{}]",images.len()-1)
            }).into_owned();
    }
    serde_json::json!({"markup":markup,"images":images})
}
#[test]
fn original_svg_structure_embedded_pixels_and_png_dpi() {
    use rapiddoc_office::metafile::{
        self,
        header::{self, Source},
        output,
    };
    let f = fixture();
    for c in f["numbers"].as_array().unwrap() {
        assert_eq!(
            output::number(c["value"].as_f64().unwrap()),
            c["output"].as_str().unwrap()
        );
    }
    for c in f["final"].as_array().unwrap() {
        let bytes: Vec<_> = c["hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
            .collect();
        let h = header::prepare(&bytes, 200, Some([24, 20])).unwrap();
        let doc = match h.source_format {
            Source::Wmf => metafile::wmf::replay(&bytes, h, Some([24, 20])),
            Source::Emf => metafile::emf::replay(&bytes, h, Some([24, 20])),
        }
        .unwrap();
        let raw = output::svg(
            &doc,
            &mut FlattenBudget::default(),
            &mut ImageSession::default(),
        )
        .unwrap();
        roxmltree::Document::parse(std::str::from_utf8(&raw).unwrap()).unwrap();
        assert_eq!(normalized_svg(raw), c["svg"], "{}", c["name"]);
        let png = output::png_document(
            &doc,
            &mut FlattenBudget::default(),
            &mut ImageSession::default(),
        )
        .unwrap();
        let mut reader = png::Decoder::new(std::io::Cursor::new(png))
            .read_info()
            .unwrap();
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut pixels).unwrap();
        pixels.truncate(info.buffer_size());
        equal(&pixels, &c["pixels"], "final encoded PNG");
        assert_eq!(reader.info().pixel_dims.unwrap().xppu, 3780);
    }
}
#[test]
fn original_final_supersampling_and_sharpen() {
    use rapiddoc_office::metafile::{
        self,
        dib_decode::RgbaImage,
        filters,
        header::{self, Source},
        raster,
    };
    let f = fixture();
    for (i, c) in f["filters"].as_array().unwrap().iter().enumerate() {
        let size: [usize; 2] = serde_json::from_value(c["size"].clone()).unwrap();
        let target = serde_json::from_value(c["target"].clone()).unwrap();
        let image = RgbaImage::new(size[0], size[1], bytes(&c["pixels"])).unwrap();
        let resized = filters::lanczos(&image, target).unwrap();
        equal(resized.pixels(), &c["resize"], &format!("resize {i}"));
        equal(
            filters::unsharp(&resized).pixels(),
            &c["unsharp"],
            &format!("unsharp {i}"),
        );
    }
    for c in f["final"].as_array().unwrap() {
        let bytes: Vec<_> = c["hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
            .collect();
        let h = header::prepare(&bytes, 200, Some([24, 20])).unwrap();
        let doc = match h.source_format {
            Source::Wmf => metafile::wmf::replay(&bytes, h, Some([24, 20])),
            Source::Emf => metafile::emf::replay(&bytes, h, Some([24, 20])),
        }
        .unwrap();
        assert_eq!(
            raster::choose_scale(&doc, 4),
            c["scale"].as_u64().unwrap() as u32
        );
        let mut budget = FlattenBudget::default();
        let result = raster::render(&doc, &mut budget, &mut ImageSession::default()).unwrap();
        equal(result.pixels(), &c["pixels"], c["name"].as_str().unwrap());
        assert_eq!(budget.used, c["used"].as_u64().unwrap() as usize);
    }
}
use std::{io::Read, sync::Arc};
fn fixture() -> Value {
    let mut data = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/metafile_paths.json.gz")
            [..],
    )
    .read_to_end(&mut data)
    .unwrap();
    serde_json::from_slice(&data).unwrap()
}
fn bytes(v: &Value) -> Vec<u8> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect()
}
fn equal(actual: &[u8], expected: &Value, label: &str) {
    let expected = bytes(expected);
    assert_eq!(actual.len(), expected.len(), "{label}");
    if let Some((i, (a, b))) = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        panic!("{label}: byte {i}: {a} != {b}");
    }
}
#[test]
fn original_pillow_polygon_edges() {
    for (i, c) in fixture()["polygons"].as_array().unwrap().iter().enumerate() {
        let points: Vec<Point> = serde_json::from_value(c["points"].clone()).unwrap();
        let mut mask = Mask::new([19, 17], 0).unwrap();
        mask.polygon(&points, 255);
        equal(mask.pixels(), &c["pixels"], &format!("polygon {i}"));
    }
}
#[test]
fn original_fill_stroke_and_hatch_masks() {
    let f = fixture();
    for (i, c) in f["subpaths"].as_array().unwrap().iter().enumerate() {
        let subpaths: paths::Subpaths = serde_json::from_value(c["paths"].clone()).unwrap();
        let mut b = FlattenBudget::default();
        let mask = paths::subpath_mask(&subpaths, [19, 17], c["rule"].as_str().unwrap(), &mut b);
        if let Some(e) = c.get("error") {
            match mask {
                Err(error) => assert_eq!(error.kind.code(), e.as_str().unwrap()),
                Ok(_) => panic!("fill {i}: expected error"),
            }
        } else {
            equal(mask.unwrap().pixels(), &c["pixels"], &format!("fill {i}"));
        }
        assert_eq!(
            b.used,
            c["used"].as_u64().unwrap() as usize,
            "fill {i} budget"
        );
    }
    for (i, c) in f["strokes"].as_array().unwrap().iter().enumerate() {
        let points = serde_json::from_value(c["points"].clone()).unwrap();
        let pen = serde_json::from_value(c["pen"].clone()).unwrap();
        let pattern: Vec<f64> = serde_json::from_value(c["pattern"].clone()).unwrap();
        let mut b = FlattenBudget::default();
        let mask = paths::stroke_mask(
            &[(points, c["closed"].as_bool().unwrap())],
            [19, 17],
            c["width"].as_f64().unwrap(),
            &pen,
            &pattern,
            2.,
            &mut b,
        )
        .unwrap();
        equal(mask.pixels(), &c["pixels"], &format!("stroke {i}"));
        assert_eq!(
            b.used,
            c["used"].as_u64().unwrap() as usize,
            "stroke {i} budget"
        );
    }
    for (i, c) in f["hatches"].as_array().unwrap().iter().enumerate() {
        let brush = serde_json::from_value(c["brush"].clone()).unwrap();
        let size = serde_json::from_value(c["size"].clone()).unwrap();
        equal(
            paths::hatch(size, &brush).unwrap().pixels(),
            &c["pixels"],
            &format!("hatch {i}"),
        );
    }
}
#[test]
fn original_dash_clips_and_path_layers() {
    let f = fixture();
    for (i, c) in f["dashes"].as_array().unwrap().iter().enumerate() {
        let points: Vec<Point> = serde_json::from_value(c["points"].clone()).unwrap();
        let pattern: Vec<f64> = serde_json::from_value(c["pattern"].clone()).unwrap();
        let actual = paths::dash_polyline(&points, &pattern, c["closed"].as_bool().unwrap());
        let expected: Vec<Vec<Point>> = serde_json::from_value(c["visible"].clone()).unwrap();
        assert_eq!(actual.len(), expected.len(), "dash {i}");
        for (a, e) in actual.iter().zip(&expected) {
            assert_eq!(a.len(), e.len());
            for (a, e) in a.iter().zip(e) {
                for k in 0..2 {
                    assert!((a[k] - e[k]).abs() <= 1e-12 * e[k].abs().max(1.));
                }
            }
        }
    }
    for (i, c) in f["clips"].as_array().unwrap().iter().enumerate() {
        let clip: Arc<[Clip]> = serde_json::from_value(c["clip"].clone()).unwrap();
        let mut b = FlattenBudget::default();
        let mut session = ImageSession::default();
        for _ in 0..2 {
            let mask = paths::clip_mask(&clip, Matrix::default(), [19, 17], &mut b, &mut session)
                .unwrap()
                .unwrap();
            equal(mask.pixels(), &c["pixels"], &format!("clip {i}"));
        }
        assert_eq!(b.used, c["used_after_hit"].as_u64().unwrap() as usize);
        assert_eq!(session.hits(), c["hits"].as_u64().unwrap() as usize);
        assert_eq!(session.misses(), c["misses"].as_u64().unwrap() as usize);
        assert_eq!(session.used(), c["cache_used"].as_u64().unwrap() as usize);
    }
    for (i, c) in f["commands"].as_array().unwrap().iter().enumerate() {
        let command = serde_json::from_value(c["command"].clone()).unwrap();
        let matrix = serde_json::from_value(c["matrix"].clone()).unwrap();
        let mut b = FlattenBudget::default();
        let layer = paths::render_path(
            &command,
            matrix,
            [19, 17],
            c["scale"].as_u64().unwrap() as u32,
            &mut b,
            &mut ImageSession::default(),
        )
        .unwrap();
        equal(layer.pixels(), &c["pixels"], &format!("command {i}"));
        assert_eq!(
            b.used,
            c["used"].as_u64().unwrap() as usize,
            "command {i} budget"
        );
    }
}
#[test]
fn original_mixed_documents_and_native_budget_errors() {
    use rapiddoc_office::metafile::{
        self,
        header::{self, Source},
        raster,
    };
    let f = fixture();
    for (index, c) in f["documents"].as_array().unwrap().iter().enumerate() {
        let bytes: Vec<_> = c["hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
            .collect();
        let mut session = ImageSession::default();
        let mut budget = FlattenBudget::default();
        let result = (|| {
            let header = header::prepare(&bytes, 200, Some([24, 20]))?;
            let doc = match header.source_format {
                Source::Wmf => metafile::wmf::replay(&bytes, header, Some([24, 20])),
                Source::Emf => metafile::emf::replay(&bytes, header, Some([24, 20])),
            }?;
            raster::render_once(
                &doc,
                c["scale"].as_u64().unwrap() as u32,
                &mut budget,
                &mut session,
            )
        })();
        let name = format!("document {index} {} scale {}", c["name"], c["scale"]);
        if let Some(error) = c.get("error") {
            match result {
                Err(e) => assert_eq!(e.kind.code(), error.as_str().unwrap(), "{name}"),
                Ok(_) => panic!("{name}: expected {error}"),
            }
        } else {
            equal(
                result.unwrap_or_else(|e| panic!("{name}: {e}")).pixels(),
                &c["pixels"],
                &name,
            );
            assert_eq!(
                budget.used,
                c["used"].as_u64().unwrap() as usize,
                "{name} flatten"
            );
            assert_eq!(
                session.hits(),
                c["hits"].as_u64().unwrap() as usize,
                "{name} cache hits"
            );
            assert_eq!(
                session.misses(),
                c["misses"].as_u64().unwrap() as usize,
                "{name} cache misses"
            );
            assert_eq!(
                session.used(),
                c["cache_used"].as_u64().unwrap() as usize,
                "{name} cache bytes"
            );
        }
    }
    let path = serde_json::from_value(f["budget_path"].clone()).unwrap();
    for c in f["budgets"].as_array().unwrap() {
        let mut budget = FlattenBudget {
            limit: c["limit"].as_u64().unwrap() as usize,
            used: 0,
        };
        let result = paths::path_mask(&path, Matrix::default(), [19, 17], "evenodd", &mut budget);
        if let Some(error) = c.get("error") {
            match result {
                Err(e) => assert_eq!(e.kind.code(), error.as_str().unwrap()),
                Ok(_) => panic!("expected budget error"),
            }
        } else {
            equal(result.unwrap().pixels(), &c["pixels"], "budget");
        }
        assert_eq!(budget.used, c["used"].as_u64().unwrap() as usize);
    }
}
