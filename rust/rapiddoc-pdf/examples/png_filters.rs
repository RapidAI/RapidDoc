//! Development-only lossless PNG filter experiment; no PDF/runtime settings.
use image::{
    codecs::png::{CompressionType, FilterType, PngEncoder},
    ExtendedColorType, ImageEncoder,
};
use std::time::Instant;
fn main() {
    for path in std::env::args().skip(1) {
        let rgb = image::open(&path).unwrap().into_rgb8();
        for (name, filter) in [
            ("adaptive", FilterType::Adaptive),
            ("up", FilterType::Up),
            ("sub", FilterType::Sub),
            ("none", FilterType::NoFilter),
        ] {
            let mut times = vec![];
            let mut sizes = vec![];
            let mut decodes = vec![];
            for _ in 0..5 {
                let start = Instant::now();
                let mut encoded = vec![];
                PngEncoder::new_with_quality(&mut encoded, CompressionType::Fast, filter)
                    .write_image(
                        rgb.as_raw(),
                        rgb.width(),
                        rgb.height(),
                        ExtendedColorType::Rgb8,
                    )
                    .unwrap();
                times.push(start.elapsed().as_secs_f64());
                sizes.push(encoded.len());
                let start = Instant::now();
                let decoded = image::load_from_memory(&encoded).unwrap().into_rgb8();
                decodes.push(start.elapsed().as_secs_f64());
                assert_eq!(decoded, rgb);
            }
            times.sort_by(f64::total_cmp);
            decodes.sort_by(f64::total_cmp);
            println!(
                "{}",
                serde_json::json!({"path":path,"filter":name,"encode_median":times[2],"decode_median":decodes[2],"bytes":sizes[2],"pixels_equal":true})
            );
        }
    }
}
