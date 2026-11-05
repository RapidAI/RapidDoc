use rapiddoc_core::ErrorCode;
use rapiddoc_inference::native_text::{Line, NativePage};
use serde_json::{json, Value};
fn facts() -> Value {
    let glyph = |index, text, bounds| json!({"index":index,"text":text,"is_generated":false,"angle_degrees":0,"visual_tight_bounds":bounds});
    json!({"visual_width_pt":100,"visual_height_pt":20,"rotation":0,"images":[],"characters":[
        glyph(0,"A",[5,4,45,16]),glyph(1," ",[45,4,55,16]),glyph(2,"B",[55,4,95,16])]})
}
fn line() -> Line {
    Line {
        points: [[0., 0.], [100., 0.], [100., 20.], [0., 20.]],
        score: 0.9,
    }
}
fn page(value: &Value) -> NativePage {
    NativePage::from_json(&value.to_string(), 100, 20).unwrap()
}
#[test]
fn native_spaces_source_indices_unique_assignment_and_fractional_scale() {
    let data = facts();
    let actual = page(&data).match_lines(&[line(), line()], 100).unwrap();
    assert_eq!(actual.lines[0].text.as_deref(), Some("A B"));
    assert_eq!(actual.lines[0].character_indices, [0, 1, 2]);
    assert_eq!(actual.lines[1].reason, "no_native_characters");
    assert_eq!(actual.work, 6);
    let scaled = NativePage::from_json(&data.to_string(), 137, 43).unwrap();
    let l = Line {
        points: [[0., 0.], [137., 0.], [137., 43.], [0., 43.]],
        score: 1.,
    };
    assert_eq!(
        scaled.match_lines(&[l], 3).unwrap().lines[0].method,
        "native"
    );
}
#[test]
fn mixed_raster_rotation_and_unreliable_native_fall_back() {
    let mut data = facts();
    data["images"] = json!([{"object_path":[1],"bounds":[0,0,1,1],"visual_bounds":[50,5,90,15],"clip_bounds":null}]);
    assert_eq!(
        page(&data).match_lines(&[line()], 100).unwrap().lines[0].reason,
        "raster_overlap"
    );
    data = facts();
    data["rotation"] = json!(90);
    assert_eq!(
        page(&data).match_lines(&[line()], 100).unwrap().lines[0].reason,
        "non_horizontal"
    );
    data = facts();
    data["characters"][0]["text"] = Value::Null;
    assert_eq!(
        page(&data).match_lines(&[line()], 100).unwrap().lines[0].reason,
        "invalid_unicode"
    );
    data = facts();
    data["characters"][0]["is_generated"] = json!(true);
    assert_eq!(
        page(&data).match_lines(&[line()], 100).unwrap().lines[0].reason,
        "unreliable_native_character"
    );
    data = facts();
    data["characters"][2]["visual_tight_bounds"] = json!([15, 4, 20, 16]);
    data["characters"][0]["visual_tight_bounds"] = json!([5, 4, 10, 16]);
    assert_eq!(
        page(&data).match_lines(&[line()], 100).unwrap().lines[0].reason,
        "insufficient_native_coverage"
    );
    data = facts();
    data["characters"][1]["text"] = json!("X");
    data["characters"][1]["visual_tight_bounds"] = json!([150, 4, 160, 16]);
    assert_eq!(
        page(&data).match_lines(&[line()], 100).unwrap().lines[0].reason,
        "noncontiguous_characters"
    );
}
#[test]
fn malformed_geometry_indices_and_work_budget_fail_explicitly() {
    assert_eq!(
        page(&facts()).match_lines(&[line()], 2).err().unwrap().code,
        ErrorCode::ResourceLimit
    );
    let mut bad = line();
    bad.points[0][0] = f64::NAN;
    assert_eq!(
        page(&facts()).match_lines(&[bad], 100).err().unwrap().code,
        ErrorCode::InvalidInput
    );
    let mut data = facts();
    data["characters"][1]["index"] = json!(3);
    assert!(NativePage::from_json(&data.to_string(), 100, 20).is_err());
    assert!(NativePage::from_json(&facts().to_string(), 0, 20).is_err());
}
