use rapiddoc_core::ErrorCode;
use rapiddoc_inference::{
    table_structure::{alphabet, decode, prepare, Variant},
    TensorOutput,
};

#[test]
fn repeated_cells_are_not_ctc_collapsed_and_eos_terminates() {
    let labels = alphabet("<tbody>\n</tbody>\n<tr>\n</tr>\n<td>").unwrap();
    let td = labels.iter().position(|s| s == "<td></td>").unwrap();
    let mut values = vec![0.; 4 * labels.len()];
    values[td] = 0.9;
    values[labels.len() + td] = 0.8;
    values[2 * labels.len() + labels.len() - 1] = 0.99;
    values[3 * labels.len() + td] = 0.9;
    let bbox = TensorOutput {
        shape: vec![1, 4, 4],
        values: vec![0.1; 16],
    };
    let prob = TensorOutput {
        shape: vec![1, 4, labels.len() as i64],
        values,
    };
    let output = decode(&bbox, &prob, &labels, 200, 100, Variant::Slanet1m).unwrap();
    assert_eq!(output.cell_bboxes.len(), 2);
    assert_eq!(
        output
            .tokens
            .iter()
            .filter(|s| s.as_str() == "<td></td>")
            .count(),
        2
    );
    assert!((output.confidence - 0.85).abs() < 1e-5);
}

#[test]
fn empty_structure_and_wrong_variant_fail_instead_of_nan_or_wrong_coordinates() {
    let labels = alphabet("<td></td>").unwrap();
    let bbox = TensorOutput {
        shape: vec![1, 2, 4],
        values: vec![0.; 8],
    };
    let prob = TensorOutput {
        shape: vec![1, 2, labels.len() as i64],
        values: vec![0.; 2 * labels.len()],
    };
    assert_eq!(
        decode(&bbox, &prob, &labels, 200, 100, Variant::Slanet1m)
            .err()
            .unwrap()
            .code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        decode(&bbox, &prob, &labels, 200, 100, Variant::SlanetPlus)
            .err()
            .unwrap()
            .code,
        ErrorCode::Unsupported
    );
    assert_eq!(
        prepare(&[0, 0, 0], 1, 1, 100).err().unwrap().code,
        ErrorCode::ResourceLimit
    );
    assert!(alphabet("sos\n<td>").is_err());
}
