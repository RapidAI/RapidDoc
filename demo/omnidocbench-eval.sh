docker run --rm `
  --entrypoint bash `
  -v "D:\Download\OmniDocBench\OmniDocBench.json:/workspace/gt/OmniDocBench.json:ro" `
  -v "D:\Download\OmniDocBench\layout_v3-ocrv6_smail_auto_tablev3:/workspace/data_md/predictions:ro" `
  -v "D:\Download\OmniDocBench\result_layout_v3-ocrv6_smail_auto_tablev3:/workspace/result" `
  ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204 `
  -c "cat > configs/custom.yaml << 'EOF'
end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist, CDM]
    table:
      metric: [TEDS, Edit_dist]
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: ./gt/OmniDocBench.json
    prediction:
      data_path: ./data_md/predictions
    match_method: quick_match
    match_workers: 4
    quick_match_truncated_timeout_sec: 300
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
EOF
python pdf_validation.py --config configs/custom.yaml"