"""Partition a standard SLANet graph in memory; keep model files portable."""
def classifier_output_type(path):
    """Only recognize a final class Softmax, not unrelated internal Softmax nodes."""
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    producers = {o: n for n in model.graph.node for o in n.output}
    name = model.graph.output[0].name
    seen = set()
    while name not in seen:
        seen.add(name)
        node = producers.get(name)
        if node is None:
            break
        if node.op_type == "Softmax":
            axis = next((a.i for a in node.attribute if a.name == "axis"), -1)
            return "probabilities" if axis in (1, -1) else "logits"
        if node.op_type not in ("Identity", "Cast"):
            break
        name = node.input[0]
    return "logits"


def partition_slanet(path):
    import onnx
    from onnx.utils import Extractor

    model = onnx.load(str(path))
    onnx.external_data_helper.convert_model_from_external_data(model)
    model = onnx.shape_inference.infer_shapes(model)
    loops = [n for n in model.graph.node if n.op_type == "Loop"]
    if len(loops) != 1:
        raise ValueError("SLANet partition requires one top-level Loop")
    body = next(a.g for a in loops[0].attribute if a.type == onnx.AttributeProto.GRAPH)
    local = {v.name for v in body.input} | {v.name for v in body.initializer}
    local.update(o for n in body.node for o in n.output)
    if any(i and i not in local for n in body.node for i in n.input):
        raise ValueError("Unsupported implicit Loop inputs")
    producers = {o: n for n in model.graph.node for o in n.output}
    cuts = [i for i in loops[0].input if i in producers and producers[i].op_type != "Constant"]
    if not cuts:
        raise ValueError("No encoder boundary found")
    inputs = [v.name for v in model.graph.input]
    outputs = [v.name for v in model.graph.output]
    extractor = Extractor(model)
    encoder = extractor.extract_model(inputs, cuts)
    decoder = extractor.extract_model(cuts, outputs)
    if any(n.op_type in ("Conv", "ConvTranspose") for n in decoder.graph.node):
        raise ValueError("Decoder still depends on encoder convolutions")
    for part in (encoder, decoder):
        del part.metadata_props[:]
        part.metadata_props.extend(model.metadata_props)
        onnx.checker.check_model(part)
    return encoder.SerializeToString(), decoder.SerializeToString()
