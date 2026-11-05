//! Original SLANet loop boundary and constant transfer, all in memory.
use super::*;
fn extract(model: &Msg, graph: &Msg, inputs: &[Msg], outputs: &[Msg]) -> Result<Msg, Error> {
    let nodes = messages(graph, 1)?;
    let mut producers = BTreeMap::new();
    for (i, n) in nodes.iter().enumerate() {
        for name in n.texts(2)? {
            producers.insert(name, i);
        }
    }
    let boundary: BTreeSet<_> = inputs.iter().map(|v| v.text(1)).collect::<Result<_, _>>()?;
    let initializers: BTreeMap<_, _> = messages(graph, 5)?
        .into_iter()
        .map(|t| Ok((t.text(8)?, t)))
        .collect::<Result<_, Error>>()?;
    let mut need: BTreeSet<String> = outputs
        .iter()
        .map(|v| v.text(1))
        .collect::<Result<_, _>>()?;
    let mut pending: Vec<_> = need.iter().cloned().collect();
    let mut indices = BTreeSet::new();
    let mut weights = BTreeSet::new();
    while let Some(name) = pending.pop() {
        if name.is_empty() || boundary.contains(&name) {
            continue;
        }
        if initializers.contains_key(&name) {
            weights.insert(name);
            continue;
        }
        let index = *producers
            .get(&name)
            .ok_or_else(|| invalid(format!("Unbound SLANet tensor {name}")))?;
        if indices.insert(index) {
            for input in nodes[index].texts(1)? {
                if need.insert(input.clone()) {
                    pending.push(input);
                }
            }
        }
    }
    let mut g = graph.clone();
    g.replace(
        1,
        indices
            .into_iter()
            .map(|i| Field::nested(1, &nodes[i]))
            .collect(),
    );
    g.replace(
        5,
        weights
            .iter()
            .map(|n| Field::nested(5, &initializers[n]))
            .collect(),
    );
    g.replace(11, inputs.iter().map(|v| Field::nested(11, v)).collect());
    g.replace(12, outputs.iter().map(|v| Field::nested(12, v)).collect());
    let infos = messages(graph, 13)?
        .into_iter()
        .filter(|v| need.contains(&v.text(1).unwrap_or_default()))
        .map(|v| Field::nested(13, &v))
        .collect();
    g.replace(13, infos);
    let mut m = model.clone();
    m.set_sub(7, &g);
    Ok(m)
}
pub(super) fn split(source: Vec<u8>) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let mut model = Msg::decode(source)?;
    normalize_metadata(&mut model)?;
    let graph = model.sub(7)?;
    external(&graph)?;
    let nodes = messages(&graph, 1)?;
    let loops: Vec<_> = nodes
        .iter()
        .filter(|n| n.text(4).ok().as_deref() == Some("Loop"))
        .collect();
    if loops.len() != 1 {
        return Err(invalid("SLANet requires one top-level Loop"));
    }
    let loop_node = loops[0];
    let body = attrs(loop_node)?
        .get("body")
        .ok_or_else(|| invalid("Missing SLANet body"))?
        .sub(6)?;
    let mut local = BTreeSet::new();
    for v in messages(&body, 11)? {
        local.insert(v.text(1)?);
    }
    for t in messages(&body, 5)? {
        local.insert(t.text(8)?);
    }
    for n in messages(&body, 1)? {
        local.extend(n.texts(2)?);
    }
    if messages(&body, 1)?.iter().any(|n| {
        n.texts(1)
            .unwrap_or_default()
            .iter()
            .any(|v| !v.is_empty() && !local.contains(v))
    }) {
        return Err(invalid("SLANet implicit Loop inputs unsupported"));
    }
    let producers: BTreeMap<String, &Msg> = nodes
        .iter()
        .flat_map(|n| {
            n.texts(2)
                .unwrap_or_default()
                .into_iter()
                .map(move |o| (o, n))
        })
        .collect();
    let body_inputs = messages(&body, 11)?;
    let loop_inputs = loop_node.texts(1)?;
    if body_inputs.len() != loop_inputs.len() {
        return Err(invalid("SLANet loop input arity"));
    }
    let mut cuts = vec![];
    for (i, name) in loop_inputs.iter().enumerate() {
        if let Some(n) = producers.get(name) {
            if n.text(4)? != "Constant" {
                let mut v = body_inputs[i].clone();
                v.set_text(1, name);
                cuts.push(v);
            }
        }
    }
    if cuts.is_empty() {
        return Err(invalid("SLANet encoder boundary missing"));
    }
    let mut input = messages(&graph, 11)?;
    if input.len() != 1 {
        return Err(invalid("SLANet single-image input required"));
    }
    input[0] = value_info(&input[0].text(1)?, 1, &[1, 3, 488, 488]);
    let encoder = extract(&model, &graph, &input, &cuts)?;
    let decoder = extract(&model, &graph, &cuts, &messages(&graph, 12)?)?;
    if messages(&decoder.sub(7)?, 1)?
        .iter()
        .any(|n| matches!(n.text(4).ok().as_deref(), Some("Conv" | "ConvTranspose")))
    {
        return Err(invalid("SLANet decoder still uses encoder convolution"));
    }
    Ok((encoder.encode(), decoder.encode()))
}
pub(super) fn finish(
    encoder: Vec<u8>,
    decoder: Vec<u8>,
    folded: Vec<u8>,
) -> Result<Prepared, Error> {
    let mut encoder = Msg::decode(encoder)?;
    let mut decoder = Msg::decode(decoder)?;
    let folded = Msg::decode(folded)?.sub(7)?;
    let mut constants: BTreeMap<_, _> = messages(&folded, 5)?
        .into_iter()
        .map(|t| Ok((t.text(8)?, t)))
        .collect::<Result<_, Error>>()?;
    for n in messages(&folded, 1)? {
        let op = n.text(4)?;
        let output = n.texts(2)?;
        if output.is_empty() {
            continue;
        }
        let value = if op == "Constant" {
            attrs(&n)?.get("value").map(|a| a.sub(5)).transpose()?
        } else if op == "Identity" {
            n.texts(1)?.first().and_then(|s| constants.get(s)).cloned()
        } else {
            None
        };
        if let Some(mut t) = value {
            t.set_text(8, &output[0]);
            constants.insert(output[0].clone(), t);
        }
    }
    let mut eg = encoder.sub(7)?;
    let mut dg = decoder.sub(7)?;
    let outputs = messages(&eg, 12)?;
    let fixed: BTreeSet<_> = outputs
        .iter()
        .filter_map(|v| v.text(1).ok())
        .filter(|n| constants.contains_key(n))
        .collect();
    let dynamic: Vec<_> = outputs
        .into_iter()
        .filter(|v| !fixed.contains(&v.text(1).unwrap_or_default()))
        .collect();
    if dynamic.is_empty()
        || dynamic
            .iter()
            .any(|v| tensor_type(v).map(|v| v.0).unwrap_or(0) != 1)
    {
        return Err(invalid("SLANet partition requires dynamic FP32 boundary"));
    }
    eg.replace(12, dynamic.iter().map(|v| Field::nested(12, v)).collect());
    let inputs = messages(&dg, 11)?;
    dg.replace(
        11,
        inputs
            .iter()
            .filter(|v| !fixed.contains(&v.text(1).unwrap_or_default()))
            .map(|v| Field::nested(11, v))
            .collect(),
    );
    let existing: BTreeSet<_> = messages(&dg, 5)?
        .iter()
        .map(|t| t.text(8))
        .collect::<Result<_, _>>()?;
    for name in &fixed {
        if existing.contains(name) {
            return Err(invalid("SLANet duplicate initializer"));
        }
        dg.fields.push(Field::nested(5, &constants[name]));
    }
    let incoming: BTreeSet<_> = messages(&dg, 11)?
        .iter()
        .map(|v| v.text(1))
        .collect::<Result<_, _>>()?;
    let outgoing: BTreeSet<_> = dynamic
        .iter()
        .map(|v| v.text(1))
        .collect::<Result<_, _>>()?;
    if incoming != outgoing {
        return Err(invalid("SLANet boundary mismatch"));
    }
    encoder.set_sub(7, &eg);
    decoder.set_sub(7, &dg);
    Ok(Prepared {
        encoder: encoder.encode(),
        decoder: Some(decoder.encode()),
        report: Report {
            partitioned: true,
            ..Default::default()
        },
    })
}
