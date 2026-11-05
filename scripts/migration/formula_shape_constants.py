"""Conservatively fold small integer shape expressions, including partially known shapes.

No image samples, floating point arithmetic, batch specialization or sequence limits.
"""
import numpy as np
from onnx import AttributeProto, TensorProto, helper, numpy_helper

INTEGER_TYPES = {TensorProto.INT64, TensorProto.INT32, TensorProto.BOOL}


def fold(graph, inherited_values=None, inherited_shapes=None):
    local = {value.name for value in graph.input} | {value.name for value in graph.initializer}
    local.update(output for node in graph.node for output in node.output)
    values = {key: value for key, value in (inherited_values or {}).items() if key not in local}
    shapes = {key: value for key, value in (inherited_shapes or {}).items() if key not in local}
    types = {}
    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        tensor = value.type.tensor_type
        types[value.name] = tensor.elem_type
        if tensor.HasField('shape'):
            shapes[value.name] = [dim.dim_value if dim.HasField('dim_value') else None
                                  for dim in tensor.shape.dim]
    for value in graph.initializer:
        shapes[value.name] = list(value.dims)
        types[value.name] = value.data_type
        if value.data_type in INTEGER_TYPES and np.prod(value.dims) <= 64:
            values[value.name] = numpy_helper.to_array(value).astype(object)
    count = 0
    rewritten = []
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                count += fold(attribute.g, values, shapes)
        attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
        args = [values.get(value) for value in node.input]
        result = None
        known = lambda value: value is not None and all(item is not None for item in value.flat)
        try:
            if node.op_type == 'Constant' and 'value' in attrs:
                tensor = attrs['value']
                if tensor.data_type in INTEGER_TYPES and np.prod(tensor.dims) <= 64:
                    result = numpy_helper.to_array(tensor).astype(object)
                    types[node.output[0]] = tensor.data_type
            elif node.op_type == 'Shape' and node.input[0] in shapes:
                dims = shapes[node.input[0]]
                result = np.asarray(dims[attrs.get('start', 0):attrs.get('end', len(dims))], dtype=object)
                types[node.output[0]] = TensorProto.INT64
            elif node.op_type == 'Identity' and args[0] is not None:
                result = args[0]
            elif node.op_type == 'Slice' and args[0] is not None and all(known(a) for a in args[1:]):
                starts, ends = args[1].reshape(-1), args[2].reshape(-1)
                axes = args[3].reshape(-1) if len(args) > 3 else range(len(starts))
                steps = args[4].reshape(-1) if len(args) > 4 else [1] * len(starts)
                if all(int(step) == 1 for step in steps):
                    slices = [slice(None)] * args[0].ndim
                    for start, end, axis in zip(starts, ends, axes):
                        slices[int(axis)] = slice(int(start), int(end))
                    result = args[0][tuple(slices)]
            elif node.op_type in ('Squeeze', 'Unsqueeze') and args[0] is not None:
                axes = args[1] if len(args) > 1 else attrs.get('axes')
                if axes is not None and known(np.asarray(axes, dtype=object)):
                    axes = tuple(int(axis) for axis in np.asarray(axes).reshape(-1))
                    function = np.squeeze if node.op_type == 'Squeeze' else np.expand_dims
                    result = function(args[0], axis=axes)
            elif node.op_type == 'Concat' and all(a is not None for a in args):
                result = np.concatenate(args, axis=attrs['axis'])
            elif node.op_type == 'Gather' and args[0] is not None and known(args[1]):
                result = np.take(args[0], args[1].astype(np.int64), axis=attrs.get('axis', 0))
            elif node.op_type == 'Cast' and args[0] is not None and attrs['to'] in INTEGER_TYPES:
                result = args[0]
                if known(result):
                    result = result.astype(helper.tensor_dtype_to_np_dtype(attrs['to'])).astype(object)
                types[node.output[0]] = attrs['to']
            elif node.op_type in ('Add', 'Mul', 'Sub') and all(known(a) for a in args):
                function = {'Add': np.add, 'Mul': np.multiply, 'Sub': np.subtract}[node.op_type]
                result = function(args[0], args[1])
        except (ValueError, TypeError, IndexError, OverflowError):
            result = None
        if result is not None:
            result = np.asarray(result, dtype=object)
            if result.size <= 64 and len(node.output) == 1:
                values[node.output[0]] = result
                dtype = types.get(node.output[0])
                if node.op_type != 'Constant' and known(result) and dtype in INTEGER_TYPES:
                    tensor = numpy_helper.from_array(result.astype(helper.tensor_dtype_to_np_dtype(dtype)))
                    rewritten.append(helper.make_node('Constant', [], list(node.output), value=tensor,
                                                      name=node.name + '__integer_constant'))
                    count += 1
                    continue
        rewritten.append(node)
    del graph.node[:]
    graph.node.extend(rewritten)
    return count
