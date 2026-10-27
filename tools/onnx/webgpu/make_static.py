# SPDX-License-Identifier: Apache-2.0
"""Make a static N=1 SOMA graph that runs entirely on the ONNX Runtime WebGPU EP.

The exported graph accepts any batch size. Its shape arithmetic (Shape, Range,
ConstantOfShape, Mod and friends) then runs on the CPU in onnxruntime-web, and each
CPU node costs a GPU<->host copy. With the batch fixed at one, those subgraphs are
constants. This script:

1. sets the batch dimension of every input and output to 1,
2. folds every node whose inputs are constant, including Shape and Size nodes of
   statically shaped tensors, until nothing changes,
3. replaces OneHot, which the WebGPU EP does not implement, with the exact
   equivalent Cast -> wrap negative indices -> Equal(arange(depth)) -> Where,
4. rewrites the nodes whose data types the WebGPU EP lacks, which onnxruntime-web
   would otherwise run on the CPU with a GPU<->host copy around each: Slice with
   constant parameters on integer data becomes Gather with constant indices, and
   Concat on bool, ScatterElements on int64 and Transpose on int64 run on int32
   between Casts. An int64 Transpose that only feeds gather indices stays int32 up
   to the gather, so ONNX Runtime's transpose optimizer cannot move an int64
   Transpose back onto the CPU. The int64 values involved are token indices and
   counts below 2^31, the range the WebGPU EP's enableInt64 option already assumes,
5. gives every ScatterND and ScatterElements a data input computed on each run.
   The WebGPU kernels copy constant data into their output with a buffer copy that
   graph capture does not record, so replays would scatter into the previous run's
   output. Adding a run-time zero lets ONNX Runtime run the scatter in place,
6. removes unused nodes and initializers, and fails if Range, ConstantOfShape, Mod
   or OneHot remain.

Large weights are set aside before folding and restored afterwards, so they are
never copied. Folding runs each node with the ONNX Runtime CPU kernels, so a folded
value is the value ONNX Runtime would have computed at run time.

    python tools/onnx/webgpu/make_static.py IN.onnx OUT.onnx

OUT.onnx holds the graph, and OUT.onnx.data holds the weights as ONNX external data,
which onnxruntime-web loads through the externalData session option. Writing the
weights separately keeps the peak memory near one copy of the model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

FORBIDDEN_OPS = ("Range", "ConstantOfShape", "Mod", "OneHot")
# Folded results above this many elements stay in the graph as computation.
MAX_FOLDED_ELEMENTS = 1 << 20
LARGE_INITIALIZER_BYTES = 1 << 20


def tensor_bytes(tensor: TensorProto) -> int:
    if tensor.raw_data:
        return len(tensor.raw_data)
    return int(np.prod(tensor.dims, dtype=np.int64)) * 8


def set_batch_dimension(graph: onnx.GraphProto, batch: int) -> None:
    for value in list(graph.input) + list(graph.output):
        dims = value.type.tensor_type.shape.dim
        if dims and not dims[0].HasField("dim_value"):
            dims[0].Clear()
            dims[0].dim_value = batch
    del graph.value_info[:]


def static_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    shapes = {}
    for value in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = tensor_type.shape.dim
        if all(dim.HasField("dim_value") for dim in dims):
            shapes[value.name] = tuple(dim.dim_value for dim in dims)
    return shapes


class Folder:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        self.graph = model.graph
        self.opsets = list(model.opset_import)
        self.large: dict[str, TensorProto] = {}
        self.constants: dict[str, np.ndarray] = {}

    def set_aside_large_initializers(self) -> None:
        kept = []
        for initializer in self.graph.initializer:
            if tensor_bytes(initializer) > LARGE_INITIALIZER_BYTES:
                self.large[initializer.name] = initializer
                self.graph.input.append(
                    helper.make_tensor_value_info(initializer.name, initializer.data_type, list(initializer.dims))
                )
            else:
                kept.append(initializer)
        del self.graph.initializer[:]
        self.graph.initializer.extend(kept)
        for initializer in kept:
            self.constants[initializer.name] = numpy_helper.to_array(initializer)

    def restore_large_initializers(self) -> None:
        used = {name for node in self.graph.node for name in node.input}
        restored = [name for name in self.large if name in used]
        inputs = [value for value in self.graph.input if value.name not in self.large]
        del self.graph.input[:]
        self.graph.input.extend(inputs)
        self.graph.initializer.extend(self.large[name] for name in restored)

    def evaluate(self, node: onnx.NodeProto) -> list[np.ndarray]:
        inputs = [name for name in node.input if name]
        graph = helper.make_graph(
            [node],
            "fold",
            [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(self.constants[name].dtype), None)
             for name in dict.fromkeys(inputs)],
            [onnx.ValueInfoProto(name=name) for name in node.output if name],
        )
        model = helper.make_model(graph, opset_imports=self.opsets, ir_version=self.model.ir_version)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        return session.run(None, {name: self.constants[name] for name in dict.fromkeys(inputs)})

    def fold_pass(self) -> int:
        shapes = static_shapes(self.model)
        kept, folded = [], 0
        for node in self.graph.node:
            outputs: list[np.ndarray] | None = None
            inputs = [name for name in node.input if name]
            if node.op_type == "Constant":
                outputs = self.evaluate(node)
            elif node.op_type in ("Shape", "Size") and inputs[0] in shapes:
                shape = np.asarray(shapes[inputs[0]], dtype=np.int64)
                if node.op_type == "Size":
                    outputs = [np.asarray(np.prod(shape), dtype=np.int64)]
                else:
                    attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
                    outputs = [shape[attrs.get("start", 0): attrs.get("end", len(shape))]]
            elif inputs and all(name in self.constants for name in inputs):
                if all(self.constants[name].size <= 4 * MAX_FOLDED_ELEMENTS for name in inputs):
                    outputs = self.evaluate(node)
                    if any(np.asarray(output).size > MAX_FOLDED_ELEMENTS for output in outputs):
                        outputs = None
            if outputs is None:
                kept.append(node)
                continue
            for name, value in zip(node.output, outputs):
                if name:
                    value = np.asarray(value)
                    self.constants[name] = value
                    self.graph.initializer.append(numpy_helper.from_array(value, name))
            folded += 1
        del self.graph.node[:]
        self.graph.node.extend(kept)
        return folded

    def replace_one_hot(self) -> int:
        replaced = 0
        nodes = []
        for node in self.graph.node:
            if node.op_type != "OneHot":
                nodes.append(node)
                continue
            indices, depth_name, values_name = node.input
            axis = next((helper.get_attribute_value(a) for a in node.attribute if a.name == "axis"), -1)
            if axis != -1 or depth_name not in self.constants or values_name not in self.constants:
                raise RuntimeError(f"OneHot {node.name or node.output[0]} needs axis -1 and constant depth and values.")
            depth = int(np.asarray(self.constants[depth_name]).reshape(-1)[0])
            values = np.asarray(self.constants[values_name]).reshape(-1)
            prefix = f"{node.output[0]}__onehot"
            constants = {
                f"{prefix}_axes": np.asarray([-1], dtype=np.int64),
                f"{prefix}_zero": np.asarray(0, dtype=np.int32),
                f"{prefix}_depth": np.asarray(depth, dtype=np.int32),
                f"{prefix}_classes": np.arange(depth, dtype=np.int32),
                f"{prefix}_off": values[:1].reshape(()),
                f"{prefix}_on": values[1:2].reshape(()),
            }
            for name, value in constants.items():
                self.constants[name] = value
                self.graph.initializer.append(numpy_helper.from_array(value, name))
            nodes += [
                helper.make_node("Cast", [indices], [f"{prefix}_i32"], to=TensorProto.INT32),
                helper.make_node("Unsqueeze", [f"{prefix}_i32", f"{prefix}_axes"], [f"{prefix}_column"]),
                helper.make_node("Less", [f"{prefix}_column", f"{prefix}_zero"], [f"{prefix}_negative"]),
                helper.make_node("Add", [f"{prefix}_column", f"{prefix}_depth"], [f"{prefix}_wrapped"]),
                helper.make_node("Where", [f"{prefix}_negative", f"{prefix}_wrapped", f"{prefix}_column"], [f"{prefix}_index"]),
                helper.make_node("Equal", [f"{prefix}_index", f"{prefix}_classes"], [f"{prefix}_hot"]),
                helper.make_node("Where", [f"{prefix}_hot", f"{prefix}_on", f"{prefix}_off"], [node.output[0]]),
            ]
            replaced += 1
        del self.graph.node[:]
        self.graph.node.extend(nodes)
        return replaced

    def elem_types(self) -> dict[str, int]:
        inferred = onnx.shape_inference.infer_shapes(self.model, strict_mode=True)
        types = {}
        for value in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
            types[value.name] = value.type.tensor_type.elem_type
        for initializer in self.graph.initializer:
            types[initializer.name] = initializer.data_type
        return types

    def add_constant(self, name: str, value: np.ndarray) -> str:
        self.constants[name] = value
        self.graph.initializer.append(numpy_helper.from_array(value, name))
        return name

    def feeds_only_indices(self, name: str, consumers: dict[str, list[onnx.NodeProto]]) -> bool:
        users = consumers.get(name, [])
        if not users or name in {value.name for value in self.graph.output}:
            return False
        for user in users:
            if user.op_type in ("Unsqueeze", "Squeeze", "Expand", "Reshape") and user.input[0] == name:
                if not self.feeds_only_indices(user.output[0], consumers):
                    return False
            elif not (user.op_type in ("Gather", "GatherElements") and list(user.input).index(name) == 1):
                return False
        return True

    def rewrite_webgpu_types(self) -> Counter:
        types = self.elem_types()
        consumers: dict[str, list[onnx.NodeProto]] = {}
        for node in self.graph.node:
            for name in node.input:
                consumers.setdefault(name, []).append(node)
        rewrites: Counter = Counter()
        nodes = []
        for node in self.graph.node:
            data_type = types.get(node.input[0]) if node.input else None
            prefix = f"{node.output[0]}__webgpu"
            if (
                node.op_type == "Slice"
                and data_type not in (TensorProto.FLOAT, TensorProto.FLOAT16)
                and all(name in self.constants for name in node.input[1:] if name)
            ):
                params = [np.asarray(self.constants[name]) if name else None for name in list(node.input[1:]) + [""] * (5 - len(node.input))]
                starts, ends, axes, steps = params[0], params[1], params[2], params[3]
                shape = static_shapes(self.model)[node.input[0]]
                axes = np.arange(len(starts)) if axes is None else axes
                steps = np.ones_like(starts) if steps is None else steps
                current = node.input[0]
                for index, (start, end, axis, step) in enumerate(zip(starts, ends, axes, steps)):
                    positions = np.arange(shape[int(axis)])[slice(int(start), int(end), int(step))]
                    target = node.output[0] if index == len(starts) - 1 else f"{prefix}_{index}"
                    indices = self.add_constant(f"{prefix}_indices_{index}", positions.astype(np.int64))
                    nodes.append(helper.make_node("Gather", [current, indices], [target], axis=int(axis)))
                    current = target
                rewrites["Slice->Gather"] += 1
                continue
            wrap = (
                (node.op_type == "Concat" and data_type == TensorProto.BOOL)
                or (node.op_type == "ScatterElements" and data_type == TensorProto.INT64)
                or (node.op_type == "Transpose" and data_type == TensorProto.INT64)
            )
            if not wrap:
                nodes.append(node)
                continue
            cast_inputs = [0, 1] if node.op_type != "ScatterElements" else [0, 2]
            if node.op_type == "Concat":
                cast_inputs = list(range(len(node.input)))
            inner = onnx.NodeProto()
            inner.CopyFrom(node)
            for position in cast_inputs[: len(node.input)] if node.op_type != "Transpose" else [0]:
                cast = f"{prefix}_in{position}"
                nodes.append(helper.make_node("Cast", [node.input[position]], [cast], to=TensorProto.INT32))
                inner.input[position] = cast
            if node.op_type == "Transpose" and self.feeds_only_indices(node.output[0], consumers):
                nodes.append(inner)
                rewrites["Transpose(INT64)->int32 indices"] += 1
                continue
            inner.output[0] = f"{prefix}_i32"
            nodes.append(inner)
            nodes.append(helper.make_node("Cast", [inner.output[0]], [node.output[0]], to=data_type))
            rewrites[f"{node.op_type}({TensorProto.DataType.Name(data_type)})->int32"] += 1
        del self.graph.node[:]
        self.graph.node.extend(nodes)
        return rewrites

    def make_scatter_data_dynamic(self) -> int:
        types = self.elem_types()
        source = next(value.name for value in self.graph.input if types[value.name] == TensorProto.INT32)
        zero = f"{source}__runtime_zero"
        nodes = [
            helper.make_node("ReduceMin", [source], [f"{zero}_min"], keepdims=0),
            helper.make_node("Mul", [f"{zero}_min", self.add_constant(f"{zero}_factor", np.asarray(0, dtype=helper.tensor_dtype_to_np_dtype(types[source])))], [zero]),
        ]
        zeros: dict[int, str] = {types[source]: zero}
        guarded = 0
        for node in self.graph.node:
            if node.op_type in ("ScatterND", "ScatterElements"):
                data_type = types[node.input[0]]
                if data_type not in zeros:
                    zeros[data_type] = f"{zero}_{TensorProto.DataType.Name(data_type).lower()}"
                    nodes.append(helper.make_node("Cast", [zero], [zeros[data_type]], to=data_type))
                fresh = f"{node.output[0]}__runtime_data"
                nodes.append(helper.make_node("Add", [node.input[0], zeros[data_type]], [fresh]))
                node.input[0] = fresh
                guarded += 1
            nodes.append(node)
        del self.graph.node[:]
        self.graph.node.extend(nodes)
        return guarded

    def remove_unused(self) -> None:
        needed = {value.name for value in self.graph.output}
        kept = []
        for node in reversed(list(self.graph.node)):
            if any(name in needed for name in node.output):
                kept.append(node)
                needed.update(name for name in node.input if name)
        del self.graph.node[:]
        self.graph.node.extend(reversed(kept))
        initializers = [init for init in self.graph.initializer if init.name in needed]
        unique = {init.name: init for init in initializers}
        del self.graph.initializer[:]
        self.graph.initializer.extend(unique.values())
        self.large = {name: init for name, init in self.large.items() if name in needed}


def make_static(source: Path, destination: Path) -> dict[str, object]:
    model = onnx.load(source)
    before = Counter(node.op_type for node in model.graph.node)
    set_batch_dimension(model.graph, 1)
    folder = Folder(model)
    folder.set_aside_large_initializers()
    passes = 0
    while folder.fold_pass():
        folder.remove_unused()
        passes += 1
    one_hots = folder.replace_one_hot()
    folder.remove_unused()
    rewrites = folder.rewrite_webgpu_types()
    folder.remove_unused()
    scatters = folder.make_scatter_data_dynamic()
    onnx.checker.check_model(model)
    folder.restore_large_initializers()
    after = Counter(node.op_type for node in model.graph.node)
    remaining = {op: after[op] for op in FORBIDDEN_OPS if after[op]}
    if remaining:
        raise RuntimeError(f"Ops the WebGPU EP cannot run remain: {remaining}")
    data = destination.with_name(destination.name + ".data")
    data.unlink(missing_ok=True)
    onnx.save(model, destination, save_as_external_data=True, all_tensors_to_one_file=True,
              location=data.name, size_threshold=LARGE_INITIALIZER_BYTES)
    return {
        "nodes_before": sum(before.values()),
        "nodes_after": sum(after.values()),
        "fold_passes": passes,
        "one_hot_replaced": one_hots,
        "webgpu_type_rewrites": dict(rewrites),
        "scatter_inputs_made_dynamic": scatters,
        "ops_after": dict(sorted(after.items(), key=lambda item: -item[1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    summary = make_static(args.source, args.destination)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

