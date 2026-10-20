#!/usr/bin/env python3
"""Generate the test models in tests/data, their expected outputs, and the example model.

Usage: python3 -I tools/gen_models.py
Needs the tensorflow (2.21), ai-edge-litert and numpy Python packages. Each model is
small and built from fixed seeds. The expected outputs come from Python's TensorFlow
Lite interpreter (ai_edge_litert). Inputs and outputs are raw little-endian arrays
(.bin) that the tests read with fs_read_all.
"""
import os
import struct
import zlib

import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")
EXAMPLES = os.path.join(ROOT, "examples", "data")


def convert(fn, path, quantize=None):
    """Converts a tf.function with an input signature to a .tflite file."""
    conv = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()])
    if quantize is not None:
        rep, io_type = quantize
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = io_type
        conv.inference_output_type = io_type
    data = conv.convert()
    with open(path, "wb") as f:
        f.write(data)
    return path


def interp(path, shapes=None):
    it = Interpreter(model_path=path, num_threads=1)
    if shapes:
        for i, s in shapes.items():
            it.resize_tensor_input(i, s)
    it.allocate_tensors()
    return it


def run(path, feeds, shapes=None):
    """Runs a model with inputs in model order; returns its outputs in model order."""
    it = interp(path, shapes)
    for d, x in zip(it.get_input_details(), feeds):
        it.set_tensor(d["index"], x)
    it.invoke()
    return [it.get_tensor(d["index"]) for d in it.get_output_details()]


def raw(name, a):
    np.ascontiguousarray(a).tofile(os.path.join(DATA, name))


def describe(path):
    it = interp(path)
    for kind, ds in (("in", it.get_input_details()), ("out", it.get_output_details())):
        for d in ds:
            print("  %s %s %s %s %s %s" % (kind, d["name"], d["dtype"].__name__, list(d["shape"]),
                                           list(d["shape_signature"]), d["quantization"]))


def mlp():
    rng = np.random.default_rng(1)
    w = tf.constant(rng.standard_normal((4, 3)).astype(np.float32))
    b = tf.constant(rng.standard_normal(3).astype(np.float32))

    @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32, name="x")])
    def f(x):
        return tf.nn.relu(tf.matmul(x, w) + b)
    path = convert(f, os.path.join(DATA, "mlp.tflite"))
    x = np.array([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32)
    raw("mlp_x.bin", x)
    raw("mlp_y.bin", run(path, [x])[0])
    return path


def conv():
    rng = np.random.default_rng(2)
    w = tf.constant(rng.standard_normal((3, 3, 1, 2)).astype(np.float32))
    b = tf.constant(np.array([0.1, -0.2], dtype=np.float32))

    @tf.function(input_signature=[tf.TensorSpec([1, 6, 6, 1], tf.float32, name="x")])
    def f(x):
        c = tf.nn.conv2d(x, w, strides=1, padding="SAME") + b
        return tf.nn.max_pool2d(c, 2, 2, "VALID")
    path = convert(f, os.path.join(DATA, "conv.tflite"))
    x = rng.standard_normal((1, 6, 6, 1)).astype(np.float32)
    raw("conv_x.bin", x)
    raw("conv_y.bin", run(path, [x])[0])
    return path


def quant(io_type, name):
    # x float[1, 8] -> Dense(6) + ReLU -> Dense(3), fully integer-quantized with
    # int8 (or uint8) inputs and outputs.
    rng = np.random.default_rng(3)
    w1 = tf.constant(rng.standard_normal((8, 6)).astype(np.float32) * 0.5)
    b1 = tf.constant(rng.standard_normal(6).astype(np.float32) * 0.1)
    w2 = tf.constant(rng.standard_normal((6, 3)).astype(np.float32) * 0.5)
    b2 = tf.constant(rng.standard_normal(3).astype(np.float32) * 0.1)

    @tf.function(input_signature=[tf.TensorSpec([1, 8], tf.float32, name="x")])
    def f(x):
        return tf.matmul(tf.nn.relu(tf.matmul(x, w1) + b1), w2) + b2

    calib = np.random.default_rng(4).uniform(-2, 2, (64, 1, 8)).astype(np.float32)

    def rep():
        for c in calib:
            yield [c]
    path = convert(f, os.path.join(DATA, name + ".tflite"), (rep, io_type))
    it = interp(path)
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    xf = np.array([[0.5, -1.0, 1.5, 0.25, -0.75, 2.0, -2.0, 0.0]], dtype=np.float32)
    s, z = di["quantization"]
    info = np.iinfo(di["dtype"])
    xq = np.clip(np.round(xf / s) + z, info.min, info.max).astype(di["dtype"])
    raw(name + "_xf.bin", xf)
    raw(name + "_x.bin", xq)
    raw(name + "_y.bin", run(path, [xq])[0])
    with open(os.path.join(DATA, name + "_q.bin"), "wb") as fq:
        fq.write(struct.pack("<fifi", s, z, do["quantization"][0], do["quantization"][1]))
    return path


def ints():
    # ids int64[3] -> y = ids * 3 + 1 (int64) and n = ids as int32;
    # pix uint8[4] -> f = pix / 255 (float).
    @tf.function(input_signature=[tf.TensorSpec([3], tf.int64, name="ids"),
                                  tf.TensorSpec([4], tf.uint8, name="pix")])
    def f(ids, pix):
        return {"y": ids * 3 + 1, "n": tf.cast(ids, tf.int32), "f": tf.cast(pix, tf.float32) / 255.0}
    return convert(f, os.path.join(DATA, "ints.tflite"))


def dynamic():
    # x float[batch, 3] -> y = softmax(x), z = sum(x, axis 1) [batch]
    @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32, name="x")])
    def f(x):
        return {"y": tf.nn.softmax(x, axis=1), "z": tf.reduce_sum(x, axis=1)}
    path = convert(f, os.path.join(DATA, "dynamic.tflite"))
    x = np.arange(15, dtype=np.float32).reshape(5, 3) / 4.0 - 1.0
    raw("dynamic_x.bin", x)
    outs = run(path, [x], {0: [5, 3]})
    it = interp(path)
    names = [d["name"] for d in it.get_output_details()]
    for n, o in zip(names, outs):
        raw("dynamic_%s.bin" % ("y" if o.ndim == 2 else "z"), o)
    return path


def multi():
    # a, b float[2, 2] -> sum = a + b, prod = a @ b, neg = -a
    @tf.function(input_signature=[tf.TensorSpec([2, 2], tf.float32, name="a"),
                                  tf.TensorSpec([2, 2], tf.float32, name="b")])
    def f(a, b):
        return {"sum": a + b, "prod": tf.matmul(a, b), "neg": -a}
    return convert(f, os.path.join(DATA, "multi.tflite"))


def half():
    # x float16[2, 3] -> s = x as float, y = (s * 1.5 + 0.25) as float16
    @tf.function(input_signature=[tf.TensorSpec([2, 3], tf.float16, name="x")])
    def f(x):
        s = tf.cast(x, tf.float32)
        return {"s": s, "y": tf.cast(s * 1.5 + 0.25, tf.float16)}
    path = convert(f, os.path.join(DATA, "half.tflite"))
    x = np.array([[0.5, -1.25, 3.0], [65504.0 / 4, 1e-4, -0.0]], dtype=np.float16)
    raw("half_x.bin", x)
    for o in run(path, [x]):
        if o.dtype == np.float16:
            raw("half_y.bin", o)
    return path


def resize():
    # x float[1, h, w, c], size int32[2] -> bilinear resize with half-pixel centers,
    # the reference for <tflite/image>'s tflite_image_resize.
    @tf.function(input_signature=[tf.TensorSpec([1, None, None, 4], tf.float32, name="x"),
                                  tf.TensorSpec([2], tf.int32, name="size")])
    def f(x, size):
        return tf.image.resize(x, size, method="bilinear")
    return convert(f, os.path.join(DATA, "resize.tflite"))


def pixels():
    # image float[1, h, w, 3] (dynamic h and w) -> the mean of each channel [1, 3];
    # and a fully quantized uint8 version on [1, 4, 4, 3] images in -1..1.
    @tf.function(input_signature=[tf.TensorSpec([1, None, None, 3], tf.float32, name="image")])
    def f(image):
        return tf.reduce_mean(image, axis=[1, 2])
    path = convert(f, os.path.join(DATA, "pixels.tflite"))

    @tf.function(input_signature=[tf.TensorSpec([1, 4, 4, 3], tf.float32, name="image")])
    def g(image):
        return tf.reduce_mean(image, axis=[1, 2])
    calib = np.random.default_rng(5).uniform(-1, 1, (32, 1, 4, 4, 3)).astype(np.float32)
    calib[0, 0, 0, 0] = [-1, -1, -1]
    calib[0, 0, 0, 1] = [1, 1, 1]

    def rep():
        for c in calib:
            yield [c]
    convert(g, os.path.join(DATA, "pixels_u8.tflite"), (rep, tf.uint8))
    return path


def colors():
    # A tiny image classifier for examples/classify.esk: x float[1, 32, 32, 3] normalized
    # with the ImageNet mean and std -> probabilities of red, green, blue and gray from
    # the mean color of the image.
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    w = tf.constant(np.array([[2, -1, -1, 0], [-1, 2, -1, 0], [-1, -1, 2, 0]], dtype=np.float32) * 8.0)
    b = tf.constant(np.array([0, 0, 0, 1.0], dtype=np.float32))

    @tf.function(input_signature=[tf.TensorSpec([1, 32, 32, 3], tf.float32, name="image")])
    def f(image):
        c = image * std + mean
        p = tf.reduce_mean(c, axis=[1, 2])
        return tf.nn.softmax(tf.matmul(p, w) + b)
    path = convert(f, os.path.join(EXAMPLES, "colors.tflite"))
    with open(os.path.join(EXAMPLES, "colors.txt"), "w") as fl:
        fl.write("red\ngreen\nblue\ngray\n")
    return path


def png(path, rgb):
    h, w, _ = rgb.shape
    rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def sample():
    # A 64x48 sky-blue gradient with a small orange square, for examples/classify.esk.
    y, x = np.mgrid[0:48, 0:64]
    rgb = np.stack([40 + x // 2, 90 + y, 200 + x // 3], axis=-1)
    rgb[10:20, 40:52] = [240, 140, 30]
    png(os.path.join(EXAMPLES, "sample.png"), np.clip(rgb, 0, 255).astype(np.uint8))


def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(EXAMPLES, exist_ok=True)
    paths = [mlp(), conv(), quant(tf.int8, "quant"), quant(tf.uint8, "quant_u8"), ints(), dynamic(),
             multi(), half(), resize(), pixels(), os.path.join(DATA, "pixels_u8.tflite"), colors()]
    sample()
    for p in paths:
        print(os.path.relpath(p, ROOT))
        describe(p)


main()
