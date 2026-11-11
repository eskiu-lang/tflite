<p align="center">
  <img src="https://raw.githubusercontent.com/eskiu-lang/eskiu/main/assets/logo.png" alt="Eskiu, a kitten with a ball of yarn" width="140">
</p>

<h2 align="center">tflite</h2>
<p align="center">TensorFlow Lite model inference for Eskiu.</p>
<p align="center"><sub>Part of the <a href="https://github.com/eskiu-lang">Eskiu</a> project</sub></p>
<p align="center">
  <a href="https://eskiu-lang.org">eskiu-lang.org</a> &nbsp;&middot;&nbsp;
  <a href="#modules">Modules</a> &nbsp;&middot;&nbsp;
  <a href="examples/">Examples</a> &nbsp;&middot;&nbsp;
  <a href="https://github.com/eskiu-lang/eskiu">The language</a>
</p>

<p align="center">
  <a href="https://github.com/eskiu-lang/tflite/actions/workflows/ci.yml"><img src="https://github.com/eskiu-lang/tflite/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/version-0.1.0-6448d4" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/TensorFlow%20Lite-2.21-orange" alt="TensorFlow Lite 2.21">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
</p>

---

`tflite` is an Eskiu package for running TensorFlow Lite (LiteRT) models. It calls
`libtensorflowlite_c` through its C API, so a `.tflite` model converted from TensorFlow,
Keras, JAX or PyTorch runs on the CPU everywhere, with XNNPACK, and on the GPU through the
Metal delegate (macOS) or the OpenCL/OpenGL delegate (Linux).

- **Models and interpreters.** Load a model from a file or from bytes, with a thread count
  and a delegate.
- **Introspection.** The name, element type, shape, dynamic dimensions (-1) and
  quantization (scale and zero point) of every input and output.
- **Typed inputs.** `float32`, `float16`, `int8`, `uint8`, `int32` and `int64` data, checked
  against the input's type and shape; a dynamic input is resized to the data's shape.
- **Quantization.** Floats quantized into an `int8` or `uint8` input with its own scale and
  zero point, and outputs read back as real values.
- **Images.** `<tflite/image>` crops, resizes (bilinear or area) and writes RGBA pixels into
  an NHWC or NCHW input, in RGB or BGR order, float or quantized.
- **Errors as `Result`.** A missing file, a broken model, a wrong type or shape, a delegate
  that is missing or cannot take the model: each comes back as an error message with
  TensorFlow Lite's own explanation, never a crash.
- **The whole API underneath.** `<tflite/raw>` declares all 150 functions of the C API
  (`c_api.h`, `c_api_opaque.h`, `operator.h`, `xnnpack_delegate.h`), generated from the
  TensorFlow 2.21.0 headers, with every struct layout and enum value checked against the C
  compiler.

## Inference

```eskiu
import <io>;
import <tflite>;

Result<int, string> run() {
    TfliteInterpreter it = tflite_load("model.tflite", null)?;
    defer it.release();
    float[4] x = {1.0, -2.0, 0.5, 3.0};
    int[2] shape = {1, 4};
    it.set_f32(0, &x[0], &shape[0], 2)?;
    it.invoke()?;
    *TfliteTensor y = it.output(0);
    *float out = y.f32();
    for (i in 0..y.len()) printf("%f\n", (double)out[i]);
    return Ok<int, string>(0);
}

Result<int, string> r = run();
if (!r.ok) printf("error: %s\n", r.error);
```

`tflite_load(path, opts)` is `tflite_model(path)` followed by `tflite_interpreter(&model,
opts)`. A model can make several interpreters, and can be released once they exist.
`tflite_model_bytes(data, len)` copies the bytes of a `.tflite` file, so the caller's
buffer can go at once.

## Documentation

- [Inference](docs/inference.md): models and interpreters, introspection, inputs, outputs,
- [Images](docs/images.md): cropping, resizing and writing pixels into an input
- [Examples](docs/examples.md): image classification and face anti-spoofing, compared with Python
- [API reference](docs/api.md): every public function and type
- [Platforms](docs/README.md#platforms)
- [Modules](docs/README.md#modules)
- [Contributing](docs/README.md#contributing)
- [API reference](docs/api.md): every public function and type

## Install

In an Eskiu package (Eskiu 0.15.1 or newer, for gpu 0.6.0), on macOS and Linux:

```bash
eskiuc add tflite --git https://github.com/eskiu-lang/tflite --rev v0.1.0
eskiuc build
```

The first build downloads the TensorFlow Lite C library (TensorFlow 2.21) for your platform and checks its SHA-256. See [installation details](docs/README.md#installation-details).

## License

MIT. See [LICENSE](LICENSE). The headers in `tools/include` are TensorFlow's, under the
Apache License 2.0 (`tools/include/LICENSE`).
