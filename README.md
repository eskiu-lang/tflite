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

## Guides

- [Inference](docs/inference.md): models and interpreters, introspection, inputs, outputs,
  errors, options and delegates (what was verified on each GPU)
- [Images](docs/images.md): cropping, resizing and writing pixels into an input
- [Examples](docs/examples.md): image classification and face anti-spoofing, compared with Python
- [API reference](docs/api.md): every public function and type

## Platforms

| Platform | Checked |
|---|---|
| macOS arm64 | CI |
| Linux x86-64 | CI |
| Windows x86-64 (MinGW) | CI |
| iOS, Android | not yet |
| WebAssembly | not yet |

The bindings follow the 64-bit prebuilt libraries that `tools/get_tflite.sh` downloads, so
building `<tflite>` for a phone, for WebAssembly or for a 32-bit target stops with an
`#error`. TensorFlow Lite has iOS and Android builds of its C library; supporting them here
is planned. In a browser, TensorFlow Lite runs from JavaScript instead.

## Install

Google publishes no desktop build of the TensorFlow Lite C library: Homebrew has no
formula, Google's LiteRT releases ship `TensorFlowLiteC` for iOS only, and the
`ai-edge-litert` and `tflite-runtime` Python wheels carry the Python bindings and the newer
LiteRT API, not `libtensorflowlite_c`. The package uses the libraries from the Unity
package of [asus4/tf-lite-unity-sample](https://github.com/asus4/tf-lite-unity-sample)
(Apache-2.0), which builds TensorFlow's own `tensorflow/lite/c:tensorflowlite_c` target
with Bazel for each platform.

On macOS and Linux there is nothing to install. The package's `eskiu.toml` names that
release (TensorFlow 2.21.0, 55 MB): the first build that links it (`eskiuc build`, `run`,
`test`) downloads it once, checks its SHA-256 and keeps it in `~/.eskiu/native`, and every
build links against this platform's directory of it with no `-L`. In an Eskiu package
(Eskiu 0.12 or newer):

```bash
eskiuc add tflite --git https://github.com/eskiu-lang/tflite --rev v0.1.0
eskiuc build
```

On Windows, and to put the libraries somewhere else, [`tools/get_tflite.sh`](tools/get_tflite.sh)
downloads the same release, checks it, and copies this platform's files into `DIR/lib`:

```bash
curl -fsSL https://raw.githubusercontent.com/eskiu-lang/tflite/v0.1.0/tools/get_tflite.sh | sh -s -- ~/tflite
export TFL=~/tflite
```

| Platform | `$TFL/lib` |
|---|---|
| macOS (arm64 and x86-64) | `libtensorflowlite_c.dylib`, `libtensorflowlite_metal_delegate.dylib` |
| Linux x86-64 | `libtensorflowlite_c.so`, `libtensorflowlite_gpu_delegate.so` |
| Linux arm64 | `libtensorflowlite_c.so` |
| Windows x86-64 | `tensorflowlite_c.dll` (needs the Visual C++ 2015-2022 runtime) |

On Windows, run it from an MSYS2 shell, then `-L` the directory that holds `tensorflowlite_c.dll`, which MinGW
links against directly (the file must keep that name, the one the DLL gives itself), and put the DLL next to the executable.

TensorFlow also documents a CMake build of the C library (`tensorflow/lite/c` at the
`v2.21.0` tag), which this package does not use or test; any `libtensorflowlite_c` of
version 2.21 in a directory given to `-L` should work, and `tools/layout_check.sh` tells
whether it exports every function.

## Modules

| Module | What it has |
|---|---|
| `<tflite>` | `tflite_model`, `tflite_model_bytes`, `tflite_interpreter`, `tflite_load`, `TfliteInterpreter` (`input`, `output`, `set_*`, `resize_input`, `invoke`), `TfliteTensor`, `TfliteOptions`, delegates, quantization and float16 conversions |
| `<tflite/image>` | `tflite_image_crop`, `tflite_image_resize`, `tflite_image_resize_area`, `tflite_image_into`, `TfliteInterpreter_set_image`, `TfliteImageFormat` |
| `<tflite/raw>` | the TensorFlow Lite C API, generated from its headers |

Models and interpreters have a `release()` to pair with `defer`. `TfliteModel.handle`,
`TfliteInterpreter.handle` and `TfliteTensor.handle` are the `<tflite/raw>` handles, so a
program can drop down to the C API for anything the idiomatic layer does not cover
(signature runners, custom operators, cancellation, writing into `t.data` directly).

### The raw API

The C API is made of ordinary exported functions, so `<tflite/raw>` declares each one as
an `extern` with its C name and signature: `TfLiteInterpreterCreate(model, options)`,
`TfLiteTensorQuantizationParams(t)` (which returns a struct by value, lowered to the C
ABI), `TfLiteXNNPackDelegateOptionsDefault()`. Enums are Eskiu enums, the option structs
of the XNNPACK, GPU, Metal and Core ML delegates are Eskiu structs, and the other types
are opaque structs used by pointer. The 16 functions of `c_api_opaque.h` that the Windows
DLL does not export are declared on macOS and Linux only.

## Contributing

Bug reports are welcome; please include the program, the `eskiuc --version` output and the
TensorFlow Lite version (`tflite_version()`). The tests and the layout check run with:

```bash
eskiuc test -L$TFL/lib --link-arg=-Wl,-rpath,$TFL/lib
python3 -I tools/gen.py                         # regenerates src/raw.esk from tools/include
python3 -I tools/gen_models.py                  # regenerates the test models
tools/layout_check.sh $TFL/lib
```

`tools/include` holds the C API headers of TensorFlow's `v2.21.0` tag, with TensorFlow's
license. The test models are made with TensorFlow 2.21 and their expected outputs come
from the `ai-edge-litert` interpreter. `tools/layout_check.sh` compiles a C program and an
Eskiu program that print every struct's size and field offsets and every enum value, and
compares their output; the Eskiu program also takes the address of every function, so it
links only if the library exports them all. CI runs it and the tests on Linux, macOS and
Windows.

## License

MIT. See [LICENSE](LICENSE). The headers in `tools/include` are TensorFlow's, under the
Apache License 2.0 (`tools/include/LICENSE`).
