# tflite documentation

Guides and reference for tflite. The [README](../README.md) has the overview.

<!-- pages -->
- [Examples](examples.md)
- [Images](images.md)
- [Inference](inference.md)
- [API reference](api.md): every public function and type
<!-- /pages -->

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

## Installation details

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
(Eskiu 0.15.1 or newer, for gpu 0.6.0):

On Windows, and to put the libraries somewhere else, [`tools/get_tflite.sh`](../tools/get_tflite.sh)
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
