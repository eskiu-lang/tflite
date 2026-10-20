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

Inputs and outputs are numbered in model order; `it.input_index("x")` and
`it.output_index("Identity")` find one by name. Before data is copied in, it is checked
against the input:

```
tflite: input 'x' is int64, the model wants float32
tflite: input 'x' cannot have shape [1, 5], the model wants [1, 4]
tflite: no input 3 (the model has 1)
tflite: cannot load model 'm.tflite': Could not open 'm.tflite'; The model allocation is null/empty
```

### Introspection

```eskiu
for (i in 0..it.input_count()) {
    *TfliteTensor t = it.input(i);
    String dims;
    tflite_shape_text(t.signature.data, t.signature.size, &dims);
    printf("%s: %s %s", t.name.data, tflite_type_name(t.type), dims.data);    // x: int8 [1, 8]
    if (t.quantized()) printf(" scale %f zero point %d", (double)t.scale, t.zero_point);
    printf("\n");
    String_free(&dims);
}
```

A `TfliteTensor` has the tensor's `name`, `type` (`TFLITE_FLOAT32`, `TFLITE_INT8`, ...),
`shape` (the current one), `signature` (the model's, with -1 for a dynamic dimension),
`count`, `bytes`, `data`, `scale` and `zero_point` (scale 0 when it is not quantized). The
data belongs to the interpreter.

### Inputs

`it.set_input(i, type, data, shape, rank)` copies data of any element type;
`set_f32`, `set_f16`, `set_i8`, `set_u8`, `set_i32` and `set_i64` take typed pointers.
The type must be the input's and the shape must fit its signature. When the shape differs
from the current one only in dynamic dimensions, the input is resized and the tensors
reallocated first; `it.resize_input(i, shape, rank)` does that on its own.

`it.set_float(i, data, shape, rank)` takes floats whatever the input is: it copies them
into a `float32` input, rounds them into a `float16` one, and quantizes them into an
`int8`, `uint8` or `int16` one with the input's scale and zero point.

### Outputs

After `invoke()`, `it.output(i)` has the output's current shape (a dynamic model can
change it on every run). `t.f32()` (or `f16`, `i8`, `u8`, `i32`, `i64`) is the data, null
when the tensor holds another type; `t.at(k)` reads any numeric element as a `double`,
as stored; `t.value(k)` is the real value, dequantized for a quantized tensor; `t.floats()`
returns a new buffer of every `value`; `t.argmax()` is the index of the largest element.

`tflite_quantize(v, scale, zero_point, type)` and `tflite_dequantize(q, scale,
zero_point)` convert single values (halves round away from zero, as TensorFlow Lite's
`Quantize`), and `tflite_f32_to_f16` and `tflite_f16_to_f32` convert `float16` bits.

### Options and delegates

```eskiu
TfliteOptions o = TfliteOptions{delegate: TFLITE_AUTO, threads: 4, fallback: true};
TfliteInterpreter it = tflite_load("model.tflite", &o)?;
printf("running on %s\n", tflite_delegate_name(it.delegate));
```

| `delegate` | |
|---|---|
| `TFLITE_CPU` | the default: TensorFlow Lite's CPU kernels |
| `TFLITE_XNNPACK` | XNNPACK on every operation it supports, quantized ones included |
| `TFLITE_GPU` | Metal on macOS, OpenCL or OpenGL ES on Linux; `precision_loss` allows 16-bit floats |
| `TFLITE_COREML` | Core ML, with a library that has it (see below) |
| `TFLITE_AUTO` | the GPU when its delegate library is found, else the CPU |

A delegate that is missing or cannot take the model is an error, unless `fallback` is
set, in which case the interpreter runs on the CPU and `it.delegate` says so.
`tflite_has_delegate(TFLITE_GPU)` asks first. The GPU delegates are separate libraries,
loaded at run time from the directory that holds `libtensorflowlite_c` (or from
`delegate_path`), so a program that never asks for them does not need them.

What was verified with the libraries below (TensorFlow Lite 2.21.0):

- **XNNPACK** works on every platform. Python's TensorFlow Lite interpreter applies it by
  default, so `TFLITE_XNNPACK` gives the same numbers as Python, quantized models
  included; the default CPU kernels round some quantized results differently.
- **Metal** works on Apple silicon: MobileNet v2 runs in 1.2 ms instead of 8 ms on one CPU
  thread, with the same top five. Models with dynamic tensors are refused by the delegate
  (and run on the CPU with `fallback`).
- **OpenCL/OpenGL** on Linux x86-64: the delegate library is included and loaded at run
  time; it needs the OpenGL ES and EGL libraries (without them it does not load and is
  reported missing), and it has not been tried on a Linux machine with a GPU. Without a usable GPU the
  delegate cannot be applied, and `fallback` runs the model on the CPU.
- **Core ML**: TensorFlow builds its Core ML delegate for iOS only, and no desktop library
  has it, so `TFLITE_COREML` reports it missing unless `delegate_path` names a library that
  exports `TfLiteCoreMlDelegateCreate`.
- **Windows** has no GPU delegate in this build.

TensorFlow Lite prints a line on stderr when it creates a delegate (`INFO: Created
TensorFlow Lite XNNPACK delegate for CPU.`).

## Images

`<tflite/image>` works on 8-bit RGBA pixels (what most decoders give), and returns new
buffers to `free`:

```eskiu
import <tflite>;
import <tflite/image>;
import <gpu/image>;                  // PNG and JPEG decoding, from the gpu package

Image img = image_load("cat.jpg")?;
defer img.free();
int w = (int)img.width;
int h = (int)img.height;
int side = w < h ? w : h;
*uint8 square = tflite_image_crop(img.rgba, w, h, (w - side) / 2, (h - side) / 2, side, side, TFLITE_BORDER_CLAMP);
defer free(square);
*uint8 small = tflite_image_resize_area(square, side, side, 224, 224);
defer free(small);
TfliteImageFormat f = tflite_format_signed();
it.set_image(0, small, 224, 224, &f)?;    // a [1, 224, 224, 3] input, float or quantized
```

| Function | |
|---|---|
| `tflite_image_crop` | a region that may reach outside the image, which is filled with zeros, the edge pixels or the image reflected about its edge (`TFLITE_BORDER_ZERO`, `_CLAMP`, `_REFLECT`) |
| `tflite_image_resize` | bilinear with half-pixel centers, as TensorFlow's `tf.image.resize`; OpenCV's fixed-point `INTER_LINEAR` can differ by one or two levels |
| `tflite_image_resize_area` | the average of the pixels each output pixel covers, the same as OpenCV's `INTER_AREA` when shrinking |
| `it.set_image(i, rgba, w, h, &f)` | writes the image into input `i` as item 0, resizing an input with a dynamic height and width |
| `tflite_image_into` | writes into item `index` of a tensor, for batches |

`TfliteImageFormat` says how a pixel becomes a value: `(pixel * scale - mean[c]) / std[c]`,
with `nchw` for the layout (NHWC, as TensorFlow Lite models take images, unless set) and
`bgr` for the channel order. An `int8` or `uint8` input gets the value quantized with its
own scale and zero point, so the same format serves a float model and its quantized
version. `tflite_format_signed()` gives -1..1 (MobileNet, EfficientNet-Lite),
`tflite_format_unit()` 0..1, `tflite_format_imagenet()` the ImageNet normalization and
`tflite_format_raw()` 0..255.

Decoding image files is left to other packages. The examples use `<gpu/image>` from
[gpu](https://github.com/eskiu-lang/gpu), a dependency of this package, which decodes PNG
and JPEG in pure Eskiu and needs no GPU. It does not apply the EXIF orientation of camera
photos.

## Examples

[`examples/classify.esk`](examples/classify.esk) prints the top classes of an image. Run
with no arguments, it uses a tiny model in `examples/data` that names the main color of an
image; with an ImageNet classifier it names objects:

```
$ curl -LO https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
$ tar xzf mobilenet_v2_1.0_224.tgz
$ curl -LO https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt
$ curl -LO https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg
$ eskiuc build -L$TFL/lib --link-arg=-Wl,-rpath,$TFL/lib
$ build/tflite --model mobilenet_v2_1.0_224.tflite --labels ImageNetLabels.txt grace_hopper.jpg
mobilenet_v2_1.0_224.tflite on grace_hopper.jpg (CPU, float32 NHWC), 224x224 input
 35.91%  military uniform
 20.60%  mortarboard
  6.79%  academic gown
  5.25%  suit
  2.88%  bearskin
```

Python's TensorFlow Lite interpreter gives the same percentages for the same input tensor,
and 35.97%, 20.71%, 6.81%, 5.23% and 2.89% when the image is decoded by Pillow instead
(the two JPEG decoders differ by one level on a few pixels). With the quantized
`mobilenet_v1_1.0_224_quant.tflite` and `--delegate xnnpack`, the output (71.48% military
uniform, 13.67% Windsor tie) is Python's to the last digit; the default CPU kernels give
76.56% and 10.55%, as Python does without its default delegate. `--format` picks the
pixel values (`signed` by default) and `--delegate` the delegate.

[`examples/minifasnet.esk`](examples/minifasnet.esk) is face anti-spoofing: it says whether a
face is a live person or a photo or screen held up to the camera. It takes Minivision's
MiniFASNetV2 (from [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing),
Apache-2.0) converted to TensorFlow Lite, as published in
[Silent-Face-Anti-Spoofing-TFLite](https://github.com/feni-katharotiya/Silent-Face-Anti-Spoofing-TFLite)
(Apache-2.0), an image and the face's box from a face detector:

```
$ curl -LO https://raw.githubusercontent.com/feni-katharotiya/Silent-Face-Anti-Spoofing-TFLite/HEAD/Model_Encryption/matrix/converted_models/tflite_f32/2.7_80x80_MiniFASNetV2.tflite
$ eskiuc -L$TFL/lib --link-arg=-Wl,-rpath,$TFL/lib examples/minifasnet.esk -o build/minifasnet
$ build/minifasnet 2.7_80x80_MiniFASNetV2.tflite image_T1.png 106 147 313 360
box 106 147 313 360, crop 0 7 479 499, logits -4.364262 4.530829 -0.169430
real (real 99.09%, CPU)
$ build/minifasnet 2.7_80x80_MiniFASNetV2.tflite image_F1.png 178 136 401 361
box 178 136 401 361, crop 0 6 479 490, logits 0.588243 -3.576411 2.989823
spoof (real 0.13%, CPU)
```

It follows the project's Python preprocessing (a box 2.7 times the face's, moved inside
the image, resized to 80x80, BGR in 0..255). The images are the project's
`images/sample/image_T1.jpg` (a live face) and `image_F1.jpg` (a photo of one), turned
upright as their EXIF orientation says, with the boxes its face detector found. Python's
TensorFlow Lite interpreter, on the same crop resized by `tf.image.resize`, gives the same
logits for these two images to five decimals. Model files are not part of this
repository.

The [ONNX MiniFASNet](https://github.com/facenox/face-antispoof-onnx) used by
[onnx](https://github.com/eskiu-lang/onnx)'s example has no TensorFlow Lite release; a
converter such as [onnx2tf](https://github.com/PINTO0309/onnx2tf) (`onnx2tf -i
best_model.onnx`) can make an NHWC `.tflite` of it (not tried here), which would take a
128x128 RGB crop in 0..1 (`tflite_format_unit()`) and give two logits.

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
with Bazel for each platform. [`tools/get_tflite.sh`](tools/get_tflite.sh) downloads that
release (TensorFlow 2.21.0, 55 MB), checks its SHA-256, and copies this platform's files
into `DIR/lib`:

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

On Windows, run it from an MSYS2 shell. Then, in an Eskiu package (Eskiu 0.12 or newer):

```bash
eskiuc add tflite --git https://github.com/eskiu-lang/tflite --rev v0.1.0
eskiuc build -L$TFL/lib --link-arg=-Wl,-rpath,$TFL/lib     # macOS and Linux
```

On Windows (MinGW), `-L` the directory that holds `tensorflowlite_c.dll`, which MinGW
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
