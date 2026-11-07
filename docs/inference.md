# Inference

Models, interpreters, inputs, outputs and delegates in `<tflite>`. Every function and type is also in the [API reference](api.md).

## Finding inputs and outputs

Inputs and outputs are numbered in model order; `it.input_index("x")` and
`it.output_index("Identity")` find one by name. Before data is copied in, it is checked
against the input:

```
tflite: input 'x' is int64, the model wants float32
tflite: input 'x' cannot have shape [1, 5], the model wants [1, 4]
tflite: no input 3 (the model has 1)
tflite: cannot load model 'm.tflite': Could not open 'm.tflite'; The model allocation is null/empty
```

## Introspection

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

## Inputs

`it.set_input(i, type, data, shape, rank)` copies data of any element type;
`set_f32`, `set_f16`, `set_i8`, `set_u8`, `set_i32` and `set_i64` take typed pointers.
The type must be the input's and the shape must fit its signature. When the shape differs
from the current one only in dynamic dimensions, the input is resized and the tensors
reallocated first; `it.resize_input(i, shape, rank)` does that on its own.

`it.set_float(i, data, shape, rank)` takes floats whatever the input is: it copies them
into a `float32` input, rounds them into a `float16` one, and quantizes them into an
`int8`, `uint8` or `int16` one with the input's scale and zero point.

## Outputs

After `invoke()`, `it.output(i)` has the output's current shape (a dynamic model can
change it on every run). `t.f32()` (or `f16`, `i8`, `u8`, `i32`, `i64`) is the data, null
when the tensor holds another type; `t.at(k)` reads any numeric element as a `double`,
as stored; `t.value(k)` is the real value, dequantized for a quantized tensor; `t.floats()`
returns a new buffer of every `value`; `t.argmax()` is the index of the largest element.

`tflite_quantize(v, scale, zero_point, type)` and `tflite_dequantize(q, scale,
zero_point)` convert single values (halves round away from zero, as TensorFlow Lite's
`Quantize`), and `tflite_f32_to_f16` and `tflite_f16_to_f32` convert `float16` bits.

## Options and delegates

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
