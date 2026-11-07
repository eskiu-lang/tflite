# Examples

The programs in [`examples/`](../examples), with what they print.

[`examples/classify.esk`](../examples/classify.esk) prints the top classes of an image. Run
with no arguments, it uses a tiny model in `examples/data` that names the main color of an
image; with an ImageNet classifier it names objects:

```
$ curl -LO https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
$ tar xzf mobilenet_v2_1.0_224.tgz
$ curl -LO https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt
$ curl -LO https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg
$ eskiuc build
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

[`examples/minifasnet.esk`](../examples/minifasnet.esk) is face anti-spoofing: it says whether a
face is a live person or a photo or screen held up to the camera. It takes Minivision's
MiniFASNetV2 (from [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing),
Apache-2.0) converted to TensorFlow Lite, as published in
[Silent-Face-Anti-Spoofing-TFLite](https://github.com/feni-katharotiya/Silent-Face-Anti-Spoofing-TFLite)
(Apache-2.0), an image and the face's box from a face detector:

```
$ curl -LO https://raw.githubusercontent.com/feni-katharotiya/Silent-Face-Anti-Spoofing-TFLite/HEAD/Model_Encryption/matrix/converted_models/tflite_f32/2.7_80x80_MiniFASNetV2.tflite
$ eskiuc examples/minifasnet.esk -o build/minifasnet
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
