# Images

Preparing image inputs with `<tflite/image>`. Every function and type is also in the [API reference](api.md).

`<tflite/image>` works on 8-bit RGBA pixels (what most decoders give), and returns new
buffers to `free`:

```eskiu
import <tflite>;
import <tflite/image>;
import <gpu/image>;                  // PNG and JPEG decoding, from the gpu package

Bitmap img = image_load("cat.jpg")?;
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
