from .inference import *


def banding(clip, mask, backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_rgb_mask(clip, mask, get_model("w-banding-fp16.onnx"), backend)


def h264(clip, mask, backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_rgb_mask(clip, mask, get_model("w-h264-fp16.onnx"), backend)


def scale(clip, mask, backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_gray_y_mask(clip, mask, get_model("w-scale-fp16.onnx"), backend)


def sharp(clip, mask, backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_gray_y_mask(clip, mask, get_model("w-sharp-fp16.onnx"), backend)


def mpeg2(clip, mask, backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_rgb_mask(clip, mask, get_model("w-mpeg2-fp16.onnx"), backend)
