from .inference import *


def banding(clip: vs.VideoNode,
            mask: float | vs.VideoNode,
            backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_rgb_mask(clip, mask, get_model("w-banding-fp16.onnx"), backend)


def h264(clip: vs.VideoNode,
         mask: float | vs.VideoNode,
         backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_yuv_mask(clip, mask, get_model("w-h264yuv-fp16.onnx"), backend)


def mpeg2(clip: vs.VideoNode,
          mask: float | vs.VideoNode,
          backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_rgb_mask(clip, mask, get_model("w-mpeg2-fp16.onnx"), backend)


def noise(clip: vs.VideoNode,
          mask: float | vs.VideoNode,
          backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_gray_y_mask(clip, mask, get_model("w-noise-fp16.onnx"), backend)


def noiseuv(clip: vs.VideoNode,
            mask: float | vs.VideoNode,
            backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_yuv_mask(clip,
                      mask,
                      get_model("w-noiseuv2-fp16.onnx"),
                      backend,
                      keep=[0])


def scale(clip: vs.VideoNode,
          mask: float | vs.VideoNode,
          backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_gray_y_mask(clip, mask, get_model("w-scale-fp16.onnx"), backend)


def sharp(clip: vs.VideoNode,
          mask: float | vs.VideoNode,
          backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_gray_y_mask(clip, mask, get_model("w-sharp-fp16.onnx"), backend)
