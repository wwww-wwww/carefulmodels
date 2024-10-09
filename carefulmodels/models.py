from .inference import *


def banding(clip: vs.VideoNode,
            mask: float | vs.VideoNode,
            backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_rgb_mask(clip, mask, get_model("w-banding-fp16.onnx"), backend)


def h264(clip: vs.VideoNode,
         mask: float | vs.VideoNode,
         backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_rgb_mask(clip, mask, get_model("w-h264-fp16.onnx"), backend)


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
  planes = core.std.SplitPlanes(clip)
  y = planes[0]

  planes[0] = core.std.RemoveFrameProps(planes[0], "_Matrix")
  planes[0] = core.resize.Point(planes[0], format=vs.GRAYH)
  planes[1] = core.resize.Point(planes[1], format=vs.GRAYH)
  planes[2] = core.resize.Point(planes[2], format=vs.GRAYH)

  clip = core.std.ShufflePlanes(planes, [0, 0, 0], vs.RGB)

  clip = inf_rgb_mask(clip, mask, get_model("w-noiseuv-fp16.onnx"), backend)

  clip = core.resize.Point(clip, format=vs.RGB48)
  clip = core.std.SplitPlanes(clip)
  return core.std.ShufflePlanes([y, clip[1], clip[0]], [0, 0, 0], vs.YUV)


def scale(clip: vs.VideoNode,
          mask: float | vs.VideoNode,
          backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_gray_y_mask(clip, mask, get_model("w-scale-fp16.onnx"), backend)


def sharp(clip: vs.VideoNode,
          mask: float | vs.VideoNode,
          backend=vsmlrt.Backend.TRT(fp16=True)):
  return inf_gray_y_mask(clip, mask, get_model("w-sharp-fp16.onnx"), backend)
