import vapoursynth as vs
import vsmlrt
import os

core = vs.core


def get_model(path):
  return os.path.join(os.path.dirname(__file__), "models", path)


def inf_rgb_mask(clip, mask, model, backend=vsmlrt.Backend.TRT(fp16=True)):
  input_fmt = clip.format
  if input_fmt.color_family != vs.YUV and input_fmt.color_family != vs.RGB:
    raise Exception("clip must be YUV or RGB")

  if type(mask) != vs.VideoNode:
    mask = core.std.BlankClip(clip, format=vs.GRAYH, color=mask)

  if input_fmt.color_family == vs.YUV:
    clip = core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")

  inf = vsmlrt.inference([clip, mask], model, backend=backend)

  if input_fmt.color_family == vs.YUV:
    fmt = input_fmt.replace(bits_per_sample=16,
                            subsampling_w=0,
                            subsampling_h=0)
    inf = core.resize.Bicubic(inf, format=fmt, matrix_s="709")

  return inf


def inf_gray_mask(clip, mask, model, backend=vsmlrt.Backend.TRT(fp16=True)):
  if clip.format.color_family != vs.GRAY:
    raise Exception("clip must be GRAY")

  if type(mask) != vs.VideoNode:
    mask = core.std.BlankClip(clip, format=vs.GRAYH, color=mask)

  clip = core.resize.Bicubic(clip, format=vs.GRAYH)

  clip = vsmlrt.inference([clip, mask], model, backend=backend)

  clip = core.resize.Bicubic(clip, format=vs.GRAY16)

  return clip


def inf_gray_3_mask(clip, mask, model, backend, planes=[0, 1, 2]):
  if clip.format.color_family != vs.YUV and clip.format.color_family != vs.RGB:
    raise Exception("clip must be YUV or RGB")

  split = core.std.SplitPlanes(clip)

  for p in planes:
    split[p] = inf_gray_mask(split[p], mask, model, backend)

  for p in range(3):
    if p not in planes:
      split[p] = core.resize.Point(split[p], format=vs.GRAY16)

  clip = core.std.ShufflePlanes(split, [0, 0, 0], clip.format.color_family)
  return clip


def inf_gray_y_mask(clip, mask, model, backend):
  if clip.format.color_family == vs.GRAY:
    return inf_gray_mask(clip, mask, model, backend)

  return inf_gray_3_mask(clip, mask, model, backend, planes=[0])
