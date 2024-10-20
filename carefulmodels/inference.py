import vapoursynth as vs
import vsmlrt
import os

core = vs.core


def get_model(path):
  return os.path.join(os.path.dirname(__file__), "models", path)


def inf_rgb_mask(clip, mask, model, backend):
  input_fmt = clip.format
  if input_fmt.color_family != vs.YUV and input_fmt.color_family != vs.RGB:
    raise Exception("clip must be YUV or RGB")

  if not isinstance(mask, vs.VideoNode):
    mask = core.std.BlankClip(clip, format=vs.GRAYH, color=mask)

  if input_fmt.color_family == vs.YUV:
    clip = core.resize.Point(clip, format=vs.RGBH, matrix_in_s="709")

  inf = vsmlrt.inference([clip, mask], model, backend=backend)

  if input_fmt.color_family == vs.YUV:
    inf = core.resize.Point(inf, format=vs.YUV444P16, matrix_s="709")

  return inf


def inf_gray_mask(clip, mask, model, backend):
  if clip.format.color_family != vs.GRAY:
    raise Exception("clip must be GRAY")

  if not isinstance(mask, vs.VideoNode):
    mask = core.std.BlankClip(clip, format=vs.GRAYH, color=mask)

  clip = core.resize.Point(clip, format=vs.GRAYH)

  clip = vsmlrt.inference([clip, mask], model, backend=backend)

  clip = core.resize.Point(clip, format=vs.GRAY16)

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


def inf_yuv_mask(clip, mask, model, backend, keep=[]):
  if clip.format.color_family != vs.YUV or clip.format.subsampling_h != 0 or clip.format.subsampling_w != 0:
    raise Exception("clip must be YUV444")

  planes = core.std.SplitPlanes(clip)
  old_planes = planes.copy()

  planes[0] = core.std.RemoveFrameProps(planes[0], "_Matrix")
  planes[0] = core.resize.Point(planes[0], format=vs.GRAYH)
  planes[1] = core.resize.Point(planes[1], format=vs.GRAYH)
  planes[2] = core.resize.Point(planes[2], format=vs.GRAYH)

  clip = core.std.ShufflePlanes(planes, [0, 0, 0], vs.RGB)

  clip = inf_rgb_mask(clip, mask, model, backend)

  clip = core.resize.Point(clip, format=vs.RGB48)
  clip = core.std.SplitPlanes(clip)

  for p in keep:
    clip[p] = core.resize.Point(old_planes[p], format=vs.GRAY16)

  clip[0] = core.std.CopyFrameProps(clip[0], old_planes[0], "_Matrix")
  return core.std.ShufflePlanes(clip, [0, 0, 0], vs.YUV)
