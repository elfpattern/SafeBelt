"""
Standard imagenet crop method for training and testing
Entry: imagenet_standard_crop
"""

from typing import Tuple
import random
import math

BOX = Tuple[int, int, int, int]
FLOAT_RANGE = Tuple[float, float]

def imagenet_standard_crop(width: int, height: int, complexity: int, phase: str, \
        *, standard: str="latest") -> BOX:
    """
    Get standard crop box

    Parameters:
        width       - image width
        height      - image height
        complexity  - model complexity in MFLOPs
        phase       - in { "TRAIN", "TEST" }
        standard    - in { "latest", "v1" }

    Returns
        (left, top, right, bottom) - where crop region lies in [left, right) and [top, bottom)
    """
    assert width > 0 and height > 0
    phase = phase.upper()
    assert phase in {"TRAIN", "TEST"}
    standard = standard.lower()
    crop_fun = _crop_train if phase == "TRAIN" else _crop_test
    return crop_fun(width, height, complexity, standard)

def _crop_train(width: int, height: int, complexity: int, standard: str) -> BOX:
    assert standard in _CROP_TRAIN_VERSIONS
    return _CROP_TRAIN_VERSIONS[standard](width, height, complexity)

def _crop_test(width: int, height: int, complexity: int, standard: str) -> BOX:
    assert standard in _CROP_TEST_VERSIONS
    return _CROP_TEST_VERSIONS[standard](width, height, complexity)

def _crop_train_v1(width: int, height: int, complexity: int) -> BOX:
    if complexity <= 500:
        box = _rand_crop_with_jitter(width, height, (0.49, 1), (3. / 4, 4. / 3))
    else:
        box = _rand_crop_with_jitter(width, height, (0.08, 1), (3. / 4, 4. / 3))

    if complexity > 2000:
        box = _add_rand_aspect_ratio(width, height, box, 0.75)

    return box

def _crop_test_v1(width: int, height: int, _) -> BOX:
    return _center_crop(width, height, 0.875)

def _center_crop(width: int, height: int, ratio: float) -> BOX:
    assert ratio > 0 and ratio <= 1
    crop_size = int(min(width, height) * ratio)
    crop_size = max(crop_size, 1)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return left, top, left + crop_size, top + crop_size

def _rand_crop(width: int, height: int, crop_width: int, crop_height: int) -> BOX:
    assert width >= crop_width and height >= crop_height
    assert crop_width > 0 and crop_height > 0
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    return left, top, left + crop_width, top + crop_height

def _rand_crop_with_jitter(width: int, height: int, ratio_range: FLOAT_RANGE, \
        aspect_range: FLOAT_RANGE) -> BOX:
    assert len(ratio_range) == 2 and ratio_range[0] <= ratio_range[1] and ratio_range[0] > 0
    assert len(aspect_range) == 2 and aspect_range[0] <= aspect_range[1] and aspect_range[0] > 0
    area = float(width * height)

    for _ in range(10):
        target_area = random.uniform(ratio_range[0], ratio_range[1]) * area
        aspect_ratio = random.uniform(aspect_range[0], aspect_range[1])
        target_w = max(round(math.sqrt(target_area * aspect_ratio)), 1)
        target_h = max(round(math.sqrt(target_area / aspect_ratio)), 1)
        if random.uniform(0, 1) < 0.5:
            target_w, target_h = target_h, target_w

        if target_w <= width and target_h <= height:
            return _rand_crop(width, height, target_w, target_h)

    crop_size = min(width, height)
    return _rand_crop(width, height, crop_size, crop_size)

def _add_rand_aspect_ratio(width: int, height: int, box: BOX, max_ratio: float) -> BOX:
    assert max_ratio > 0
    min_ratio = 1.
    if min_ratio > max_ratio:
        min_ratio, max_ratio = max_ratio, min_ratio

    sel_ratio = random.uniform(min_ratio, max_ratio)
    if random.uniform(0, 1) < 0.5:
        sel_ratio = 1 / sel_ratio

    area = float((box[2] - box[0]) * (box[3] - box[1]))
    new_h = max(round(math.sqrt(area) / sel_ratio), 1)
    new_w = max(round(new_h * sel_ratio), 1)
    center_x = box[0] + (box[2] - box[0]) // 2
    center_y = box[1] + (box[3] - box[1]) // 2
    new_left = max(center_x - new_w // 2, 0)
    new_top = max(center_y - new_h // 2, 0)

    new_right = min(new_left + new_w, width)
    new_bottom = min(new_top + new_h, height)

    return new_left, new_top, new_right, new_bottom

_CROP_TRAIN_VERSIONS = {
    "latest": _crop_train_v1,
    "v0": _crop_train_v1
    }

_CROP_TEST_VERSIONS = {
    "latest": _crop_test_v1,
    "v0": _crop_test_v1
    }
