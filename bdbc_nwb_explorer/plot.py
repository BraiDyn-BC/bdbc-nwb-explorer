# MIT License
#
# Copyright (c) 2024-2025 Keisuke Sehara
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import Optional
from pathlib import Path

import numpy as _np
import numpy.typing as _npt
import h5py as _h5

MASK_FILE = Path(__file__).parent / 'data' / 'AllenCCF_masks.h5'

MASK_REG: Optional[dict[str, _npt.NDArray[bool]]] = None
MASK_SHAPE: Optional[tuple[int]] = None


def get_roi_masks() -> dict[str, _npt.NDArray[bool]]:
    """returns the 512x512 masks"""
    global MASK_REG
    if MASK_REG is None:
        return load_roi_masks()
    return MASK_REG


def load_roi_masks() -> dict[str, _npt.NDArray[bool]]:
    def _as_name(name, side):
        return f"{name}_{side[0].lower()}"

    global MASK_REG
    global MASK_SHAPE
    MASK_REG = dict()
    with _h5.File(MASK_FILE, 'r') as src:
        for ID in sorted(src['masks'].keys()):
            entry = src['masks'][ID]
            if 'outline' in entry.attrs['name']:
                continue
            mask = (_np.array(entry, copy=False) > 0)
            MASK_REG[_as_name(entry.attrs['name'], entry.attrs['side'])] = mask
            MASK_SHAPE = mask.shape
    return MASK_REG


def convert_to_Allen_CCF(roi_values: dict[str, _np.floating]) -> _npt.NDArray[_np.float32]:
    """
    `roi_values` can be of any type, as long as it implements `keys()` and `__getitem__()`.
    returns a 512x512 float array representing the Allen CCF atlas.
    """
    global MASK_REG
    global MASK_SHAPE
    if MASK_REG is None:
        load_roi_masks()
    CCF = _np.empty(MASK_SHAPE) * _np.nan
    for name, mask in MASK_REG.items():
        if name in roi_values.keys():
            CCF[mask] = roi_values[name]
    return CCF
