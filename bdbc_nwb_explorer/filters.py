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
"""filtering functions for NWBEntrySet objects."""
from typing import Callable, Optional
from collections import namedtuple

import numpy as _np
import numpy.typing as _npt
import pandas as _pd

from . import io as _io


NWBDataEntry = _io.NWBDataEntry
EntryData = _pd.Series | _pd.DataFrame
ArrayFilter = Callable[[_npt.NDArray], _npt.NDArray]
EntryFilter = Callable[[NWBDataEntry], NWBDataEntry]


Conversion = namedtuple('Conversion', ('data', 'metadata'))


def foreach(fn: ArrayFilter) -> EntryFilter:
    """returns a filter that works by applying `fn` to the data of each entry.
    `fn` is supposed to take one 1-d numpy array, and return one 1-d numpy array.
    """
    def convert(data: EntryData) -> EntryData:
        data = data.copy()
        if isinstance(data, _pd.Series):
            data.values[:] = fn(data.values)
        elif isinstance(data, _pd.DataFrame):
            for col in data.columns:
                data[col] = fn(data[col])
        else:
            raise ValueError(f"expected Series or DataFrame, got {data.__class__}")
        return data

    def apply_array_filter(entry: NWBDataEntry) -> NWBDataEntry:
        return entry.replace(data=convert(entry.data))

    return apply_array_filter


def zscore() -> EntryFilter:
    """returns a filter that can be used to standardize the data of each entry."""
    def convert(data: EntryData) -> Conversion:
        data = data.copy()
        if isinstance(data, _pd.Series):
            mu = _np.nanmean(data)
            sig = _np.nanstd(data)
            data.values[:] = (data.values - mu) / sig
        elif isinstance(data, _pd.DataFrame):
            arr = data.to_numpy()
            mu = _np.nanmean(arr, axis=0, keepdims=True)
            sig = _np.nanstd(arr, axis=0, keepdims=True)
            arr = (arr - mu) / sig
            for i, col in enumerate(data.columns):
                data[col] = arr[:, i]
        else:
            raise ValueError(f"expected Series or DataFrame, got {data.__class__}")
        return Conversion(data, dict(mean=mu, std=sig))

    def zscore_entry(entry: NWBDataEntry) -> NWBDataEntry:
        data, metadata = convert(entry.data)
        return entry.replace(data=data, datatype='Z-score', unit='a.u.', metadata=metadata)

    return zscore_entry


def normalize(
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    smin: float = 0,
    smax: float = 1,
) -> EntryFilter:
    """
    returns a filter that applies a linear scaling to the data of each entry,
    so that the values in range [vmin, vmax] will be scaled to [smin, smax].
    Without any specification, [vmin, vmax] will be set to the [min, max] of
    the source values.
    """
    ref = dict(vmin=vmin, vmax=vmax, smin=smin, smax=smax)

    def convert(data: EntryData) -> Conversion:
        data = data.copy()
        if isinstance(data, _pd.Series):
            if ref['vmin'] is None:
                vmin = _np.nanmin(data)
            else:
                vmin = ref['vmin']
            if ref['vmax'] is None:
                vmax = _np.nanmax(data)
            else:
                vmax = ref['vmax']
            data.values[:] = (data - vmin) * (smax - smin) / (vmax - vmin) + smin
        elif isinstance(data, _pd.DataFrame):
            arr = data.to_numpy()
            if ref['vmin'] is None:
                vmin = _np.nanmin(arr, axis=0, keepdims=True)
            else:
                vmin = ref['vmin']
            if ref['vmax'] is None:
                vmax = _np.nanmax(arr, axis=0, keepdims=True)
            arr = (arr - vmin) * (smax - smin) / (vmax - vmin) + smin
        else:
            raise ValueError(f"expected Series or DataFrame, got {data.__class__}")
        return Conversion(data, dict(input_range=(vmin, vmax), output_range=(smin, smax)))

    def normalize_entry(entry: NWBDataEntry) -> NWBDataEntry:
        data, metadata = convert(entry.data)
        return entry.replace(data=data, datatype='Normalized-value', unit='a.u.', metadata=metadata)

    return normalize_entry


def clip_percentile(
    q_lower: Optional[float] = None,
    q_upper: Optional[float] = None,
) -> EntryFilter:
    """
    returns a filter that "clips" the data of each entry based on the percentile thresholds,

    i.e. sets the timepoints that are above the `q_upper` percentile to NaN,
    and sets the timepoints that are below the `q_lower` percentile to NaN.
    """
    def convert(data: EntryData) -> Conversion:
        data = data.copy()
        if isinstance(data, _pd.Series):
            if q_lower is not None:
                vmin = _np.nanpercentile(data, q_lower)
                data.values[data.values < vmin] = _np.nan
            else:
                vmin = None
            if q_upper is not None:
                vmax = _np.nanpercentile(data, q_upper)
                data.values[data.values > vmax] = _np.nan
            else:
                vmax = None
        elif isinstance(data, _pd.DataFrame):
            arr = data.to_numpy()
            if q_lower is not None:
                vmin = _np.nanpercentile(arr, axis=0, keepdims=True)
                arr[arr < vmin] = _np.nan
            else:
                vmin = None
            if q_upper is not None:
                vmax = _np.nanpercentile(arr, axis=0, keepdims=True)
                arr[arr > vmax] = _np.nan
            else:
                vmax = None
            for i, col in enumerate(data.columns):
                data[col] = arr[:, i]
        else:
            raise ValueError(f"expected Series or DataFrame, got {data.__class__}")
        return Conversion(data, dict(vmin=vmin, vmax=vmax))

    def clip_entry_percentile(entry: NWBDataEntry) -> NWBDataEntry:
        data, metadata = convert(entry.data)
        return entry.replace(data=data, metadata=metadata)

    return clip_entry_percentile


def take_axis(axis: str = 'x') -> EntryFilter:
    """
    returns a filter that returns the entries only containing the specified axis (in case it exists).

    The filter has no effect to entries that does _not_ contain the axis with the specified name,
    _or_ the data is 1-d (i.e. not having multiple columns/axis).
    """
    def convert(data: EntryData) -> Conversion:
        if isinstance(data, _pd.Series):
            return Conversion(data, None)
        elif isinstance(data, _pd.DataFrame):
            if axis in data.columns:
                data = data[axis]
            return Conversion(data, None)
        else:
            raise ValueError(f"expected Series or DataFrame, got {data.__class__}")

    def take_axis_from_entry(entry: NWBDataEntry) -> NWBDataEntry:
        data, _ = convert(entry.data)
        return entry.replace(data=data)

    return take_axis_from_entry
