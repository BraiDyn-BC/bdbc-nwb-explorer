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
from typing import Optional, NamedTuple
import warnings as _warnings

import numpy as _np
import numpy.typing as _npt
import pandas as _pd


class AlignmentWarning(UserWarning):
    pass


class PETH(NamedTuple):
    """the tuple representing a set of peri-event time histograms."""
    lags: _npt.NDArray
    values: _npt.NDArray


class Correlogram(NamedTuple):
    """the tuple representing a correlogram."""
    lags: _npt.NDArray
    coefs: _npt.NDArray


def pearson_r(
    x: _npt.NDArray,
    y: _npt.NDArray
) -> float:
    """computes the correlation coefficient (Pearson's R),
    in a NaN-aware fashion."""
    x = (x - _np.nanmean(x)) / _np.nanstd(x)
    y = (y - _np.nanmean(y)) / _np.nanstd(y)
    return float(_np.nanmean(x * y))


def triggers_to_indices(
    t: _npt.NDArray,
    triggers: _npt.NDArray,
    tol: float = 1e-6,
) -> _npt.NDArray[_np.int32]:
    """locates the positions of `triggers` (time points) on the given timebase `t`
    and returns their indices on `t`."""
    in_range = _np.logical_and(
        _np.logical_and(triggers >= t.min(), triggers <= t.max()),
        ~_np.isnan(triggers),
    )
    triggers = triggers[in_range]
    indices = _np.empty((triggers.size,), dtype=_np.int32)
    for i, trig in enumerate(triggers):
        deviations = _np.abs(t - trig)
        candidates = _np.where(deviations < tol)[0]
        if candidates.size == 1:
            indices[i] = candidates[0]
        elif candidates.size == 0:
            _warnings.warn(
                f"failed to locate on timebase: t={trig}",
                category=AlignmentWarning
            )
            indices[i] = deviations.argmin()
        else:
            _warnings.warn(
                f"multiple matches for t={trig} ({candidates.size} found)",
                category=AlignmentWarning
            )
            indices[i] = deviations.argmin()
    return indices


def peth_1d(
    x: _npt.NDArray,
    triggers: _npt.NDArray[_np.integer],
    pretrigger: int = 30,
    posttrigger: int = 60,
    rate: Optional[float] = None,
    baseline: Optional[slice] = None,
    reduction=_np.nanmean,
    dtype=_np.float32,
) -> PETH:
    """
    computes a set of peri-event time histograms (PETHs),
    i.e. a set of 1-D traces aligned to a set of specified events
    (or triggers).

    arguments
    ---------
    x (array): 1-D array, in shape (num_samples,)

    triggers (non-negative integer array): a set of indices.
       it is assumed to, but does not have to, be limited to
       0 <= triggers <= num_samples.

    pretrigger, posttrigger (int): specify the extent to which the
       aligned traces will be extracted. The resulting lags around
       each trigger, in samples, would be [-pretrigger, posttrigger-1].
       By default, pretrigger is 30 and posttrigger is 60.

    rate (float, optional): if specified, the values of resulting
       `lags` will be divided with this value (for the ease of use
       in plotting).

    baseline (slice, optional): if specified, the procedure performs
       baseline subtraction. For example, specifying `slice(None, pretrigger)`
       will result in considering the whole pre-trigger period to be
       the "baseline" period. By default, no baseline subtraction will
       occur.

    reduction (default: numpy.nanmean): the method to compute
       baseline; matters only when baseline subtraction is specified.
       Any function that receives a numpy array and returns a value
       may be specified here.

    dtype (default: numpy float32): the data type of the array to be
       returned.
    """
    triggers = triggers.astype(_np.int32).ravel()
    start_idxx = triggers - pretrigger
    stop_idxx  = triggers + posttrigger
    in_range = _np.logical_and(start_idxx >= 0, stop_idxx < x.size)
    start_idxx = start_idxx[in_range]
    stop_idxx  = stop_idxx[in_range]
    out = _np.empty(
        (posttrigger + pretrigger, start_idxx.size),
        dtype=dtype,
    )
    for i, start, stop in zip(range(start_idxx.size), start_idxx, stop_idxx):
        out[:, i] = x[start:stop]
    if baseline is not None:
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore', category=RuntimeWarning)
            out = out - reduction(out[baseline, :], axis=0)
    lags = _np.arange(pretrigger + posttrigger) - pretrigger
    if rate is not None:
        lags = lags / rate
    return PETH(lags, out)


def correlogram(
    target: _npt.NDArray,
    reference: Optional[_npt.NDArray] = None,
    standardize: bool = True,
    rate: Optional[float] = None,
) -> Correlogram:
    """
    computes a correlogram for `target` with respect to `reference`.

    **IMPORTANT**: this algorithm internally use `numpy.correlate`.
    Computing will fail if `target` or `reference` contains any NaN.

    parameters
    ----------
    target (numpy array): the values to compute the correlogram for.

    reference (numpy array, optional): the reference to compute the
       correlogram based on. If not specified, the target values
       themselves will be used (i.e. computes an auto-correlogram).

    standardize (bool, default: True): whether or not to compute
       Z-scores of the values before computing (which should
       correspond to the strict definition of a correlogram).

    rate (float, optional): the sampling rate of the series.
       If specified, the resulting values of `lags` will be
       divided by this value (for the ease of use in plotting).
    """
    if reference is None:
        reference = target

    if standardize:
        target = (target - _np.nanmean(target)) / _np.nanstd(target)
        reference = (reference - _np.nanmean(reference)) / _np.nanstd(reference)

    N = target.size
    C = _np.correlate(target, reference, mode="full")
    lags = _np.arange(2 * N - 1) - (N - 1)
    weights = N - _np.abs(lags)
    if rate is not None:
        lags = lags / rate
    return Correlogram(lags=lags, coefs=C / weights)


def block_1d(
    x: _npt.NDArray,
    tol: float = 1e-6,
    retain_zeros: bool = True,
) -> _pd.DataFrame:
    """
    detects and returns the time windows (or 'blocks') where the value of `x` stays the same.

    the resulting `blocks` will be returned in the form of a pandas DataFrame, where:

    - `start` indicating the starting index of each block.
    - `stop` indicating the stopping (i.e. exclusive) index of each block.
    - `value` indicating the value of each block.

    if `retain_zeros` (True by default) are set to False, the blocks with zero-values will be
    discarded prior to being returned. This should be useful when you are only interested in
    non-zero blocks (e.g. only those containing True).

    The 'sameness' or 'zeroness' (particularly when using floting-point arrays) will be assessed
    based on the value of `tol`: the difference will be considered to be zero if its absolute
    value is below `tol`.
    """
    x = x.ravel()
    if _np.issubdtype(x.dtype, bool):
        v = x.astype(_np.int8)
    elif _np.issubdtype(x.dtype, _np.uint):
        v = x.astype(_np.int64)
    else:
        v = x
    dx = _np.diff(v)
    dx[-1] = 0  # exclude any changes that occur at the last minute
    jumps = _np.where(_np.abs(dx) > tol)[0]

    starts = _np.concatenate([_np.array((0,), dtype=jumps.dtype), (jumps + 1)])
    stops = _np.concatenate([starts[1:], (x.size,)])
    values = x[starts]

    if retain_zeros == False:
        valid = (_np.abs(values) > tol)
        starts = starts[valid]
        stops = stops[valid]
        values = values[valid]

    blocks = {
        'start': starts,
        'stop': stops,
        'value': values,
    }
    return _pd.DataFrame(data=blocks)
