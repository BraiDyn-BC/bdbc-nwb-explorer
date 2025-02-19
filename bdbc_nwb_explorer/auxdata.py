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

from pathlib import Path

import pandas as _pd

DATADIR = Path(__file__).parent / 'data'


def daq_metadata(downsampled: bool = True) -> _pd.DataFrame:
    basetab = _pd.read_csv(DATADIR / 'daq.csv')
    if downsampled:
        mask = (basetab.Available_ds.values == 1)
        data = {
            'Name': basetab.Name.values[mask],
            'Unit': basetab.Unit_ds.values[mask],
            'Type': basetab.Type_ds.values[mask],
            'Description': basetab.Description.values[mask],
        }
    else:
        mask = (basetab.Available_raw.values == 1)
        data = {
            'Name': basetab.Name.values[mask],
            'Unit': basetab.Unit_raw.values[mask],
            'Type': basetab.Type_raw.values[mask],
            'Description': basetab.Description.values[mask],
        }
    return _pd.DataFrame(data=data)


def body_video_metadata() -> _pd.DataFrame:
    return _pd.read_csv(DATADIR / 'body_video.csv')


def face_video_metadata() -> _pd.DataFrame:
    return _pd.read_csv(DATADIR / 'face_video.csv')


def eye_video_metadata() -> _pd.DataFrame:
    return _pd.read_csv(DATADIR / 'eye_video.csv')
