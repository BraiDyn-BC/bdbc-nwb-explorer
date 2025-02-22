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

from importlib import reload as _reload  # DEBUG

from . import (
    io,
    view,
    process,
    filters,
    plot,
)

_reload(io)  # DEBUG
_reload(view)  # DEBUG
_reload(process)  # DEBUG
_reload(filters)  # DEBUG
_reload(plot)  # DEBUG

NWBData = view.NWBData
NWBDataEntry = io.NWBDataEntry

read_nwb = view.read_nwb

triggers_to_indices = process.triggers_to_indices
peth_1d = process.peth_1d
block_1d = process.block_1d
correlogram = process.correlogram

convert_to_Allen_CCF = plot.convert_to_Allen_CCF
