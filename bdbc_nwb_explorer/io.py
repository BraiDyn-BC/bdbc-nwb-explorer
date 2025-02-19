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

from typing import Literal, Optional, Iterator
from pathlib import Path
from dataclasses import dataclass

import numpy as _np
import pandas as _pd
from pynwb import (
    NWBHDF5IO as _NWBHDFIO,
    NWBFile as _NWBFile,
)

from . import auxdata as _auxdata


PathLike = str | Path

ANALOG_DATA_TYPE = 'Analog'
ARBITRARY_UNIT = 'a.u.'


@dataclass
class NWBMetadata:
    downsampled: bool
    session_id: str
    session_description: str
    session_notes: str
    subject_id: str
    subject_DoB: str
    subject_age: str
    subject_sex: str
    rois: _pd.DataFrame
    trials: Optional[_pd.DataFrame] = None
    daq: _pd.DataFrame = None
    body_video: _pd.DataFrame = None
    face_video: _pd.DataFrame = None
    eye_video: _pd.DataFrame = None

    def __post_init__(self):
        self.daq = _auxdata.daq_metadata(downsampled=self.downsampled)
        self.body_video = _auxdata.body_video_metadata()
        self.face_video = _auxdata.face_video_metadata()
        self.eye_video = _auxdata.eye_video_metadata()


@dataclass(frozen=True)
class NWBDataEntry:
    name: str
    domain: str
    data: _pd.Series | _pd.DataFrame
    datatype: str
    unit: str
    description: str
    metadata: Optional[dict[str, object]] = None

    @property
    def timestamps(self):
        return self.data.index.values

    @property
    def values(self):
        if isinstance(self.data, _pd.DataFrame):
            return self.data.to_numpy()
        else:
            return self.data.values

    @property
    def shape(self):
        if isinstance(self.data, _pd.DataFrame):
            return self.data.shape
        else:
            return (self.data.values.size,)

    @property
    def size(self):
        if isinstance(self.data, _pd.DataFrame):
            return self.data.shape[0]
        else:
            return self.data.values.size

    @property
    def dt(self):
        return _np.diff(self.timestamps).mean()

    @property
    def rate(self):
        return 1 / self.dt

    def to_dataframe(
        self,
        with_domain_name: bool = False
    ) -> _pd.DataFrame:
        if with_domain_name == True:
            basename = f"{self.domain}_{self.name}"
        else:
            basename = self.name
        basedata = dict()
        if isinstance(self.data, _pd.DataFrame):
            for col in self.data.columns:
                basedata[f"{basename}_{col}"] = self.data[col].values
        else:
            basedata[basename] = self.data.values
        return _pd.DataFrame(data=basedata, index=self.data.index)


@dataclass(frozen=True)
class NWBDataStore:
    filepath: str
    metadata: NWBMetadata
    trials: Optional[_pd.DataFrame]
    daq: _pd.DataFrame
    imaging: _pd.DataFrame
    body_video: Optional[_pd.DataFrame]
    face_video: Optional[_pd.DataFrame]
    eye_video: Optional[_pd.DataFrame]
    pupil: Optional[_pd.DataFrame]

    def iterate_over_daq_entries(self) -> Iterator[NWBDataEntry]:
        for _, row in self.metadata.daq.iterrows():
            yield NWBDataEntry(
                name=row.Name,
                domain='daq',
                data=self.daq[row.Name],
                datatype=row.Type,
                unit=row.Unit,
                description=row.Description,
            )

    def iterate_over_imaging_entries(self) -> Iterator[NWBDataEntry]:
        for _, row in self.metadata.rois.iterrows():
            yield NWBDataEntry(
                name=row.roi_name,
                domain='imaging',
                data=self.imaging[row.roi_name],
                datatype=ANALOG_DATA_TYPE,
                unit=ARBITRARY_UNIT,
                description=row.roi_description,
                metadata=dict(mask=row.image_mask)
            )

    def iterate_over_body_video_entries(self) -> Iterator[NWBDataEntry]:
        if self.body_video is not None:
            for _, row in self.metadata.body_video.iterrows():
                data = _pd.DataFrame(data={
                    'x': self.body_video[row.Name, 'x'],
                    'y': self.body_video[row.Name, 'y'],
                })
                if not self.metadata.downsampled:
                    metadata = dict(likelihood=self.body_video[row.Name, 'likelihood'])
                else:
                    metadata = None

                yield NWBDataEntry(
                    name=row.Name,
                    domain='body_video',
                    data=data,
                    datatype=ANALOG_DATA_TYPE,
                    unit='px',
                    description=row.Description,
                    metadata=metadata,
                )

    def iterate_over_face_video_entries(self) -> Iterator[NWBDataEntry]:
        if self.face_video is not None:
            for _, row in self.metadata.face_video.iterrows():
                data = _pd.DataFrame(data={
                    'x': self.face_video[row.Name, 'x'],
                    'y': self.face_video[row.Name, 'y'],
                })
                if not self.metadata.downsampled:
                    metadata = dict(likelihood=self.face_video[row.Name, 'likelihood'])
                else:
                    metadata = None

                yield NWBDataEntry(
                    name=row.Name,
                    domain='face_video',
                    data=data,
                    datatype=ANALOG_DATA_TYPE,
                    unit='px',
                    description=row.Description,
                    metadata=metadata,
                )

    def iterate_over_eye_video_entries(
        self,
        include_pupil_edges: bool = False
    ) -> Iterator[NWBDataEntry]:
        if self.eye_video is not None:
            for _, row in self.metadata.eye_video.iterrows():
                if (not include_pupil_edges) and ('edge' in row.Name):
                    continue

                if row.Name == 'pupilcenter':
                    data = _pd.DataFrame(data={
                        'x': self.pupil['center_x'],
                        'y': self.pupil['center_y']
                    })
                    metadata = None
                elif row.Name == 'pupildia':
                    data = _pd.Series(data=self.pupil['diameter'])
                    metadata = None
                else:
                    data = _pd.DataFrame(data={
                        'x': self.eye_video[row.Name, 'x'],
                        'y': self.eye_video[row.Name, 'y'],
                    })
                    if not self.metadata.downsampled:
                        metadata = dict(likelihood=self.eye_video[row.Name, 'likelihood'])
                    else:
                        metadata = None

                yield NWBDataEntry(
                    name=row.Name,
                    domain='eye_video',
                    data=data,
                    datatype=ANALOG_DATA_TYPE,
                    unit='px',
                    description=row.Description,
                    metadata=metadata,
                )

    def iterate_over_entries(self, include_pupil_edges: bool = False) -> Iterator[NWBDataEntry]:
        yield from self.iterate_over_daq_entries()
        yield from self.iterate_over_imaging_entries()
        yield from self.iterate_over_body_video_entries()
        yield from self.iterate_over_face_video_entries()
        yield from self.iterate_over_eye_video_entries(include_pupil_edges=include_pupil_edges)


def read_metadata(
    nwbfile: _NWBFile,
    downsampled: bool = True,
) -> NWBMetadata:
    # TODO: better including trials/acquisition/ROIs/keypoints metadata
    dff_entry = nwbfile.get_processing_module('ophys').get_data_interface('DfOverF').get_roi_response_series('dFF')
    if nwbfile.trials is None:
        trials_metadata = None
    else:
        trials_metadata = {
            'Name': [],
            'Description': []
        }
        for name in nwbfile.trials.colnames:
            trials_metadata['Name'].append(name)
            trials_metadata['Description'].append(nwbfile.trials[name].description)
        trials_metadata = _pd.DataFrame(data=trials_metadata)
    metadata = {
        'downsampled': downsampled,
        'session_id': nwbfile.session_id,
        'session_description': nwbfile.session_description,
        'session_notes': nwbfile.notes,
        'subject_id': nwbfile.subject.subject_id,
        'subject_DoB': nwbfile.subject.date_of_birth.strftime('%Y-%m-%d'),
        'subject_age': nwbfile.subject.age,
        'subject_sex': nwbfile.subject.sex,
        'trials': trials_metadata,
        'rois': dff_entry.rois.table.to_dataframe(),
    }
    return NWBMetadata(**metadata)


def read_trials(
    nwbfile: _NWBFile,
    downsampled: bool = True
) -> Optional[_pd.DataFrame]:
    if downsampled:
        if 'trials' not in nwbfile.get_processing_module('downsampled').data_interfaces:
            return None
        entry = nwbfile.get_processing_module('downsampled').get_data_interface('trials')
    else:
        entry = nwbfile.trials
        if entry is None:
            return None
    trials = entry.to_dataframe()
    trials.index.name = None
    return trials


def read_acquisition(
    nwbfile: _NWBFile,
    downsampled: bool = True
) -> _pd.DataFrame:
    if downsampled:
        root = nwbfile.get_processing_module('downsampled')
        node_names = tuple(root.data_interfaces)

        def _get(node):
            return root.get_data_interface(node)
    else:
        node_names = tuple(nwbfile.acquisition.keys())

        def _get(node):
            return nwbfile.get_acquisition(node)

    data = dict()
    time = None
    for node in node_names:
        # exclude some of the nodes
        if node.startswith('widefield_') or node.endswith('_video') or node.endswith('_keypoints'):
            continue
        elif node in ('eye_position', 'pupil_tracking', 'trials'):
            continue
        entry = _get(node)
        values = _np.array(entry.data)
        if time is None:
            time = _pd.Index(data=_np.array(entry.timestamps), name='time')
        if node.startswith('state_'):
            values = values.astype(_np.int16)
        data[node] = values
    return _pd.DataFrame(data=data, index=time)


def read_video_tracking(
    nwbfile: _NWBFile,
    view: Literal['body', 'face', 'eye', 'pupil'] = 'body',
    downsampled: bool = True
) -> Optional[_pd.DataFrame]:
    if view in ('body', 'face', 'eye'):
        parent_name = f"{view}_video_keypoints"
        if downsampled:
            root = nwbfile.get_processing_module('downsampled')
        else:
            root = nwbfile.get_processing_module('behavior')
        if parent_name not in root.data_interfaces:
            return None
        entries = root.get_data_interface(parent_name)

        data = dict()
        time = None
        for kpt in entries.pose_estimation_series.keys():
            entry = entries.get_pose_estimation_series(kpt)
            if time is None:
                time = _pd.Index(data=_np.array(entry.timestamps), name='time')
            values = _np.array(entry.data)
            data[kpt, 'x'] = values[:, 0]
            data[kpt, 'y'] = values[:, 1]
            if not downsampled:
                data[kpt, 'likelihood'] = _np.array(entries.get_pose_estimation_series(kpt).confidence)
    elif view == 'pupil':
        if downsampled:
            root = nwbfile.get_processing_module('downsampled')
        else:
            root = nwbfile.get_processing_module('behavior')
        if 'pupil_tracking' not in root.data_interfaces:
            return None

        data = dict()
        time = _pd.Index(
            data=_np.array(root.get_data_interface('pupil_tracking').get_timeseries('diameter').timestamps),
            name='time',
        )
        data['diameter'] = _np.array(root.get_data_interface('pupil_tracking').get_timeseries('diameter').data)
        data['center_x'] = _np.array(root.get_data_interface('eye_position').get_spatial_series('center_x').data)
        data['center_y'] = _np.array(root.get_data_interface('eye_position').get_spatial_series('center_y').data)
    else:
        raise ValueError(f"expected 'body', 'face', 'eye' or 'pupil', but got '{view}'")
    return _pd.DataFrame(data=data, index=time)


def read_roi_dFF(nwbfile: _NWBFile) -> _pd.DataFrame:
    dff_entry = nwbfile.get_processing_module('ophys').get_data_interface('DfOverF').get_roi_response_series('dFF')
    time = _np.array(dff_entry.timestamps)
    dff = _np.array(dff_entry.data)
    names = tuple(str(name) for name in _np.array(dff_entry.rois.table.get('roi_name')))
    # TODO: it may be needed at some time in the future
    # descs = tuple(str(desc) for desc in _np.array(dff_entry.rois.table.get('roi_description')))
    # roidescs = dict((name, desc) for name, desc in zip(names, descs))
    return _pd.DataFrame(data=dff, columns=names, index=time)


def read_all(
    nwbfilepath: PathLike,
    downsampled: bool = True,
) -> NWBDataStore:
    with _NWBHDFIO(nwbfilepath, mode='r') as src:
        nwbfile = src.read()
        data = dict()
        data['metadata'] = read_metadata(nwbfile, downsampled=downsampled)
        data['daq'] = read_acquisition(nwbfile, downsampled=downsampled)
        data['imaging'] = read_roi_dFF(nwbfile)
        data['trials'] = read_trials(nwbfile, downsampled=downsampled)
        for view in ('body', 'face', 'eye'):
            data[f'{view}_video'] = read_video_tracking(nwbfile, view=view, downsampled=downsampled)
        data['pupil'] = read_video_tracking(nwbfile, view='pupil', downsampled=downsampled)
    return NWBDataStore(filepath=str(nwbfilepath), **data)
