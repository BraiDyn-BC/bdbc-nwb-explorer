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

from typing import ClassVar, Optional, Iterable, Iterator, Callable
from typing_extensions import Self
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from collections import namedtuple
import re as _re
import fnmatch as _fnmatch
import warnings as _warnings

import numpy as _np
import pandas as _pd

from . import io as _io

ID_PATTERN = _re.compile(r'([a-zA-Z0-9-#]+)_([0-9-]+)_(task|resting-state|sensory-stim)-day([0-9-]+)')
DATE_FORMAT = '%Y-%m-%d'
DOMAINS: tuple[str] = ('daq', 'imaging', 'body_video', 'eye_video', 'face_video')

StringEntrySpec = Optional[str | tuple[str]]


class DataNotFoundWarning(UserWarning):
    pass


@dataclass(frozen=True)
class SessionSpec:
    type: str
    date: datetime
    day: int


@dataclass(frozen=True)
class NWBTrials:
    metadata: Optional[_pd.DataFrame]
    data: Optional[_pd.DataFrame]

    def __repr__(self):
        return f"{self.__class__.__name__}({self._describe()})"

    def has_data(self) -> bool:
        return (self.data is not None)

    def _describe(self) -> str:
        if self.has_data():
            return f"<{self.count()} trials>"
        else:
            return "<no trials>"

    def count(self) -> int:
        if self.data is None:
            return 0
        else:
            return self.data.shape[0]


class EntryPattern(namedtuple('EntryPattern', ('domain', 'name'))):
    SEP = '/'

    @property
    def may_be_multiple(self) -> bool:
        multiple_domains = (self.domain is None) or ('*' in self.domain)
        multiple_names = (self.name is None) or ('*' in self.name)
        return multiple_domains or multiple_names

    def __str__(self):
        if self.domain is None:
            if self.name is None:
                return '*'
            else:
                return self.name
        else:
            if self.name is None:
                return f"{self.domain}/*"
            else:
                return f"{self.domain}/{self.name}"

    def match_domain(self, domain: str) -> bool:
        return (self.domain is None) or _fnmatch.fnmatchcase(domain, self.domain)

    def match_name(self, name: str) -> bool:
        return (self.name is None) or _fnmatch.fnmatchcase(name, self.name)

    def match(self, entry: _io.NWBDataEntry) -> bool:
        return self.match_domain(entry.domain) and self.match_name(entry.name)

    @classmethod
    def parse(cls, spec: StringEntrySpec) -> Self:
        if spec is None:
            return cls(None, '*')
        elif isinstance(spec, tuple):
            if any(not ((item is None) or isinstance(item, str)) for item in spec):
                raise ValueError(f'the entry specification needs to be a tuple of str objects, got {repr(spec)}')
            if len(spec) == 1:
                return cls(domain=None, name=spec[0])
            elif len(spec) == 2:
                return cls(domain=spec[0], name=spec[1])
            else:
                raise ValueError(f"too long the entry specification tuple (size {len(spec)}: perhaps use double-brackets instead of single ones?)")
        else:
            if len(spec) == 0:
                raise ValueError('unexpected empty string')
            items = spec.split(cls.SEP)
            if len(items) > 3:
                raise ValueError(f"too long the entry specification path ({repr(spec)}: size {len(items)})")
            elif any(len(item) == 0 for item in items):
                raise ValueError("empty path component(s) found (perhaps use '*' to represent a wildcard?)")
            if len(items) == 1:
                return cls(domain=None, name=items[0])
            else:
                return cls(domain=items[0], name=items[1])


class EntryLookup:
    specs: list[EntryPattern] = None
    matched: list[bool] = None

    def __init__(
        self,
        specs: tuple[EntryPattern],
    ):
        self.specs = list(specs)
        self.matched = [False] * len(self.specs)

    def has_more(self) -> bool:
        return len(self.specs) > 0

    def match_domain(self, domain: str) -> bool:
        return any(spec.match_domain(domain) for spec in self.specs)

    def consume_matched(self, entry: _io.NWBDataEntry) -> bool:
        matched = False
        for i, spec in enumerate(self.specs):
            if spec.match(entry) == True:
                matched = True
                break
        if matched == True:
            if spec.may_be_multiple == False:
                self.specs.pop(i)
                self.matched.pop(i)
            else:
                self.matched[i] = True
        return matched

    def finalize(self):
        remainings = tuple(str(spec) for matched, spec in zip(self.matched, self.specs) if matched == False)
        if len(remainings) > 0:
            raise ValueError(f"pattern(s) did not match: {repr(remainings)}")


@dataclass(frozen=True)
class NWBEntrySet:
    """
    domain/name-based lookup tables for entries.

    A set of square brackets may be used to
    retrieve a selected set of items, in terms of
    another `NWBEntrySet` object.

    The conventions largely follow those of Pandas:

    entries['name'] -- single entry (name-based lookup)
    entries['domain', 'name'] -- single entry (domain/name-based lookup)
    entries['domain/name'] -- single entry (domain/name-based lookup)
    entries[['name1', 'name2', ('domain', 'name3')]] -- multiple entries

    `entries['domain', None]` or `entries['domain/*']` may be used to retrieve
    all the entries in the specified domain.
    """

    names_lookup_: dict[str, _io.NWBDataEntry]
    domains_lookup_: dict[str, dict[str, _io.NWBDataEntry]]
    overlaps_lookup_: tuple[str]
    sizes_lookup_: dict[str, int]
    rates_lookup_: dict[str, float]

    REPR_FORMAT: ClassVar[str] = """{this}(
    {spec_daq},
    {spec_imaging},
    {spec_body_video},
    {spec_face_video},
    {spec_eye_video},
)"""

    @classmethod
    def setup(cls, entries: Iterator[_io.NWBDataEntry]) -> Self:
        names = dict()
        domains = dict()
        overlap = []
        rates = dict()
        sizes = dict()
        for domain in DOMAINS:
            domains[domain] = dict()
            rates[domain] = _np.nan
            sizes[domain] = _np.nan
        for entry in entries:
            name = entry.name
            domain = entry.domain
            #
            # NOTE:
            #
            # the corresponding `entry` will be added
            # only when `name` does _not_ appear in `names` yet.
            # i.e. in case of different entries having the same name,
            # only the first one will be registered in `names`.
            #
            # use domain-based lookup to ensure which entries to be loaded.
            #
            if name not in names.keys():
                names[name] = entry
            else:
                overlap.append(name)
            domains[domain][name] = entry
            if _np.isnan(rates[domain]):
                rates[domain] = float(entry.rate)
                sizes[domain] = int(entry.size)
        return cls(
            names_lookup_=names,
            domains_lookup_=domains,
            overlaps_lookup_=tuple(overlap),
            rates_lookup_=rates,
            sizes_lookup_=sizes
        )

    def __repr__(self):
        spec = dict((f"spec_{domain}", self._repr_domain(domain)) for domain in DOMAINS)
        return self.REPR_FORMAT.format(this=self.__class__.__name__, **spec)

    def _repr_domain(self, domain: str) -> str:
        num_entries = len(self.domains_lookup_[domain])
        if num_entries == 0:
            return f"{domain}=<no entries found>"
        else:
            return f"{domain}=<{num_entries} x {self.sizes_lookup_[domain]} timepoints ({round(self.rates_lookup_[domain])} Hz)>"

    def __iter__(self) -> Iterator[_io.NWBDataEntry]:
        """works as a shorthand for `iterate_over_entries()`"""
        return self.iterate_over_entries()

    def __getitem__(self, key):
        if isinstance(key, tuple):
            specs = (EntryPattern.parse(key),)
        else:
            if isinstance(key, list):
                specs = tuple(EntryPattern.parse(elem) for elem in key)
            else:
                specs = (EntryPattern.parse(str(key)),)
        return self.__class__.setup(self.lookup(specs))

    @property
    def names(self) -> tuple[str]:
        return tuple(self.names_lookup_.keys())

    @property
    def domains(self) -> tuple[str]:
        return tuple(domain for domain in self.domains_lookup_.keys() if len(self.domains_lookup_[domain]) > 0)

    @property
    def metadata(self) -> _pd.DataFrame:
        data = []
        for entry in self.iterate_over_entries():
            data.append({
                'Name': entry.name,
                'Domain': entry.domain,
                'Type': entry.datatype,
                'Unit': entry.unit,
                'Description': entry.description,
            })
        return _pd.DataFrame(data=data)

    @property
    def data(self) -> _pd.DataFrame:
        basedata = []
        for entry in self.iterate_over_entries():
            basedata.append(entry.to_dataframe(with_domain_name=(entry.name in self.overlaps_lookup_)))
        return _pd.concat(basedata, axis=1)

    def count(self, domain: Optional[str | Iterable[str]] = None) -> int:
        """returns the number of entries in a domain, or a set of domains.
        if `domain` is not specified, it returns the number of all the available entries."""
        if domain is None:
            domain = DOMAINS
        elif isinstance(domain, str):
            domain = (domain,)
        else:
            domain = tuple(domain)
        return sum(len(domain) for domain in self.domains_lookup_.values())

    def iterate_over_entries(self) -> Iterator[_io.NWBDataEntry]:
        for domain in self.domains_lookup_.values():
            yield from domain.values()

    def lookup(
        self,
        specs: tuple[EntryPattern],
        allow_multiple: bool = True
    ) -> Iterator[_io.NWBDataEntry]:
        server = EntryLookup(specs)
        for domain, entries in self.domains_lookup_.items():
            if server.match_domain(domain) == False:
                continue
            for entry in entries.values():
                matched = server.consume_matched(entry)
                if matched == True:
                    yield entry
                    if not server.has_more():
                        return
        server.finalize()

    def apply(self, fn: Callable[[_io.NWBDataEntry], _io.NWBDataEntry]) -> Self:
        return self.__class__.setup(fn(entry) for entry in self.iterate_over_entries())

    def _asdict(self) -> dict[str, dict[str, _io.NWBDataEntry]]:
        """returns the deep copy of domains_lookup_"""
        lookup = dict()
        for domain, entries in self.domains_lookup_.items():
            lookup[domain] = dict()
            for name, entry in entries.items():
                lookup[domain][name] = entry
        return lookup

    def __or__(self, other):
        """merges two NWBEntrySet objects, with `self` being prioritized over `other`"""
        # deep-copy domains_lookup_
        lookup = self._asdict()
        for entry in other.iterate_over_entries():
            if entry.name in lookup[entry.domain].keys():
                continue
            lookup[entry.domain][entry.name] = entry
        entries = []
        for dentries in lookup.values():
            for entry in dentries.values():
                entries.append(entry)
        return self.__class__.setup(entries)


@dataclass
class NWBData:
    data: _io.NWBDataStore
    session_spec: SessionSpec = None
    trials_: NWBTrials = None
    entries: NWBEntrySet = None
    include_pupil_edges: bool = False
    REPR_FORMAT: ClassVar[str] = """{this}(
    subject={subject},
    session_date={datestr},
    session_type={session_type},
    session_description={session_description},
    session_notes={session_notes},
    {spec_trials},
    {spec_daq},
    {spec_imaging},
    {spec_body_video},
    {spec_face_video},
    {spec_eye_video},
)"""

    def __post_init__(self):
        if self.session_spec is None:
            self.session_spec = parse_session_id(self.data.filepath)
        if self.trials_ is None:
            self.trials_ = NWBTrials(metadata=self.data.metadata.trials, data=self.data.trials)
        if self.entries is None:
            self.entries = NWBEntrySet.setup(self.data.iterate_over_entries(
                include_pupil_edges=self.include_pupil_edges
            ))

    def __repr__(self):
        return self.REPR_FORMAT.format(
            this=self.__class__.__name__,
            subject=repr(self.subject),
            datestr=repr(self.datestr),
            session_type=repr(self.session_type),
            session_description=repr(self.session_description),
            session_notes=repr(self.session_notes),
            **self._repr_specs()
        )

    def _repr_specs(self) -> dict[str, str]:
        specs = dict()
        specs['spec_trials'] = f"trials={self.trials_._describe()}"
        for domain in DOMAINS:
            specs[f'spec_{domain}'] = self.entries._repr_domain(domain)
        return specs

    @property
    def metadata(self) -> _io.NWBMetadata:
        return self.data.metadata

    @property
    def subject(self) -> str:
        return self.data.metadata.subject_id

    @property
    def date(self) -> datetime:
        return self.session_spec.date

    @property
    def datestr(self) -> str:
        return self.date.strftime(DATE_FORMAT)

    @property
    def session_type(self) -> str:
        return self.session_spec.type

    @property
    def day_index(self) -> int:
        return self.session_spec.day

    @property
    def session_description(self) -> str:
        return self.data.metadata.session_description

    @property
    def session_notes(self) -> str:
        return self.data.metadata.session_notes

    @property
    def trials(self) -> NWBTrials:
        if self.data.trials is None:
            _warnings.warn(
                f"{self.subject}, {self.datestr} ({self.session_type}): no trials are registered",
                category=DataNotFoundWarning,
                stacklevel=2,
            )
        return self.trials_

    @property
    def daq(self) -> NWBEntrySet:
        return self.entries['daq/*']

    @property
    def imaging(self) -> NWBEntrySet:
        return self.entries['imaging/*']

    @property
    def body_video(self) -> NWBEntrySet:
        if self.data.body_video is None:
            _warnings.warn(
                f"{self.subject}, {self.datestr} ({self.session_type}): no body video was found",
                category=DataNotFoundWarning,
                stacklevel=2,
            )
        return self.entries['body_video/*']

    @property
    def face_video(self) -> NWBEntrySet:
        if self.data.face_video is None:
            _warnings.warn(
                f"{self.subject}, {self.datestr} ({self.session_type}): no face video was found",
                category=DataNotFoundWarning,
                stacklevel=2,
            )
        return self.entries['face_video/*']

    @property
    def eye_video(self) -> NWBEntrySet:
        if self.data.eye_video is None:
            _warnings.warn(
                f"{self.subject}, {self.datestr} ({self.session_type}): no eye video was found",
                category=DataNotFoundWarning,
                stacklevel=2,
            )
        return self.entries['eye_video/*']


def parse_session_id(filepath: str) -> SessionSpec:
    name = Path(filepath).stem
    matches = ID_PATTERN.match(name)
    if not matches:
        raise ValueError(f'does not match the session ID pattern: {name}')
    session_date = datetime.strptime(matches.group(2), DATE_FORMAT)
    session_type = matches.group(3)
    day_index = int(matches.group(4))
    return SessionSpec(session_type, session_date, day_index)


def read_nwb(
    filepath: _io.PathLike,
    downsampled: bool = True,
    include_pupil_edges: bool = False,
) -> NWBData:
    return NWBData(
        data=_io.read_all(filepath, downsampled=downsampled),
        include_pupil_edges=include_pupil_edges
    )
