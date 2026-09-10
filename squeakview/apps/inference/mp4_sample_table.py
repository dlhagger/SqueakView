"""Bounded ISO-BMFF video sample-table validation.

This reader deliberately handles only the box structure needed to prove the
number of samples committed to an MP4 video track. It never loads a sample
table or media payload into memory.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


MAX_BOX_DEPTH = 8
MAX_TABLE_ENTRIES = 1_000_000_000


@dataclass(frozen=True, slots=True)
class Box:
    kind: bytes
    start: int
    payload_start: int
    end: int


@dataclass(frozen=True, slots=True)
class VideoSampleTable:
    sample_count: int
    timing_sample_count: int
    chunk_sample_count: int
    chunk_count: int


def _read_exact(handle: BinaryIO, offset: int, size: int) -> bytes:
    handle.seek(offset)
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("MP4 ended inside a required box field")
    return value


def _boxes(handle: BinaryIO, start: int, end: int, *, depth: int) -> Iterator[Box]:
    if depth > MAX_BOX_DEPTH:
        raise ValueError("MP4 box nesting exceeds the bounded validation limit")
    cursor = start
    while cursor < end:
        if end - cursor < 8:
            raise ValueError("MP4 contains a truncated box header")
        size32, kind = struct.unpack(">I4s", _read_exact(handle, cursor, 8))
        header_size = 8
        if size32 == 1:
            size = struct.unpack(">Q", _read_exact(handle, cursor + 8, 8))[0]
            header_size = 16
        elif size32 == 0:
            size = end - cursor
        else:
            size = size32
        if kind == b"uuid":
            header_size += 16
        if size < header_size or size > end - cursor:
            raise ValueError(f"invalid MP4 box size for {kind!r}")
        box_end = cursor + size
        yield Box(kind, cursor, cursor + header_size, box_end)
        cursor = box_end


def _children(handle: BinaryIO, parent: Box, *, depth: int) -> tuple[Box, ...]:
    return tuple(_boxes(handle, parent.payload_start, parent.end, depth=depth))


def _one(boxes: tuple[Box, ...], kind: bytes, description: str) -> Box:
    matches = [box for box in boxes if box.kind == kind]
    if len(matches) != 1:
        raise ValueError(f"MP4 must contain exactly one {description}")
    return matches[0]


def _handler_type(handle: BinaryIO, hdlr: Box) -> bytes:
    if hdlr.end - hdlr.payload_start < 12:
        raise ValueError("MP4 handler box is truncated")
    return _read_exact(handle, hdlr.payload_start + 8, 4)


def _stsz_count(handle: BinaryIO, stsz: Box) -> int:
    payload_size = stsz.end - stsz.payload_start
    if payload_size < 12:
        raise ValueError("MP4 stsz box is truncated")
    sample_size, count = struct.unpack(">II", _read_exact(handle, stsz.payload_start + 4, 8))
    if count > MAX_TABLE_ENTRIES:
        raise ValueError("MP4 sample count exceeds the bounded validation limit")
    required = 12 if sample_size else 12 + count * 4
    if payload_size != required:
        raise ValueError("MP4 stsz table length does not match its sample count")
    return count


def _stts_count(handle: BinaryIO, stts: Box) -> int:
    payload_size = stts.end - stts.payload_start
    if payload_size < 8:
        raise ValueError("MP4 stts box is truncated")
    entry_count = struct.unpack(">I", _read_exact(handle, stts.payload_start + 4, 4))[0]
    if entry_count > MAX_TABLE_ENTRIES or payload_size != 8 + entry_count * 8:
        raise ValueError("MP4 stts table length is invalid")
    total = 0
    offset = stts.payload_start + 8
    for _ in range(entry_count):
        count, delta = struct.unpack(">II", _read_exact(handle, offset, 8))
        if count == 0 or delta == 0:
            raise ValueError("MP4 stts contains a zero sample count or duration")
        total += count
        if total > MAX_TABLE_ENTRIES:
            raise ValueError("MP4 timing sample count exceeds the bounded limit")
        offset += 8
    return total


def _chunk_count(
    handle: BinaryIO,
    box: Box,
    media_ranges: tuple[tuple[int, int], ...],
) -> int:
    width = 8 if box.kind == b"co64" else 4
    payload_size = box.end - box.payload_start
    if payload_size < 8:
        raise ValueError("MP4 chunk-offset box is truncated")
    count = struct.unpack(">I", _read_exact(handle, box.payload_start + 4, 4))[0]
    if count > MAX_TABLE_ENTRIES or payload_size != 8 + count * width:
        raise ValueError("MP4 chunk-offset table length is invalid")
    offset = box.payload_start + 8
    format_code = ">Q" if width == 8 else ">I"
    for _ in range(count):
        chunk_offset = struct.unpack(
            format_code, _read_exact(handle, offset, width)
        )[0]
        if not any(start <= chunk_offset < end for start, end in media_ranges):
            raise ValueError("MP4 chunk offset points outside every media-data box")
        offset += width
    return count


def _stsc_sample_count(handle: BinaryIO, stsc: Box, chunk_count: int) -> int:
    payload_size = stsc.end - stsc.payload_start
    if payload_size < 8:
        raise ValueError("MP4 stsc box is truncated")
    entry_count = struct.unpack(">I", _read_exact(handle, stsc.payload_start + 4, 4))[0]
    if entry_count == 0 or entry_count > MAX_TABLE_ENTRIES or payload_size != 8 + entry_count * 12:
        raise ValueError("MP4 stsc table length is invalid")
    offset = stsc.payload_start + 8
    previous_first = 0
    previous_samples = 0
    total = 0
    for index in range(entry_count):
        first_chunk, samples_per_chunk, description_index = struct.unpack(
            ">III", _read_exact(handle, offset, 12)
        )
        if first_chunk == 0 or samples_per_chunk == 0 or description_index == 0:
            raise ValueError("MP4 stsc contains an invalid zero field")
        if index == 0 and first_chunk != 1:
            raise ValueError("MP4 stsc must begin at chunk one")
        if index and first_chunk <= previous_first:
            raise ValueError("MP4 stsc chunk runs are not strictly ordered")
        if index:
            total += (first_chunk - previous_first) * previous_samples
        previous_first = first_chunk
        previous_samples = samples_per_chunk
        if total > MAX_TABLE_ENTRIES:
            raise ValueError("MP4 chunk sample count exceeds the bounded limit")
        offset += 12
    if previous_first > chunk_count:
        raise ValueError("MP4 stsc references chunks outside the chunk table")
    total += (chunk_count - previous_first + 1) * previous_samples
    if total > MAX_TABLE_ENTRIES:
        raise ValueError("MP4 chunk sample count exceeds the bounded limit")
    return total


def read_video_sample_table(path: Path) -> VideoSampleTable:
    """Validate and return the sole video track's structural sample counts."""

    path = Path(path)
    size = path.stat().st_size
    if size < 8:
        raise ValueError("MP4 is empty or truncated")
    with path.open("rb") as handle:
        top = tuple(_boxes(handle, 0, size, depth=0))
        moov = _one(top, b"moov", "movie box")
        media_ranges = tuple(
            (box.payload_start, box.end) for box in top if box.kind == b"mdat"
        )
        if not media_ranges:
            raise ValueError("MP4 contains no media-data box")
        tracks = [box for box in _children(handle, moov, depth=1) if box.kind == b"trak"]
        video_tables: list[VideoSampleTable] = []
        for track in tracks:
            track_children = _children(handle, track, depth=2)
            mdia = _one(track_children, b"mdia", "track media box")
            media_children = _children(handle, mdia, depth=3)
            hdlr = _one(media_children, b"hdlr", "track handler box")
            if _handler_type(handle, hdlr) != b"vide":
                continue
            minf = _one(media_children, b"minf", "video media-information box")
            stbl = _one(_children(handle, minf, depth=4), b"stbl", "video sample-table box")
            sample_boxes = _children(handle, stbl, depth=5)
            sample_count = _stsz_count(handle, _one(sample_boxes, b"stsz", "video sample-size box"))
            timing_count = _stts_count(handle, _one(sample_boxes, b"stts", "video time-to-sample box"))
            offsets = [box for box in sample_boxes if box.kind in {b"stco", b"co64"}]
            if len(offsets) != 1:
                raise ValueError("MP4 must contain exactly one video chunk-offset box")
            chunks = _chunk_count(handle, offsets[0], media_ranges)
            chunk_samples = _stsc_sample_count(
                handle, _one(sample_boxes, b"stsc", "video sample-to-chunk box"), chunks
            )
            if sample_count != timing_count or sample_count != chunk_samples:
                raise ValueError(
                    "MP4 video sample tables disagree "
                    f"(stsz={sample_count}, stts={timing_count}, stsc={chunk_samples})"
                )
            video_tables.append(
                VideoSampleTable(sample_count, timing_count, chunk_samples, chunks)
            )
        if len(video_tables) != 1:
            raise ValueError("MP4 must contain exactly one video track")
        return video_tables[0]


__all__ = ["VideoSampleTable", "read_video_sample_table"]
