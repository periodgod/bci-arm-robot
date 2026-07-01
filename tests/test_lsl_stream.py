"""Quick LSL smoke test for BrainVision EEG streams.

This script only verifies that the local Python process can discover an LSL
stream and pull samples from it. It does not train or run the BCI decoder.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:
    import pylsl
except ImportError as exc:
    raise SystemExit(
        "pylsl is not installed in this Python environment. "
        "Install it with: python -m pip install pylsl"
    ) from exc


def _resolve_all(timeout: float) -> List[object]:
    """Return all visible streams, handling pylsl API spelling differences."""
    try:
        return list(pylsl.resolve_streams(wait_time=timeout))
    except TypeError:
        return list(pylsl.resolve_streams(timeout))


def _resolve_byprop(prop: str, value: str, timeout: float) -> List[object]:
    try:
        return list(pylsl.resolve_byprop(prop, value, minimum=1, timeout=timeout))
    except AttributeError:
        return list(pylsl.resolve_stream(prop, value, timeout=timeout))


def _first_text(nodes: Iterable[object]) -> Optional[str]:
    for node in nodes:
        try:
            text = node.child_value()
        except Exception:
            text = ""
        if text:
            return str(text)
    return None


def channel_labels(info: object, limit: int = 12) -> List[str]:
    labels: List[str] = []
    try:
        channel = info.desc().child("channels").child("channel")
        while not channel.empty() and len(labels) < limit:
            label = _first_text([channel.child("label"), channel.child("name")])
            if label:
                labels.append(label)
            channel = channel.next_sibling()
    except Exception:
        pass
    return labels


def print_streams(streams: Sequence[object]) -> None:
    if not streams:
        print("No LSL streams found.")
        return
    print(f"Found {len(streams)} LSL stream(s):")
    for idx, info in enumerate(streams, start=1):
        name = info.name()
        stream_type = info.type()
        channels = info.channel_count()
        fs = info.nominal_srate()
        source_id = info.source_id()
        print(
            f"  [{idx}] name={name!r} type={stream_type!r} "
            f"channels={channels} fs={fs:g} source_id={source_id!r}"
        )
        labels = channel_labels(info)
        if labels:
            more = " ..." if int(channels) > len(labels) else ""
            print(f"      labels={', '.join(labels)}{more}")


def choose_stream(name: str, stream_type: str, timeout: float) -> object:
    streams: List[object] = []
    if name:
        streams = _resolve_byprop("name", name, timeout)
    if not streams and stream_type:
        streams = _resolve_byprop("type", stream_type, timeout)
    if not streams:
        all_streams = _resolve_all(timeout)
        print_streams(all_streams)
        raise SystemExit(
            f"No matching LSL stream found for name={name!r}, type={stream_type!r}."
        )
    print_streams(streams)
    return streams[0]


def summarize_samples(samples: np.ndarray, timestamps: np.ndarray, info: object) -> None:
    if samples.size == 0:
        print("RESULT: connected, but received 0 samples.")
        return

    duration = float(timestamps[-1] - timestamps[0]) if timestamps.size > 1 else 0.0
    measured_fs = (len(timestamps) - 1) / duration if duration > 0 else math.nan
    nominal_fs = float(info.nominal_srate())
    print("RESULT: data received.")
    print(f"  samples={samples.shape[0]} channels={samples.shape[1]}")
    print(f"  nominal_fs={nominal_fs:g} measured_fs={measured_fs:.2f}")
    print(f"  first_timestamp={timestamps[0]:.6f} last_timestamp={timestamps[-1]:.6f}")

    preview_channels = min(samples.shape[1], 8)
    means = np.mean(samples[:, :preview_channels], axis=0)
    stds = np.std(samples[:, :preview_channels], axis=0)
    mins = np.min(samples[:, :preview_channels], axis=0)
    maxs = np.max(samples[:, :preview_channels], axis=0)
    print("  first channels summary:")
    for idx in range(preview_channels):
        print(
            f"    ch{idx + 1:02d}: mean={means[idx]: .3f} "
            f"std={stds[idx]: .3f} min={mins[idx]: .3f} max={maxs[idx]: .3f}"
        )


def run(args: argparse.Namespace) -> int:
    print(
        f"Resolving LSL stream: name={args.name!r} type={args.type!r} "
        f"timeout={args.timeout:.1f}s"
    )
    info = choose_stream(args.name, args.type, args.timeout)
    print(
        f"Connecting to name={info.name()!r}, type={info.type()!r}, "
        f"channels={info.channel_count()}, fs={info.nominal_srate():g}"
    )

    inlet = pylsl.StreamInlet(info, max_buflen=int(max(args.duration + 5, 10)))
    deadline = time.perf_counter() + args.duration
    chunks: List[np.ndarray] = []
    timestamps: List[np.ndarray] = []
    chunk_count = 0

    while time.perf_counter() < deadline:
        chunk, ts = inlet.pull_chunk(timeout=0.25, max_samples=args.max_samples)
        if chunk and ts:
            arr = np.asarray(chunk, dtype=np.float64)
            chunks.append(arr)
            timestamps.append(np.asarray(ts, dtype=np.float64))
            chunk_count += 1
            print(
                f"  chunk {chunk_count}: samples={arr.shape[0]} "
                f"channels={arr.shape[1]} latest_ts={timestamps[-1][-1]:.6f}"
            )
        else:
            print("  waiting for samples...")

    inlet.close_stream()

    if chunks:
        all_samples = np.vstack(chunks)
        all_timestamps = np.concatenate(timestamps)
    else:
        all_samples = np.empty((0, int(info.channel_count())), dtype=np.float64)
        all_timestamps = np.empty((0,), dtype=np.float64)

    summarize_samples(all_samples, all_timestamps, info)
    return 0 if all_samples.shape[0] > 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test whether BrainVision LSL EEG data is reachable.")
    parser.add_argument("--name", default="BrainVision", help="Preferred LSL stream name.")
    parser.add_argument("--type", default="EEG", help="Fallback LSL stream type.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait while resolving.")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to pull samples.")
    parser.add_argument("--max-samples", type=int, default=512, help="Maximum samples per pull_chunk call.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
