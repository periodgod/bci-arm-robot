"""Quick LSL test for the first four EMG/AUX channels.

Expected LiveAmp layout for the current setup:
  channels 1-32   -> EEG
  channels 33-40  -> EMG/AUX
  channels 41-43  -> extra direction/accelerometer channels

This script selects channels 33-36 by default, which are Python indices 32-35.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Sequence

import numpy as np

from test_lsl_stream import choose_stream, pylsl


def _channel_indices(start: int, count: int) -> List[int]:
    return list(range(int(start), int(start) + int(count)))


def _print_window_stats(samples: np.ndarray, indices: Sequence[int], label: str) -> None:
    selected = samples[:, indices]
    means = np.mean(selected, axis=0)
    stds = np.std(selected, axis=0)
    rms = np.sqrt(np.mean(np.square(selected - means), axis=0))
    mins = np.min(selected, axis=0)
    maxs = np.max(selected, axis=0)
    peak_to_peak = maxs - mins
    print(label)
    for offset, idx in enumerate(indices):
        print(
            f"  EMG{offset + 1} stream_ch={idx + 1:02d} index={idx:02d} "
            f"mean={means[offset]: .3f} std={stds[offset]: .3f} "
            f"rms_ac={rms[offset]: .3f} p2p={peak_to_peak[offset]: .3f} "
            f"min={mins[offset]: .3f} max={maxs[offset]: .3f}"
        )


def run(args: argparse.Namespace) -> int:
    info = choose_stream(args.name, args.type, args.timeout)
    channel_count = int(info.channel_count())
    indices = _channel_indices(args.emg_start, args.emg_count)
    if not indices or min(indices) < 0 or max(indices) >= channel_count:
        raise SystemExit(
            f"Invalid EMG channel selection {indices}; stream has {channel_count} channels."
        )

    print(
        f"Testing EMG/AUX channels {indices[0] + 1}-{indices[-1] + 1} "
        f"(0-based indices {indices}) from stream {info.name()!r}."
    )
    print("Ask the subject to relax, then clench/change muscle tension during the test.")

    inlet = pylsl.StreamInlet(info, max_buflen=int(max(args.duration + 5, 10)))
    deadline = time.perf_counter() + float(args.duration)
    next_report = time.perf_counter() + float(args.report_interval)
    chunks: List[np.ndarray] = []
    timestamps: List[np.ndarray] = []
    recent_chunks: List[np.ndarray] = []

    while time.perf_counter() < deadline:
        chunk, ts = inlet.pull_chunk(timeout=0.1, max_samples=args.max_samples)
        if chunk and ts:
            arr = np.asarray(chunk, dtype=np.float64)
            chunks.append(arr)
            timestamps.append(np.asarray(ts, dtype=np.float64))
            recent_chunks.append(arr)

        now = time.perf_counter()
        if now >= next_report:
            if recent_chunks:
                recent = np.vstack(recent_chunks)
                _print_window_stats(recent, indices, f"Window {args.report_interval:.1f}s stats:")
                recent_chunks.clear()
            else:
                print("Waiting for EMG samples...")
            next_report = now + float(args.report_interval)

    inlet.close_stream()

    if not chunks:
        print("RESULT: connected, but received 0 samples.")
        return 2

    all_samples = np.vstack(chunks)
    all_timestamps = np.concatenate(timestamps)
    duration = float(all_timestamps[-1] - all_timestamps[0]) if all_timestamps.size > 1 else 0.0
    measured_fs = (len(all_timestamps) - 1) / duration if duration > 0 else float("nan")
    print("RESULT: EMG data received.")
    print(
        f"  total_samples={all_samples.shape[0]} selected_channels={len(indices)} "
        f"measured_fs={measured_fs:.2f} nominal_fs={float(info.nominal_srate()):g}"
    )
    _print_window_stats(all_samples, indices, "Full test stats:")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test first four LiveAmp EMG/AUX channels over LSL.")
    parser.add_argument("--name", default="", help="Optional exact LSL stream name. Empty means use stream type.")
    parser.add_argument("--type", default="EEG", help="LSL stream type to resolve.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait while resolving.")
    parser.add_argument("--duration", type=float, default=8.0, help="Seconds to collect EMG samples.")
    parser.add_argument("--report-interval", type=float, default=1.0, help="Seconds between live stat reports.")
    parser.add_argument("--max-samples", type=int, default=512, help="Maximum samples per pull_chunk call.")
    parser.add_argument("--emg-start", type=int, default=32, help="0-based index of EMG/AUX1 in the stream.")
    parser.add_argument("--emg-count", type=int, default=4, help="Number of EMG/AUX channels to test.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
