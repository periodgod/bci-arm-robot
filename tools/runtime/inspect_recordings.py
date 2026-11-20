import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "recordings"


def scalar(value):
    try:
        if hasattr(value, "shape") and value.shape == ():
            return value.item()
        if isinstance(value, np.ndarray) and value.size == 1:
            return value.reshape(-1)[0].item()
    except Exception:
        pass
    return value


def parse_json_field(value):
    value = scalar(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except Exception:
            return None
    return None


def label_counts_from_records(records):
    labels = []
    names = {}
    for rec in records or []:
        cid = rec.get("class_id", rec.get("label", rec.get("y")))
        if cid is None:
            continue
        try:
            cid_int = int(cid)
        except Exception:
            continue
        labels.append(cid_int)
        name = rec.get("class_name") or rec.get("cue_title") or rec.get("task_name") or rec.get("label_name")
        if name:
            names[cid_int] = str(name)
    return Counter(labels), names


def inspect_npz(path):
    item = {
        "file": str(path.relative_to(DATA_DIR)),
        "bytes": path.stat().st_size,
        "mb": round(path.stat().st_size / 1024 / 1024, 2),
    }
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.files)
        item["keys"] = keys
        for key in ("fs", "sampling_rate", "sample_rate"):
            if key in data:
                item["fs"] = scalar(data[key])
                break
        for key in ("eeg", "raw_stream", "aux", "y"):
            if key in data:
                item[f"{key}_shape"] = tuple(int(x) for x in data[key].shape)
        if "channel_names" in data:
            channels = data["channel_names"]
            item["channel_count"] = int(len(channels))
        if "aux_channel_names" in data:
            item["aux_channel_count"] = int(len(data["aux_channel_names"]))
        records = parse_json_field(data["trial_records_json"]) if "trial_records_json" in data else None
        if records is not None:
            item["trial_records"] = len(records)
            counts, names = label_counts_from_records(records)
            item["record_label_counts"] = dict(sorted(counts.items()))
            item["record_label_names"] = names
        if "y" in data:
            y = np.asarray(data["y"]).reshape(-1)
            counts = Counter()
            for val in y:
                try:
                    counts[int(val)] += 1
                except Exception:
                    counts[str(val)] += 1
            item["y_counts"] = dict(sorted(counts.items(), key=lambda kv: str(kv[0])))
            item["trial_count"] = int(y.size)
        elif "eeg" in data and len(data["eeg"].shape) >= 1:
            item["trial_count"] = int(data["eeg"].shape[0])
        elif "trial_records" in item:
            item["trial_count"] = item["trial_records"]
    return item


def inspect_archives():
    rows = []
    for path in sorted(DATA_DIR.rglob("*.zip")):
        row = {
            "file": str(path.relative_to(DATA_DIR)),
            "bytes": path.stat().st_size,
            "mb": round(path.stat().st_size / 1024 / 1024, 2),
        }
        try:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                row["members"] = len(names)
                row["npz_members"] = sum(1 for name in names if name.lower().endswith(".npz"))
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
    return rows


def main():
    result = {
        "npz": [inspect_npz(path) for path in sorted(DATA_DIR.rglob("*.npz"))],
        "archives": inspect_archives(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
