"""Check the extracted BTL package and recompute metrics from aggregate data."""

from __future__ import annotations

import csv
import hashlib
import math
import re
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def verify_manifest() -> int:
    manifest = ROOT / "SHA256SUMS.txt"
    paths = set()
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_./-]+)", line)
        require(match is not None, "Dòng SHA256SUMS.txt không hợp lệ")
        expected, name = match.groups()
        path = (ROOT / name).resolve()
        require(
            path.is_relative_to(ROOT) and path != manifest and path not in paths,
            f"Đường dẫn manifest không hợp lệ hoặc bị trùng: {name}",
        )
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Sai SHA-256: {name}")
        paths.add(path)
    require({manifest, *paths} == {p.resolve() for p in ROOT.rglob("*") if p.is_file()},
            "Tệp giải nén không khớp danh mục gói")
    print(f"SHA-256: {len(paths)}/{len(paths)} tệp đạt.")
    return len(paths)


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "ket_qua" / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_close(actual: float, recorded: str, key: str) -> None:
    require(math.isclose(actual, float(recorded), rel_tol=0, abs_tol=1e-12),
            f"Số liệu không khớp: {key}")


def confusion_metrics(row: dict[str, str]) -> tuple[int, int, int, int]:
    counts = tuple(int(row[key]) for key in ("tn", "fp", "fn", "tp"))
    tn, fp, fn, tp = counts
    n = int(row["n"])
    key = row["run_key"] + "/" + (row["domain"] or "all")
    require(n > 0 and min(counts) >= 0 and sum(counts) == n, f"Sai số mẫu: {key}")

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    scores = {
        "accuracy": (tn + tp) / n,
        "macro_precision": (ratio(tn, tn + fn) + ratio(tp, tp + fp)) / 2,
        "macro_recall": (ratio(tn, tn + fp) + ratio(tp, tp + fn)) / 2,
        "f1_label_0": ratio(2 * tn, 2 * tn + fp + fn),
        "f1_label_1": ratio(2 * tp, 2 * tp + fp + fn),
    }
    scores["macro_f1"] = (scores["f1_label_0"] + scores["f1_label_1"]) / 2
    for metric, value in scores.items():
        validate_close(value, row[metric], f"{key}/{metric}")
    return tn, fp, fn, tp


def verify_results() -> None:
    rows = read_csv("comparison.csv")
    runs = {r["run_key"]: r for r in rows if r["row_type"] == "run"}
    domains = [r for r in rows if r["row_type"] == "domain"]
    summaries = {r["configuration"]: r for r in rows if r["row_type"] == "family_summary"}
    families = {"S0": (42,), "P0": (42, 43, 44), "P1": (42, 43, 44)}
    require(len(rows) == 38 and set(summaries) == set(families), "Sai danh mục bảng so sánh")
    require(set(runs) == {f"{family}_seed{seed}" for family, seeds in families.items() for seed in seeds},
            "Thiếu hoặc thừa lượt thực nghiệm")
    require(len(domains) == 28 and {r["run_key"] for r in domains} == set(runs),
            "Sai danh mục kết quả theo miền")
    for key, run in runs.items():
        by_domain = [r for r in domains if r["run_key"] == key]
        require({r["domain"] for r in by_domain} == {"app", "cosmetic", "fashion", "mobile"},
                f"Thiếu miền: {key}")
        total = tuple(map(sum, zip(*(confusion_metrics(r) for r in by_domain))))
        require(total == confusion_metrics(run) and int(run["n"]) == 546, f"Sai tổng theo miền: {key}")

    means = {}
    for family, seeds in families.items():
        scores = [float(runs[f"{family}_seed{seed}"]["macro_f1"]) for seed in seeds]
        means[family] = statistics.mean(scores)
        validate_close(means[family], summaries[family]["macro_f1"], family)
        if len(scores) > 1:
            sd = statistics.stdev(scores)
            validate_close(sd, summaries[family]["macro_f1_sd"], f"{family}/seed SD")
            print(f"{family}: Macro-F1 {means[family]:.5f} (SD giữa {len(scores)} seed: {sd:.5f})")
        else:
            print(f"{family}: Macro-F1 {means[family]:.5f} ({len(scores)} lượt)")

    bootstrap = read_csv("paired_bootstrap_differences.csv")
    require(len(bootstrap) == 10_000 and all(int(r["replicate"]) == i
            for i, r in enumerate(bootstrap, 1)), "Thiếu bản lặp bootstrap")
    differences = [float(r["paired_macro_f1_difference_p1_minus_p0"]) for r in bootstrap]
    require(all(-1 <= d <= 1 for d in differences), "Chênh lệch bootstrap ngoài miền hợp lệ")
    percentiles = statistics.quantiles(differences, n=40, method="inclusive")
    delta, low, high = means["P1"] - means["P0"], percentiles[0], percentiles[-1]
    require(tuple(round(v, 5) for v in (delta, low, high)) == (-0.00607, -0.01521, 0.00307),
            "Chênh lệch hoặc khoảng tin cậy khác báo cáo v42")
    print(f"P1 - P0: {delta:.5f}; KTC percentile 95% [{low:.5f}; {high:.5f}].")

    print("\nS0 trên test lịch sử: báo nhầm / số nhãn 0 | bỏ sót / số nhãn 1")
    for row in [runs["S0_seed42"], *(r for r in domains if r["run_key"] == "S0_seed42")]:
        tn, fp, fn, tp = (int(row[key]) for key in ("tn", "fp", "fn", "tp"))
        print(f"{row['domain'] or 'all':9} {fp:2}/{tn+fp:3} ({fp/(tn+fp):.1%}) | "
              f"{fn:2}/{tp+fn:3} ({fn/(tp+fn):.1%})")
    print("Đã kiểm tra 7 lượt, 28 dòng theo miền và 10.000 chênh lệch bootstrap đã lưu.")


if __name__ == "__main__":
    try:
        verify_manifest()
        verify_results()
    except (OSError, ValueError, KeyError) as exc:
        print(f"Kiểm tra thất bại: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
