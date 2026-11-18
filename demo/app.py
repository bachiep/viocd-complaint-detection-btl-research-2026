"""Local, non-persisting demonstration for the validation-selected S0 model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


LOCKED_PROTOCOL_SHA256 = "65b26012eb1117f0674f8eb3502d0e289ba006b8c287b99f1a72cec115e76ae3"
LOCKED_PROTOCOL_TEXT_SHA256 = "5dc1c3fd3831a8738568b3b1064ffdb937e77df5ca90f6d93be41b1d69a81c0c"
LOCKED_SELECTION_SHA256 = "36354e5a389651eae7223d66e4566246e0b78659330f12d63929f64f75e581d1"
LOCKED_RUN_METADATA_SHA256 = "069ba05b557465acb2ce52137035836b29c875dee8b7421114d790e541a7cc00"
LOCKED_PIPELINE_SHA256 = "f03708959c4694cafdf05232c4d80e86e9637418f0c7901b9c1a111428a1bcca"
LOCKED_SOURCE_PROVENANCE_SHA256 = {
    "baseline.py": "6a0bba3106ba1eaf385d50c3c08d3e59f57b0d10c3404600f282f515852ed240",
    "data.py": "05024028f0212141d151abcd5b6e6653e2be9f1d1b21020aaa4b0dd4c419eaf1",
    "phobert.py": "5855789b39e0827ad9aa53532d4a14f856b25a13f723ce76250c374aa2d52101",
    "preprocessing.py": "5a4921ed1466d899bf8decf195e90af391ea8e2260932a5867129c5911c14b01",
    "selection.py": "fafdb8c08879a36eea11a5356764c3c0a0c0303bda5b78c54fbcd0051b1f8b00",
}
LOCKED_SOURCE_TEXT_SHA256 = {
    "baseline.py": "6a0bba3106ba1eaf385d50c3c08d3e59f57b0d10c3404600f282f515852ed240",
    "data.py": "849a35ed971a9b5c16fc3abb23e847256544035ec159de5fd9d329b8ba914880",
    "phobert.py": "f7811831f8633688359faec2015a24c10c00627f483f62cdc197b96130498510",
    "preprocessing.py": "5a4921ed1466d899bf8decf195e90af391ea8e2260932a5867129c5911c14b01",
    "selection.py": "fafdb8c08879a36eea11a5356764c3c0a0c0303bda5b78c54fbcd0051b1f8b00",
}

EXPECTED_PACKAGES = {
    "numpy": "2.1.3",
    "pandas": "2.3.3",
    "scikit-learn": "1.8.0",
    "underthesea": "9.5.0",
}
MAX_REVIEW_CHARS = 5_000
MAX_REQUEST_BYTES = 16_384


class DemoIntegrityError(RuntimeError):
    """Raised when the runnable demo differs from its reviewed model evidence."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_file(path: Path) -> str:
    """Hash UTF-8 source/protocol bytes after canonicalizing platform newlines."""
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoIntegrityError(f"Không đọc được hồ sơ đã khóa: {path.name}") from exc
    if not isinstance(value, dict):
        raise DemoIntegrityError(f"Hồ sơ không đúng cấu trúc: {path.name}")
    return value


def require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise DemoIntegrityError(f"Hash {label} không khớp hồ sơ đã khóa")


def require_text_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_text_file(path)
    if observed != expected:
        raise DemoIntegrityError(f"Hash {label} không khớp hồ sơ đã khóa")


def load_model(project_root: Path) -> tuple[Any, dict[str, Any], Any]:
    project_root = project_root.resolve(strict=True)
    sys.path.insert(0, str(project_root / "src"))

    for package, expected in EXPECTED_PACKAGES.items():
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise DemoIntegrityError(f"Thiếu package runtime: {package}") from exc
        if observed != expected:
            raise DemoIntegrityError(
                f"{package}={observed}; yêu cầu phiên bản đã khóa {expected}"
            )

    from joblib import load
    from nlp_btl.preprocessing import UndertheseaSegmenter

    protocol_path = project_root / "preparation" / "protocol.json"
    selection_path = project_root / "results/m2_dev_v15_20261118_r3/selection.json"
    run_dir = project_root / "results/m2_dev_v15_20261118_r3/S0_seed42"
    metadata_path = run_dir / "run_metadata.json"
    require_text_hash(protocol_path, LOCKED_PROTOCOL_TEXT_SHA256, "protocol")
    require_hash(selection_path, LOCKED_SELECTION_SHA256, "selection")
    require_hash(metadata_path, LOCKED_RUN_METADATA_SHA256, "run metadata")

    protocol = read_json(protocol_path)
    selection = read_json(selection_path)
    metadata = read_json(metadata_path)
    if protocol.get("protocol_id") != "viocd-segmentation-core-v1":
        raise DemoIntegrityError("Protocol ID khác phiên bản đã nghiệm thu")
    if selection.get("status") != "SELECTION_SAVED":
        raise DemoIntegrityError("Selection chưa ở trạng thái hoàn tất")
    if selection.get("selected_demo_run") != "S0_seed42":
        raise DemoIntegrityError("Run demo không khớp lựa chọn validation")
    if metadata.get("run_id") != "S0_seed42_6491091012fc":
        raise DemoIntegrityError("Run metadata không khớp run đã chọn")
    if metadata.get("status") != "FIT_COMPLETE" or metadata.get("fit_data_split") != "train":
        raise DemoIntegrityError("Run không chứng minh fit hoàn tất trên train")
    if metadata.get("raw_review_text_saved") is not False:
        raise DemoIntegrityError("Hồ sơ model không xác nhận chính sách lưu văn bản")
    if metadata.get("protocol_sha256") != LOCKED_PROTOCOL_SHA256:
        raise DemoIntegrityError("Run metadata trỏ đến protocol khác")
    if metadata.get("pipeline_sha256") != LOCKED_PIPELINE_SHA256:
        raise DemoIntegrityError("Pipeline hash trong metadata khác")
    if selection.get("protocol_sha256") != LOCKED_PROTOCOL_SHA256:
        raise DemoIntegrityError("Selection trỏ đến protocol khác")
    if selection.get("source_sha256") != LOCKED_SOURCE_PROVENANCE_SHA256:
        raise DemoIntegrityError("Selection chứa hash nguồn khác demo đã khóa")

    for filename, expected in LOCKED_SOURCE_TEXT_SHA256.items():
        require_text_hash(
            project_root / "src/nlp_btl" / filename,
            expected,
            filename,
        )

    mapping = protocol["dataset"]["label_mapping"]
    if mapping != {"0": "không có lời phàn nàn", "1": "có lời phàn nàn"}:
        raise DemoIntegrityError("Ánh xạ nhãn trong protocol không đúng")
    pipeline_path = run_dir / metadata["pipeline_file"]
    require_hash(pipeline_path, LOCKED_PIPELINE_SHA256, "pipeline")

    # The artifact is loaded only after all expected hashes and runtime pins pass.
    estimator = load(pipeline_path)
    segmenter = UndertheseaSegmenter()
    public_metadata = {
        "model": "S0 · TF-IDF + LinearSVC",
        "selected_run": metadata["run_id"],
        "validation_macro_f1": selection["selected_demo_validation_macro_f1"],
        "protocol_id": protocol["protocol_id"],
        "pipeline_sha256": LOCKED_PIPELINE_SHA256,
    }
    return estimator, public_metadata, segmenter


def create_handler(estimator: Any, public_metadata: dict[str, Any], segmenter: Any):
    from nlp_btl.data import normalize_review
    from nlp_btl.preprocessing import preprocess_review

    class DemoHandler(BaseHTTPRequestHandler):
        server_version = "ViOCDLocalDemo/1.0"
        sys_version = ""

        def log_message(self, _format: str, *_args: Any) -> None:
            # Requests contain private user text; avoid request/body logging.
            return

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/api/health":
                self.send_json(200, {"status": "ready", **public_metadata})
                return
            if self.path != "/":
                self.send_error(404)
                return
            page = (Path(__file__).parent / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:
            if self.path != "/api/predict":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_json(400, {"error": "Kích thước yêu cầu không hợp lệ."})
                return
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self.send_json(413, {"error": "Yêu cầu quá lớn hoặc rỗng."})
                return
            if "application/json" not in self.headers.get("Content-Type", ""):
                self.send_json(415, {"error": "Chỉ nhận dữ liệu JSON."})
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_json(400, {"error": "JSON không hợp lệ."})
                return
            raw_text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(raw_text, str):
                self.send_json(400, {"error": "Nội dung cần là chuỗi văn bản."})
                return
            if len(raw_text) > MAX_REVIEW_CHARS:
                self.send_json(
                    400,
                    {"error": f"Văn bản tối đa {MAX_REVIEW_CHARS:,} ký tự."},
                )
                return
            if not normalize_review(raw_text):
                self.send_json(400, {"error": "Nhập nội dung đánh giá trước khi phân tích."})
                return
            try:
                segmented = preprocess_review(raw_text, segmenter)
                prediction = int(estimator.predict([segmented])[0])
            except Exception:
                self.send_json(500, {"error": "Không xử lý được nội dung này."})
                return
            if prediction not in (0, 1):
                self.send_json(500, {"error": "Mô hình trả về nhãn ngoài phạm vi."})
                return
            label = "có lời phàn nàn" if prediction == 1 else "không có lời phàn nàn"
            self.send_json(
                200,
                {
                    "prediction": prediction,
                    "label": label,
                    "interpretation": "Gợi ý của mô hình; cần con người xem xét trong ngữ cảnh.",
                    "model": public_metadata["model"],
                    "selected_run": public_metadata["selected_run"],
                },
            )

    return DemoHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Đường dẫn gốc của bản dự án đã nghiệm thu.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Demo chỉ bind loopback để không phát tán văn bản nhập.")
    estimator, metadata, segmenter = load_model(args.project_root)
    handler = create_handler(estimator, metadata, segmenter)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Demo sẵn sàng tại http://{args.host}:{args.port}")
    print(f"Run: {metadata['selected_run']} | Model: {metadata['model']}")
    print("Văn bản nhập chỉ ở trong bộ nhớ của request; server không ghi request log.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
