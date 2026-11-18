"""Exercise the local demo API with synthetic examples; save no review text."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8765"


def request_json(path: str, payload: object | None = None) -> tuple[int, dict]:
    body = None
    headers = {}
    method = "GET"
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = Request(BASE_URL + path, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def check(name: str, passed: bool) -> None:
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
    if not passed:
        raise SystemExit(1)


def main() -> None:
    try:
        status, health = request_json("/api/health")
    except (URLError, TimeoutError) as exc:
        raise SystemExit("Demo server is unavailable at 127.0.0.1:8765") from exc
    check(
        "health is ready and bound to selected validation run",
        status == 200
        and health.get("status") == "ready"
        and health.get("selected_run") == "S0_seed42_6491091012fc"
        and health.get("protocol_id") == "viocd-segmentation-core-v1",
    )

    examples = (
        (
            "synthetic non-complaint prediction",
            "Sản phẩm đúng mô tả, chất lượng tốt và giao hàng nhanh.",
            0,
            "không có lời phàn nàn",
        ),
        (
            "synthetic complaint prediction",
            "Máy bị lỗi sau vài ngày sử dụng, giao hàng chậm và bộ phận hỗ trợ chưa phản hồi.",
            1,
            "có lời phàn nàn",
        ),
    )
    for name, text, expected_id, expected_label in examples:
        response_status, output = request_json("/api/predict", {"text": text})
        check(
            name,
            response_status == 200
            and output.get("prediction") == expected_id
            and output.get("label") == expected_label,
        )

    blank_status, blank_output = request_json("/api/predict", {"text": "   "})
    check(
        "blank input is rejected",
        blank_status == 400 and "error" in blank_output,
    )
    long_status, long_output = request_json("/api/predict", {"text": "x" * 5001})
    check(
        "input over 5,000 characters is rejected",
        long_status == 400 and "error" in long_output,
    )
    print("No request body or review text was written by this verifier.")


if __name__ == "__main__":
    main()
