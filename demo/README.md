# Demo phát hiện lời phàn nàn

Demo cục bộ tải đúng artifact **S0 seed 42** đã được chọn bằng Macro-F1 trên validation. Demo không fit lại mô hình và không đọc tập test. Điểm validation hiển thị trên giao diện là 0,9207; đó không phải điểm kiểm tra hay xác suất của dự đoán.

## Chạy trên Windows

Từ thư mục gốc dự án, dùng Python 3.13:

```powershell
py -3.13 -m venv staging/.venv_demo
staging/.venv_demo/Scripts/python.exe -m pip install --upgrade pip
staging/.venv_demo/Scripts/python.exe -m pip install -r demo/requirements.txt
staging/.venv_demo/Scripts/python.exe demo/app.py
```

Mở `http://127.0.0.1:8765`. Dừng máy chủ bằng `Ctrl+C`.

Khi khởi động, ứng dụng kiểm tra hash của protocol, selection, run metadata, pipeline và năm tệp nguồn; đồng thời yêu cầu đúng phiên bản NumPy, scikit-learn và Underthesea. Hash của protocol và mã nguồn chuẩn hóa CRLF/LF thành LF để cùng một nội dung hoạt động trên Windows và Git checkout; hash của selection, run metadata và pipeline vẫn kiểm tra theo byte gốc. Các trường hash trong artifact tiếp tục đối chiếu với provenance của lần chạy. Nếu nội dung hoặc artifact khác bằng chứng đã nghiệm thu, ứng dụng dừng thay vì tải artifact khác.

Có thể kiểm tra quy tắc chuẩn hóa dòng bằng kiểm thử hồi quy:

```powershell
staging/.venv_demo/Scripts/python.exe -m unittest tests.test_demo_integrity
```

## Phạm vi và quyền riêng tư

- Máy chủ chỉ bind `127.0.0.1`; không cấu hình phục vụ mạng công cộng.
- Văn bản nhập được chuẩn hóa, phân đoạn bằng Underthesea 9.5.0 rồi đưa qua pipeline S0. Nội dung không được ghi vào log hoặc tệp bởi demo.
- Nhãn hiển thị theo protocol: `0` là “không có lời phàn nàn”, `1` là “có lời phàn nàn”.
- Demo không hiển thị xác suất hoặc độ tin cậy: LinearSVC chưa được hiệu chuẩn thành xác suất.
- Các câu mẫu được viết riêng để trình diễn, không lấy từ tập test.
- Nhãn phản ánh định nghĩa dữ liệu nghiên cứu; không đại diện cho xếp hạng sao, mức hài lòng tổng thể hoặc sự thật khách quan về sản phẩm.

## Bằng chứng demo

Ảnh giao diện và receipt runtime nằm trong gói bàn giao kỹ thuật tại `demo/evidence/`; chúng không thuộc repo mã nguồn public dạng source-only. Receipt chỉ ghi phiên bản môi trường và hash artifact, không ghi văn bản nhập. Ảnh minh họa dùng câu mẫu tổng hợp, không dùng review lấy từ dữ liệu nghiên cứu.
