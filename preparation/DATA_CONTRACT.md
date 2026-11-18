# Hợp đồng dữ liệu cho phiên bản nghiên cứu mới

Ngày kiểm tra: 16/11/2026. Chủ sở hữu: đầu mối dữ liệu; người phản biện: QA; người chốt lựa chọn nguồn: điều phối. Đây là hợp đồng **đã được chấp thuận và khóa trong hồ sơ chuẩn bị**; điều phối đã chấp nhận ViOCD cho core preparation sau [review nguồn thay thế](DATASET_OPTIONS_REVIEW.md) ngày 16/11. Thay nguồn phải có quyết định có bằng chứng và tái tạo manifest/protocol, không vì mong điểm cao hơn.

## 1. Đơn vị mẫu và ý nghĩa nhãn

Một mẫu là toàn bộ một review tiếng Việt, không phải từng câu. Đầu vào suy luận là review không rỗng; đầu ra 0/1 tương ứng không/có lời phàn nàn theo corpus. Complaint liên quan điều không đáp ứng kỳ vọng; không đồng nhất với negative sentiment hoặc số sao. Nguồn: [Vietnamese Complaint Detection on E-Commerce Websites](https://arxiv.org/abs/2104.11969).

Đề xuất UIT-ViOCD vì task fit. Không chuyển nhãn dataset sentiment/hate-speech thành complaint, không thay gold nguồn bằng nhãn AI để tăng điểm. Nhãn source vẫn có giới hạn chất lượng; mô hình khớp nhãn không chứng minh mọi review được hiểu đúng về mặt ngữ dụng.

## 2. Snapshot nguồn được kiểm chứng

- Repository `tarudesu/ViOCD`, revision `25e1851683ebc407d28e5a2d514b7ae0eadd4e4b`.
- Ba URL `https://huggingface.co/datasets/tarudesu/ViOCD/resolve/<revision>/{train,val,test}.csv`; không dùng `main` trôi.
- Ngày16/11 đã tải lại bytes vào bộ nhớ, đối chiếu SHA256 với audit15/11, tái tính vệ sinh và split counts; không fit/train và không lưu raw corpus ở thư mục chuẩn bị.
- [data_manifest.json](data_manifest.json) ghi bytes/SHA nguồn, counts, nhãn/miền và digest membership. Mô tả paper5.485 so với raw snapshot5.484 là khác biệt nguồn được công khai, không tự thêm một mẫu cho đủ.

| Split | Raw | Xung đột nhãn trong split | Trùng nội bộ | Trùng với split ưu tiên cao hơn | Giữ lại | Nhãn0 / nhãn1 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 4.387 | 2 | 153 | 75 | 4.157 | 1.928 / 2.229 |
| Validation | 548 | 0 | 0 | 4 | 544 | 262 / 282 |
| Test | 549 | 0 | 3 | 0 | 546 | 268 / 278 |

Tổng giữ 5.247. Missing/blank đều0 trên snapshot. Bốn miền là app, cosmetic, fashion, mobile. N theo miền train:1605/1016/1183/353; val203/135/154/52; test194/149/157/46 theo đúng thứ tự miền trên. Không coi mobile46 là đủ chứng minh độ bao phủ ngành hàng.

## 3. Thuật toán vệ sinh cố định để đối chiếu lịch sử

1. Đọc từng CSV giữ nguyên source row index (pandas zero-based), yêu cầu cột review/label/domain; label chỉ0/1, domain có giá trị và thuộc bốn miền dự kiến.
2. Loại review/label thiếu; chuẩn hóa NFC, lowercase, collapse whitespace; loại chuỗi rỗng.
3. **Trong mỗi split riêng**, nếu cùng normalized text có nhiều nhãn, loại toàn bộ các dòng của nhóm xung đột; sau đó giữ dòng xuất hiện đầu tiên của các nhóm trùng còn lại.
4. Ưu tiên giữ test → validation → train. Sau khi vệ sinh nội bộ, loại normalized text đã thuộc split ưu tiên cao hơn. Bộ `seen` chứa text, không chứa nhãn.
5. Giữ split/source index để join predictions; phải key bằng `(split, source_row)` vì cùng số index ở CSV khác không phải cùng mẫu. Không reset index rồi làm mất liên kết nguồn.

Đây là corpus-wide **heldout-text-aware hygiene**, không gọi là toàn bộ tiền xử lý train-only.75 train removals gồm41 trùng test và34 trùng validation-only. Audit trên revision này cho thấy bỏ bước kiểm xung đột nhãn của heldout không làm đổi train membership (test/val có 0 xung đột). Điều đó không cho phép khẳng định mọi revision tương lai đều như vậy.

Giữ cohort này nhằm so sánh cùng dân số đã xác định. Đổi train-first dedup sẽ tạo dân số khác; không gọi đó là tự động sửa số liệu cũ. Tập đã xem vẫn là hậu nghiệm dù đổi code, seed hoặc chia lại.

## 4. Hợp đồng fit và preprocessing

- Raw/normalized text không phải đặc trưng đã fit. Underthesea và RDR dùng tài nguyên được ghim, không huấn luyện lại trên corpus ở core.
- TF-IDF vocabulary/IDF, classifier và mọi tham số học từ corpus chỉ được fit trên **retained train**. Validation chỉ chọn epoch/configuration; test không dùng chọn tham số, threshold, quy tắc lọc hoặc ví dụ để vá mô hình.
- Mọi nhánh train/validation/test/demo đi qua cùng hàm normalization và segmenter của cấu hình; tokenizer/version/maxlen giống nhau. Chỉ khác phương pháp tách từ giữa P0/P1.
- RDR trả nhiều câu thì ghép theo thứ tự bằng một dấu cách. Không tách mỗi câu thành một mẫu hoặc nhân nhãn toàn review xuống câu.
- Trước fit mới phải audit trùng/collision ở normalized text, segmented text và chuỗi tokenizer IDs đã truncation, dùng key input để kiểm tra label-blind ở biên split. Nếu cross-split collision xuất hiện: dừng, lưu counts, review và amendment; không tự loại mẫu theo nhãn test hay giữ cấu hình có số liệu thuận lợi.
- [Audit tách từ toàn corpus](segmentation_audit.json) đã chạy thật Underthesea và RDR trên5.247 review: từng split/nhánh có 0 output rỗng,0 trùng nội bộ và0 overlap qua split. Normalized disjoint cũng đã fresh-verify. [Audit tokenizer IDs](tokenizer_input_audit.json) kiểm tra riêng đầu vào đầy đủ và sau cắt128: cả hai phương án có 0 collision qua split. Chưa đủ metadata để khẳng định hết near-duplicate, quan hệ cùng sản phẩm hoặc cùng người dùng.
- **Giới hạn đã biết:** mỗi phương án có 1 dòng train trùng một input train khác sau truncation128, mặc dù chuỗi IDs đầy đủ không trùng. Audit xác minh trong từng phương án có một nhóm gồm hai dòng, nhãn của hai dòng không xung đột; không xuất row IDs hoặc review text. Giữ nguyên cohort và trọng số theo review, theo quyết định điều phối trước train; không tự sửa nhãn/drop mẫu/tăng max_length. Đây là trùng nội bộ tập học, không phải rò rỉ qua split. Chưa được suy rằng cặp row trùng ở hai phương án giống nhau nếu chưa kiểm tra. Không được viết mọi model input đều duy nhất.

## 5. Digest và kiểm tra bất biến

SHA nguồn là bytes CSV tải về, không phải hash text sau làm sạch. Manifest có digest membership theo chuẩn UTF-8 của các dòng sắp theo source index: `split<TAB>source_row<TAB>label<TAB>domain<TAB>sha256(normalized_text)<LF>`. Manifest bổ sung digest ràng buộc dự đoán theo `split<TAB>source_row<TAB>label<TAB>domain<LF>`; nó cho phép numerical audit kiểm tra từng hàng prediction trỏ đúng khóa, nhãn chuẩn và miền đã khóa mà không lưu text hay digest text từng hàng. Chỉ các digest cuối được lưu; không xuất bảng cấp dòng vào hồ sơ chuẩn bị. Các digest phát hiện lệch membership/order/nhãn/miền, không chứng minh nhãn đúng.

Phải fail trước fit nếu sourceSHA, split counts, membership digest hoặc prediction-binding digest lệch; nếu thiếu cột, nhãn ngoài0/1, domain rỗng/khác tập dự kiến; hoặc nếu nguồn không truy cập được. Không tự fallback sang mirror/revision khác. SHA asset RDR và config/model revision được kiểm tra riêng, không trộn với hash CSV.

## 6. Quyền dùng và giới hạn chia sẻ

Người dùng đã xác nhận phạm vi nghiên cứu cá nhân/bài tập tại trường. Đây là cơ sở quản lý phạm vi nội bộ của dự án, không phải suy ra quyền phát hành lại corpus từ việc tải được trên Hugging Face. Dataset/code/library/paper đều phải được dẫn nguồn. Giữ raw corpus, checkpoint, log đầy đủ, review cấp dòng và dự đoán cấp dòng riêng tư; công khai code không đương nhiên cho phép công khai các tài nguyên đó.

Nếu muốn corpus xác nhận mới: khóa sampling/label guide, xác minh quyền dùng, lọc trùng toàn lịch sử, hai người thật gán nhãn độc lập và phân xử thật khi cần. Không lấy review sao thấp làm gold complaint; không gọi AI là người gán nhãn độc lập. Tập tuyển chọn ca khó phải ghi giới hạn đại diện.

## 7. Trạng thái bàn giao dữ liệu

**Đã xác minh:** fresh source bytes/hash; raw/retained counts; nhãn nhị phân; domain có giá trị; normalized text pairwise disjoint; Underthesea/RDR toàn corpus không có output rỗng hoặc trùng nội bộ/qua split; full/truncated tokenizer IDs không collision qua split; giới hạn1 dòng train trùng sau truncation ở từng phương án đã ghi; các digest và hồ sơ aggregate đã đọc lại.

**Đã chấp nhận cho core preparation:** ViOCD sau review nguồn thay thế; quyết định của điều phối chưa thay freeze receipt toàn giao thức.

**Chờ triển khai/pilot:** kiểm tra loader mới dùng hợp đồng này và tái kiểm tra các ranh giới đầu vào bằng code thực thi mới, môi trường Kaggle mới, kiểm tra fit boundary bằng test có đối chứng, training/evaluation thật. Không đánh dấu các mục này hoàn tất bằng tài liệu hoặc audit tiền xử lý cục bộ.

## 8. Lệnh tái lập audit chuẩn bị

Source thực thi được giữ tại [scripts/audit_data.py](scripts/audit_data.py). Script tải ba CSV ghim vào bộ nhớ, kiểm tra SHA, dựng lại chính xác cohort/digest, chạy cả hai segmenter và so mọi count/digest với hai manifest chuẩn. Nó kiểm tra SHA tài nguyên RDR và các phiên bản thư viện liên quan, không tải trọng số PhoBERT, không fit mô hình và không ghi review hoặc row-level hashes ra đĩa.

Chuẩn bị phụ thuộc/asset theo [RUNBOOK.md](RUNBOOK.md), rồi chạy từ thư mục gốc của repo mới:

~~~powershell
python -B .\preparation\scripts\audit_data.py --output .\staging\data_preflight_check.json
~~~

Phải kiểm tra exit code0 và status PASS, expected_comparison đều true trong output mới. Không ghi đè các bằng chứng đã khóa để chạy kiểm tra lại. Lượt chuẩn bị đã thực chạy với output [data_preflight_reproduction.json](data_preflight_reproduction.json); artifact ghi SHA của chính source đã chạy và SHA hai manifest được so, không gán hồi tố source hash cho các probe inline trước đó. Canonical data_manifest.json và segmentation_audit.json chỉ được đọc và được kiểm tra giữ nguyên bytes trước/sau.

VnCoreNLP wrapper có thể thay working directory của tiến trình; script bind mọi đường output tuyệt đối trước khi load wrapper. Tokenizer/BPE là audit riêng do MLOps giữ source, không được suy từ kết quả tách từ để thay kết quả tokenizer audit. Sau khi triển khai trainer mới, vẫn phải kiểm tra lại cùng hợp đồng tại đường gọi thực tế.
