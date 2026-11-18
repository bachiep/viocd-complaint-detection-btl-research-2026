# Ma trận yêu cầu và bằng chứng nghiệm thu

Ngày lập: 16/11/2026. Đề tài: **Xây dựng và đánh giá hệ thống phát hiện lời phàn nàn trong đánh giá sản phẩm tiếng Việt**.

## Căn cứ và cách đọc

Nguồn chuẩn là [hướng dẫn BTL gốc](../../../Huong_dan_bai_tap_lon_Xu_ly_ngon_ngu_tu_nhien.docx), đã đọc toàn bộ văn bản và bảng. [Quyết định thiết kế lại](../../reassessment_2026-11-16/DECISION_AND_PLAN.md) xác định phạm vi vòng mới. Chưa có đề cương học phần riêng để đối chiếu mọi nội dung giảng trên lớp.

- **BB:** yêu cầu bắt buộc trong hướng dẫn; **GY:** gợi ý/khuyến nghị của hướng dẫn; **ND:** yêu cầu bổ sung của người dùng; **TK:** quy tắc phương pháp do nhóm lựa chọn để thực hiện yêu cầu.
- **Lịch sử:** có hồ sơ cũ được đọc/đối chiếu, không có nghĩa đã chạy lại trong lượt này. **Chuẩn bị:** đã xác định cách nghiệm thu; artifact hoặc thực thi vòng mới còn phải kiểm tra.
- Bằng chứng không đồng nhất với kế hoạch: ghi một tiêu chí vào tài liệu chưa làm tiêu chí đó đạt. Bảng này không tự cho điểm hay xác nhận thay giảng viên.

## Yêu cầu, sản phẩm và phép kiểm tra

| ID / loại | Căn cứ | Yêu cầu áp dụng | Artifact đích và phép kiểm tra | Trạng thái khi lập |
|---|---|---|---|---|
| R01 BB | §1; §2 | Một bài toán rõ, đầu vào/đầu ra, người sử dụng, mục tiêu | Protocol/giới thiệu: review tiếng Việt → complaint/non-complaint; giải thích khác sentiment; kiểm tra ví dụ ranh giới nhãn | Đề tài giữ nguyên; vòng mới chuẩn bị |
| R02 BB | §1; §3.1 | Nguồn corpus, chất lượng và EDA | Data manifest: revision/hash, đơn vị mẫu, nhãn/miền, số dòng, độ dài, thiếu/trùng; tái tính từ nguồn đã ghim | Lịch sử có; manifest vòng mới cần kiểm tra |
| R03 BB | §2; §3.1 | Pipeline từ dữ liệu thô đến phân tích, lưu trung gian | Dữ liệu → xử lý → biểu diễn → fit → evaluate → phân tích; kiểm tra đường chạy và artifact contract | Lịch sử có; vòng mới chưa thực thi |
| R04 BB | §2 | Baseline và ≥1 đối chứng/cải tiến so sánh được | S0 TF-IDF+LinearSVC, P0/P1 PhoBERT; chạy trên cùng split và định nghĩa metric | Danh sách phương án đã đề xuất; chưa chạy mới |
| R05 BB | §2; §3.2 | Không train/fit vectorizer hoặc bước học tham số trên test | Split assertions; log tập fit; negative control làm lộ overlap/fit sai phải bị từ chối | Kiểm tra lịch sử đã có; vòng mới chưa nghiệm thu |
| R06 BB | §3.2 | Chọn cấu hình/epoch chỉ theo validation; ghi tham số, seed và môi trường | Protocol khóa trước; per-epoch history; decision record chỉ đọc validation; test chấm cuối theo giao thức | Cần khóa artifact cụ thể trước train |
| R07 BB | §3.2 | Ít nhất một khảo sát có kiểm soát | P0/P1 chỉ đổi quy trình tách từ, giữ model revision, train scope, normalization phía trước, split, seed, lịch train; audit diff cấu hình | Đã chọn câu hỏi; chưa có số mới |
| R08 BB + TK | §2; §3.3 | Chỉ số phù hợp và bảng so sánh | Nhóm chọn Macro-F1 chính; precision/recall/F1 từng lớp, accuracy, confusion, N. Tái tính độc lập từ dự đoán lưu | Tên metric là lựa chọn phù hợp với gợi ý, không phải ngưỡng điểm bắt buộc |
| R09 BB | §2; §4 | Phân tích lỗi/giới hạn | Error sample plan trước khi xem kết quả mới; lỗi có source/key nội bộ, nhãn thật, dự đoán và giả thuyết giải thích | Lịch sử có 10 lỗi; vòng mới chưa lấy mẫu |
| R10 GY | §2 | Nên minh họa 5–10 lỗi cụ thể | Ưu tiên 10 ví dụ có quy tắc, bao phủ hai lớp và hai hướng sai khi có đủ ca; ghi không đủ nếu số thực tế ít | Kế hoạch; không bịa để đủ số |
| R11 BB | §4.1 | PDF tên NLP_BTL_MaNhom.pdf | PDF cuối có bảng/hình truy nguồn; render toàn bộ trang; tên nhóm điền sau | Bản lịch sử có; chưa tạo bản vòng mới |
| R12 GY | §4.1 | Đề xuất 20–30 trang nội dung chính | Kiểm tra số trang chính riêng bìa/phần đầu/phụ lục; không gọi đây là giới hạn tuyệt đối | Dùng làm mục tiêu trình bày |
| R13 BB | §4.1 | Link notebook/repository xem/chạy phù hợp cho giảng viên | README trỏ đúng phiên bản; người dùng cấp quyền khi nộp; kiểm tra phiên bản chia sẻ không lộ dữ liệu ngoài phạm vi | Repo lịch sử có; quyền giảng viên user hoãn |
| R14 BB | §4.1 | README, môi trường và cách chạy lại từ đầu | Cài đặt, lấy nguồn đúng quyền, lệnh, output mong đợi, cách tái tạo bảng/hình; một lần kiểm tra môi trường mới | Cần cập nhật sau triển khai |
| R15 BB | §4.1 | Thư mục kết quả và minh họa | Source revision/hash → cấu hình → run ID → dự đoán → metric → hình/bảng; inventory/hash/negative control | Hợp đồng artifact cần khóa ở chuẩn bị |
| R16 BB + GY | §4.1 | Demo và thành viên hiểu phần mình; đề xuất 10+5 phút | Demo đúng pipeline, xử lý rỗng/dài, checkpoint restore; biên bản tập bảo vệ thật của nhóm. 10+5 chỉ là khuyến nghị | Chưa xác nhận hiểu biết/đóng góp sinh viên |
| R17 BB | §6 | Trích nguồn, phạm vi AI, không bịa kết quả/ảnh | References và phụ lục AI; metric truy artifact; screenshot thật có URL/run ID/ngày; không coi ảnh thay kiểm tra số liệu | Áp dụng xuyên suốt |
| R18 BB | §3.1; §6 | Tuân thủ quyền dùng và bảo vệ thông tin | User xác nhận nghiên cứu cá nhân/bài tập; giữ corpus, trọng số, prediction cấp dòng riêng; public code không tự mở quyền dữ liệu | Phạm vi sử dụng do user xác nhận; không suy ra quyền phân phối mới |
| U01 ND | Trao đổi người dùng | Huấn luyện thật trên GPU Kaggle | Giai đoạn train sau: runtime metadata GPU thật, nguồn submit/pull, đủ seed, log và checkpoint; preflight không được gọi huấn luyện hoàn tất | User yêu cầu chuẩn bị trước; chưa chạy mới |
| U02 ND | Trao đổi người dùng | Word chỉnh sửa được, chuẩn đóng quyển, mục lục/hình/bảng và số trang | Word TNR/cỡ chữ/căn lề thống nhất, field TOC/SEQ; cover không số, phần đầu La Mã, nội dung Arabic; Word export PDF và kiểm tra trang | Dùng quy cách lịch sử làm mốc; chưa sửa báo cáo ở chuẩn bị |
| U03 ND | Trao đổi người dùng | Nhóm 3 người, phân công đều, tên/MSSV bổ sung sau | Phân công dự kiến ≠ đóng góp thực tế; mỗi người có việc, bằng chứng và đọc chéo; không tự điền hoặc giả xác nhận | Danh tính và thực hiện người thật còn hoãn |
| U04 ND + TK | Trao đổi mới nhất | Chuẩn bị đầy đủ, có vai trò và phản biện trước khi triển khai | WBS/owner/dependency; protocol; manifest dữ liệu; preflight môi trường; QA gate; reviewer không tự duyệt phần mình | Chính là phạm vi công việc hiện tại |

Không có yêu cầu giảng viên bắt buộc mô hình mới, tự thu thập/gán nhãn, GPU, số mẫu tối thiểu, PhoBERT thắng baseline hoặc một mức F1 tối thiểu. Full fine-tuning, CV và tập xác nhận mới là nhánh có điều kiện trong kế hoạch, không được tự nâng thành điều kiện hoàn thành rubric.

## Liên hệ rubric và trách nhiệm

| Nhóm rubric | Điểm tối đa của hướng dẫn | ID liên quan | Chủ trì chuẩn bị / người phản biện |
|---|---:|---|---|
| Xác định bài toán và dữ liệu | 15 | R01,R02,R18 | Đầu mối dữ liệu / QA-methodology |
| Thiết kế và triển khai mô hình | 25 | R03,R04,R07 | Đầu mối triển khai / QA-code |
| Thực nghiệm và chỉ số | 25 | R05–R08,R15,U01 | Đầu mối thực nghiệm / QA-methodology và kiểm tra metric độc lập |
| Phân tích và thảo luận | 15 | R09,R10 | Đầu mối đánh giá / reviewer ranh giới nhãn và diễn giải |
| Báo cáo và tính tái lập | 20 | R11–R18,U02,U03 | Đầu mối tài liệu/tái lập / QA-release |

Các đầu mối trên là vai trò công việc của agent trong quá trình hỗ trợ, không phải nhân viên thật hoặc ba sinh viên đã làm. Tên người học để trống theo yêu cầu; việc tiếp nhận, chạy kiểm chứng và bảo vệ của nhóm phải được ghi nhận thực tế trước nộp. Không cộng điểm tối đa thành điểm tự chấm.

## Giới hạn không được che khuất

Test lịch sử đã tiếp xúc trong phát triển; Run5 lấy từ pool đã dùng. Chia lại, thêm seed, dùng CV hoặc viết lại mã không tạo tập độc lập. CI chứa0 không chứng minh ưu thế hoặc tương đương. Corpus mới chỉ gọi là xác nhận độc lập khi nguồn, nhãn và cách giữ khỏi lựa chọn mô hình thực sự đáp ứng điều kiện; AI không thay thế người gán nhãn độc lập.

Chi tiết bằng chứng lịch sử: [nghiệm thu nguồn](../../release_2026-11-15/source_acceptance.md), [đối chiếu rubric trước](../../reassessment_2026-11-16/rubric_review.md). Chúng là mốc đối chiếu; vòng mới phải có bằng chứng riêng.
