# Giao thức thực nghiệm đề xuất — viocd-segmentation-core-v1

Ngày soạn: 16/11/2026. Chủ trì kỹ thuật: đầu mối ML/data; phản biện: QA; người khóa phiên bản sau tích hợp: điều phối. Đây là vai trò agent, không phải xác nhận đóng góp của ba sinh viên.

**Trạng thái: đã khóa hồ sơ chuẩn bị; receipt là `preparation_lock.json`.** Chưa huấn luyện hoặc submit Kaggle. Sau review nguồn thay thế, điều phối đã chấp nhận ViOCD cho core preparation ngày 16/11; quyết định được đưa vào snapshot chuẩn bị; việc triển khai và runtime GPU vẫn thuộc gate kế tiếp. [protocol.json](protocol.json) là cấu hình máy đọc; sau QA, điều phối ghi receipt/SHA để khóa. `PREPARATION_READY` không đồng nghĩa `TRAINING_READY`.

## 1. Phạm vi và quyết định dữ liệu

Giữ nguyên đề tài **Xây dựng và đánh giá hệ thống phát hiện lời phàn nàn trong đánh giá sản phẩm tiếng Việt**. Chọn UIT-ViOCD vì nhãn complaint/non-complaint đúng bài toán, không vì đã tốn công triển khai. [Rà soát lựa chọn dữ liệu](DATASET_OPTIONS_REVIEW.md) đã hoàn tất và điều phối chấp nhận lựa chọn cho core preparation: chưa xác minh được nguồn lớn hơn phù hợp để thay trực tiếp; ViSFD chỉ là ứng viên văn bản cần gán lại nhãn cho nhánh mới. Nếu có nguồn khác với bằng chứng phù hợp hơn về định nghĩa nhãn, chất lượng và quyền dùng, mở quyết định thay đổi, tái tạo manifest và chỉnh protocol trước fit. Không ghép nhãn cảm xúc/số sao thành complaint một cách tự động.

Nếu giữ ViOCD, dùng revision `25e1851683ebc407d28e5a2d514b7ae0eadd4e4b` và tập lịch sử sau vệ sinh 4.157 train, 544 validation, 546 test. Nguồn CSV đã được tải lại trong bộ nhớ và kiểm tra hash/count vào 16/11; bằng chứng [data_manifest.json](data_manifest.json). Việc này xác minh snapshot, không biến corpus đã xem thành dữ liệu mới.

Không thu thập hoặc tự gán nhãn corpus mới trong gói core. Nhánh xác nhận độc lập cần kế hoạch riêng với dữ liệu mới thật, quyền dùng và nhãn người thật; phần này chưa được thực hiện và không là blocker của bước chuẩn bị core.

## 2. Câu hỏi, giả thuyết và đối chứng

**Câu hỏi chính:** Trong cùng lịch tinh chỉnh một phần PhoBERT và cùng split, thay quy trình tách từ Underthesea bằng VnCoreNLP/RDR ảnh hưởng ra sao đến Macro-F1, đầu vào token và lỗi phân loại?

Giả thuyết có thể kiểm tra: quy trình tách từ được dùng lúc tiền huấn luyện có thể thay đổi biểu diễn đầu vào và hiệu năng. Chưa có cơ sở định trước chiều hoặc độ lớn thay đổi. Tác giả [PhoBERT khuyến nghị RDRSegmenter](https://github.com/VinAIResearch/PhoBERT#notes); khuyến nghị không chứng minh Underthesea sai hoặc RDR chắc thắng.

RDR có bước xử lý dấu thanh/câu đi kèm. Kết luận áp dụng cho **quy trình tách từ cụ thể**, không quy riêng nguyên nhân cho một thuật toán nội bộ không được tách. Cả hai giữ lowercase kế thừa; không gọi toàn bộ tiền xử lý đã giống tiền huấn luyện.

| Mã | Cấu hình | Lượt fit |
|---|---|---:|
| S0 | Underthesea → TF-IDF word1–2gram, min_df2 → LinearSVC | 1, seed42 |
| P0 | Underthesea → PhoBERT; mở hai encoder cuối và classifier | 3, seed42/43/44 |
| P1 | RDR → PhoBERT; cùng phạm vi tham số học và lịch train P0 | 3, seed42/43/44 |

Đối chứng chính là **P1 so với P0**. S0 là baseline mạnh được fit lại, không phải điểm lấy từ một tập khác. Core có sáu lượt PhoBERT, không mặc định chạy factorial12 lượt. Full fine-tuning, CV, char n-gram và augmentation là **deferred**, không tự kích hoạt khi điểm test chưa đẹp.

## 3. Cấu hình chính xác và nguồn kế thừa

Đã đọc source canonical ở commit `fd7369774fd53e5a1e6a930b2909fa4465ec38d6`, notebook tham chiếu `baseline_reference/ViOCD_Kaggle_GPU_3seed.ipynb`, cell3 (loader/baseline), cell12 (train), cell13 (môi trường), cell14 (restore/demo). Bản tham chiếu phải giữ nguyên; code triển khai mới chưa có ở lượt chuẩn bị này.

| Thành phần | Giá trị đề xuất khóa |
|---|---|
| PhoBERT | `vinai/phobert-base`, revision `01daacda68afe13d83023d16ec647239e344a1e6`, classifier2 lớp |
| Tokenizer | Cùng revision, `use_fast=False`, truncation128 gồm special tokens, dynamic pad bội8 |
| Normalization | Unicode NFC → lowercase → gộp whitespace, trước segmenter; không xóa stopwords/stemming |
| Underthesea | 9.5.0, `word_tokenize(text, format="text")` |
| RDR | VnCoreNLP commit `62bbc58fe5d113c898eae112656be97dcf50b3a0`; jar1.2 + hai tài nguyên wseg đã hash; wrapper0.1.4, pyjnius1.9.0 |
| Ghép câu RDR | `annotators=["wseg"]`; ghép các câu theo thứ tự bằng một dấu cách, không xóa dấu câu |
| Phạm vi học | Hai encoder cuối và classifier; embedding và mười encoder đầu đóng băng và đặt eval ở từng epoch |
| Train | 3 epoch, lr3e-5, microbatch8, accumulation4; nominal effective batch32 |
| Optimizer | PyTorch AdamW fused, weight_decay0.01; betas0.9/0.999, eps1e-8, amsgradFalse, phải assert actual params |
| Schedule/gradient | Linear decay, warmup floor10% tổng optimizer steps; clip norm1; scale gradient theo số mẫu thật của từng nhóm accumulation cuối |
| Precision | CUDA autocast float16 + GradScaler; scheduler chỉ tiến khi optimizer step không bị AMP skip; TF32 bật |
| Loader | Train shuffle; val/test không shuffle; workers0; pin_memoryTrue; torch threads4 |
| Dropout | Không override pretrained config: hidden0.1, attention0.1, classifier_dropout null dùng cơ chế mặc định của classifier |
| Khởi tạo | `set_seed` trước từng fit; tải lại base revision, không lấy checkpoint đã tinh chỉnh ở Run4/Run5 |

TF-IDF giữ `tokenizer=str.split`, `token_pattern=None`, `lowercase=False`, `ngram_range=(1,2)`, `min_df=2`; normL2, IDF/smoothing bật, sublinearFalse. SVM C1, L2, squared_hinge, dualauto, tol1e-4, max_iter1000, class_weightNone, random_state42. Defaults đã đối chiếu với sklearn1.8.0 cục bộ; triển khai mới phải lưu `get_params` thay vì phụ thuộc ngầm.

Môi trường lịch sử quan sát: Python3.13.16; numpy2.1.3, pandas2.3.3, sklearn1.8.0, underthesea9.5.0, matplotlib3.10.8, transformers4.51.3, tokenizers0.21.4, torch2.11.0+cu130/CUDA13/T4. Đây không phải xác minh image Kaggle mới. Không âm thầm thay phiên bản/thuật toán nếu image mới không tương thích: ghi bằng chứng và amendment trước fit. Xem [bằng chứng RDR](rdr_assets_verification.json); local Java/wrapper synthetic smoke đã qua, Linux Kaggle wrapper và GPU pilot chưa chạy. [Audit tách từ toàn corpus](segmentation_audit.json) đã xử lý thật 5.247 review bằng cả Underthesea và RDR tại máy cục bộ: mỗi nhánh có 0 output rỗng,0 trùng nội bộ và0 overlap qua split; cohort không đổi. Đây là tiền xử lý/kiểm tra dữ liệu, không phải huấn luyện.

[Audit tokenizer đầu vào](tokenizer_input_audit.json) cũng đã chạy cục bộ bằng config/vocab/BPE đã ghim, không tải trọng số. Cả hai phương án có 0 overlap qua split ở IDs đầy đủ và IDs cắt128; số review bị cắt là 15 train / 0 validation / 2 test mỗi phương án. Mỗi nhánh có 1 dòng train trùng một input train khác sau cắt128: một nhóm hai dòng, không có xung đột nhãn trong nhóm của từng nhánh. Điều phối chấp nhận giữ nguyên cohort và trọng số theo review trước fit; ghi đây là giới hạn biểu diễn nội bộ train, không là cross-split leakage. Không khẳng định mọi input duy nhất hoặc cặp row hai nhánh là cùng một cặp nếu chưa kiểm tra. Phải tái chạy assertions bằng trainer mới trước fit.

## 4. Tách phát triển và chấm test

1. Kiểm tra source/model/data SHA, split membership, dependencies và đầu vào; chưa fit.
2. Chuẩn bị đúng S0/P0/P1; audit normalized/segmented/truncated-token-ID cross-split collisions. Gặp collision thì dừng để sửa protocol, không âm thầm đổi dân số test hoặc dùng nhãn test để chọn cách lọc.
3. Fit train và tính validation từng epoch. **Trainer mới phải tách riêng chấm test**, thay vì kế thừa lệnh predict test nằm cuối hàm `fine_tune` cũ.
4. Mỗi seed chọn checkpoint có validation Macro-F1 cao nhất trong 3 epoch; hòa chính xác giữ epoch sớm nhất.
5. Lưu tất cả validation predictions/history, chọn cấu hình theo quy tắc dưới và ghi `selection.json` không có test metrics.
6. Sau khóa lựa chọn, chấm mô tả cuối vòng tất cả cấu hình đã tuyên bố trên cùng test546. Lưu dự đoán, khôi phục checkpoint và kiểm tra lại; không mở vòng thử cấu hình dựa trên test.

Lịch fit PhoBERT đề xuất: P0_42 → P1_42 → P1_43 → P0_43 → P0_44 → P1_44, nhằm tránh tất cả lượt của một phương án nằm liền nhau. So sánh thời gian vẫn chỉ là số đo của môi trường thực tế, không có bảo đảm loại bỏ mọi nhiễu tải.

## 5. Quy tắc chọn mô hình và demo

Tính S0 validation Macro-F1 một lượt; với P0/P1 lấy trung bình số học của ba validation Macro-F1 từ checkpoint đã chọn. Chọn cấu hình lớn nhất; chênh lệch tuyệt đối ≤1e-12 xử lý như hòa số học theo thứ tự S0 → P0 → P1. Đây là tie rule có trước, không chứng minh tương đương khoa học.

Không chọn seed bằng test; không gộp ba F1 thành ensemble. Demo chính dùng họ mô hình được validation chọn; nếu là PhoBERT, dùng seed42 cố định và ghi đúng điểm riêng của checkpoint. Demo nghiên cứu có thể cho xem các phương án khác nhưng phải ghi nhãn. S0 thắng validation là kết quả hợp lệ. Không refit lên train+validation trong core vì như vậy phải tạo và nghiệm thu checkpoint khác.

## 6. Thước đo và phân tích khóa trước

- Chính: Macro-F1 hai nhãn `[0,1]`, zero_division0; chênh lệch mean(P1)−mean(P0). Bảng có cả validation, từng seed, mean±SD mẫu.
- Bổ sung: precision/recall/F1 từng lớp, accuracy, confusion matrix; theo miền kèm N/nhãn; số token và tỉ lệ truncation; trainable params, thời gian và VRAM thật. Không gọi softmax là xác suất tin cậy đã hiệu chỉnh.
- Bootstrap:10.000 lần, seed20261116, lấy mẫu review ghép cặp phân tầng nhãn; cùng chỉ số mẫu cho S0 và mọi seed; percentile2.5–97.5. Estimand là chênh lệch mean F1 của những mô hình đã fit, không phải F1 của ensemble. CI có điều kiện trên mô hình/split; không bao trùm model-selection bias, các seed tương lai hay miền mới. CI chứa0 không xác lập ưu thế hoặc tương đương.
- Lỗi: xét P0/P1 seed42, ba nhóm P0 sai/P1 đúng, P0 đúng/P1 sai, cùng sai. Lấy tối đa3 review mỗi nhãn thật trong mỗi nhóm; phân bổ miền bằng largest remainder, tên miền tăng dần khi hòa, sampling seed42. Ghi số lỗi toàn nhóm trước chọn ví dụ. Đây là ca minh họa, không là ước lượng lỗi đại diện.
- Tách ranh giới complaint với sentiment; nhóm nguyên nhân được agent đề xuất chỉ là giả thuyết. Không viết người gán nhãn độc lập/đồng thuận sinh viên nếu chưa thực hiện.

## 7. Điều kiện khóa và quản lý thay đổi

Điều phối chỉ đóng dấu freeze sau khi có quyết định dữ liệu, QA phản biện, đọc lại cấu hình/đường dẫn và readiness checker. Ghi SHA của protocol/data manifest/source trong receipt riêng, không đặt hash tự tham chiếu trong chính file. Thay corpus, split, normalization, segmenter resource, metric, budget hoặc tie rule sau freeze phải có amendment và version mới, trước lượt fit bị ảnh hưởng.

`TRAINING_READY` còn cần trainer triển khai/kiểm thử và Kaggle runtime pilot thật. Những việc này thuộc giai đoạn tiếp theo, không được đánh dấu hoàn thành bằng checklist. Chưa có GPU quota mới, điểm số P1, kết quả full-fine-tuning hoặc tập gold mới để công bố.

## 8. Artifact giao giữa các đầu mối

ML → QA: sourceSHA/protocolSHA, runtime manifest, data/collision audits, tất cả history/validation predictions, lựa chọn khóa, checkpoint/config/tokenizer và số tham số học. Evaluation → QA: test predictions keyed theo split/source row, confusion/metric/CI tái tính, restore equality, lỗi có mẫu số. QA → báo cáo: bảng/hình đã truy nguồn và giới hạn diễn giải. Báo cáo → điều phối: Word/PDF đã cập nhật fields và kiểm tra render; bản đóng gói mới có inventory/hash.

Ảnh minh chứng chụp từ phiên bản thực tế sau khi có kết quả; không chụp trang chuẩn bị để gọi là đã train. Dữ liệu thô, trọng số, log đầy đủ và đầu ra cấp dòng giữ riêng; public source theo phạm vi người dùng cho phép, không suy ra quyền phân phối corpus.

Nguồn phương pháp: [scikit-learn về leakage/Pipeline](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), [model selection/CV](https://scikit-learn.org/stable/modules/cross_validation.html), [Dror et al. về significance trong NLP](https://aclanthology.org/P18-1128/). Không có CV hoặc significance mới được chạy ở bước chuẩn bị.
