# Phát hiện lời phàn nàn trong đánh giá sản phẩm tiếng Việt

Các thuật ngữ dùng thống nhất trong bài toán nghiên cứu này. Thuật ngữ mô tả đối tượng và ý nghĩa nhãn; quy trình thực hiện nằm trong đặc tả và giao thức.

## Ngôn ngữ của bài toán

**Đánh giá sản phẩm (review)**:
Toàn bộ nội dung một người viết để nhận xét trải nghiệm về một sản phẩm hoặc ứng dụng. Một review có thể chứa nhiều câu và nhiều ý.
_Tránh_: dùng “câu” để gọi mọi mẫu, hoặc dùng “đánh giá” mà không rõ đang nói review hay phép đo mô hình.

**Lời phàn nàn (complaint)**:
Nội dung thể hiện sự bất mãn về một sự việc hoặc sản phẩm không đáp ứng kỳ vọng, theo định nghĩa nhãn của corpus. Review vừa khen vừa nêu điều không đạt kỳ vọng vẫn có thể chứa lời phàn nàn.
_Tránh_: đồng nhất complaint với cảm xúc tiêu cực, lời xúc phạm hoặc số sao thấp.

**Không có lời phàn nàn (non-complaint)**:
Nhãn dành cho review không thỏa định nghĩa lời phàn nàn theo hướng dẫn nhãn đang sử dụng. Nhãn này không nhất thiết có nghĩa người viết hài lòng hoặc tích cực.
_Tránh_: gọi mọi non-complaint là “tích cực”.

**Cảm xúc (sentiment)**:
Sắc thái đánh giá tích cực, tiêu cực hoặc trung tính của nội dung. Đây là khái niệm liên quan nhưng khác sự hiện diện của lời phàn nàn.
_Tránh_: thay nhãn sentiment bằng nhãn complaint khi chưa gán lại theo cùng định nghĩa.

**Miền (domain)**:
Nhóm bối cảnh sản phẩm hoặc ứng dụng mà review đề cập. Các miền trong corpus có thể khác nhau về cách diễn đạt và đối tượng được nhận xét.
_Tránh_: nhầm miền dữ liệu với tên miền website hoặc giả định một miền đại diện cho mọi ngành hàng.

**Nhãn chuẩn (gold label)**:
Nhãn tham chiếu được chấp nhận theo một quy trình gán nhãn xác định để chấm kết quả. Nguồn và quy trình tạo nhãn quyết định phạm vi tin cậy của nhãn đó.
_Tránh_: gọi nhãn do mô hình tự sinh là nhãn người gán độc lập.

**Bất đồng nhãn**:
Trường hợp những người gán nhãn đưa ra các nhãn khác nhau cho cùng một review theo cùng bộ hướng dẫn. Bất đồng có thể phản ánh ranh giới ngữ nghĩa hoặc sự chưa rõ của hướng dẫn.
_Tránh_: mặc định một bên sai chỉ vì bất đồng với mô hình.

**Mẫu chẩn đoán tổng hợp**:
Review hoặc cặp review được soạn có chủ đích để thăm dò một hiện tượng ngôn ngữ. Nó không tự đại diện cho phân bố review thực tế.
_Tránh_: gọi độ đúng trên mẫu tự soạn là độ chính xác ngoài thực địa.

**Xác nhận độc lập**:
Đánh giá trên dữ liệu không tham gia quá trình phát triển hoặc lựa chọn phương pháp đang được xác nhận. Việc chạy lại trên dữ liệu đã biết không tự tạo tính độc lập.
_Tránh_: gọi một cách chia lại corpus đã tiếp xúc là dữ liệu hoàn toàn chưa thấy.

**Đánh giá hậu nghiệm**:
Phân tích được thực hiện khi đã có lịch sử tiếp xúc dữ liệu hoặc kết quả liên quan. Nó vẫn có thể cung cấp bằng chứng mô tả nếu lịch sử và giới hạn được công khai.
_Tránh_: mô tả hồi cứu như một kế hoạch đã đăng ký từ trước.
