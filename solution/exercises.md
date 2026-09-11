# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Temperature tăng dần làm cho câu trả lời chuyển từ tính xác định, chuẩn mực sang ngẫu nhiên và sáng tạo hơn. Ở 0.0 và 0.5, phản hồi tập trung vào các sự thật phổ biến (như vị thế xuất khẩu cà phê hay văn hóa tà áo dài) với văn phong nhất quán, chính xác. Khi nâng lên 1.0 và 1.5, mô hình bắt đầu chọn các sự thật độc lạ hơn, từ vựng đa dạng hơn nhưng ở mức 1.5 văn phong có thể trở nên lan man hoặc thiếu tự nhiên.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Nên chọn temperature trong khoảng 0.0 đến 0.3 cho chatbot hỗ trợ khách hàng. Mức này giữ cho câu trả lời mang tính ổn định, chính xác cao, tuân thủ đúng dữ liệu/chính sách của doanh nghiệp và tránh tình trạng mô hình tự sáng tạo thông tin sai sự thật (hallucination).

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> GPT-4o đắt hơn GPT-4o-mini khoảng 12.5 lần đối với đầu ra (giả định mức giá $10.00/M tokens của GPT-4o so với $0.60/M tokens của GPT-4o-mini). Với tổng nhu cầu ~10.5 triệu token đầu ra/ngày, việc dùng GPT-4o sẽ tốn chi phí lớn hơn rất nhiều. GPT-4o xứng đáng chi phí cho các bài toán phức tạp đòi hỏi lập luận logic sâu, phân tích hợp đồng pháp lý hoặc viết mã nguồn phức tạp. Ngược lại, nên dùng GPT-4o-mini cho các tác vụ thông thường như phân loại ý định người dùng (intent classification), tóm tắt văn bản ngắn, hoặc làm chatbot CSKH trả lời FAQ.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi khác biệt hoàn toàn về độ dài, ngữ cảnh và trường từ vựng. System prompt giáo viên tiểu học cho ra câu trả lời ngắn gọn, sử dụng ẩn dụ đơn giản (như cuốn sổ nhật ký chung) để trẻ 8 tuổi dễ hình dung. Trong khi đó, system prompt chuyên gia tài chính cho phản hồi dài, chi tiết với các thuật ngữ kỹ thuật chuyên sâu như "sổ cái phân tán (distributed ledger)", "mã hóa thuật toán" và "cơ chế đồng thuận (consensus mechanism)". System prompt đóng vai trò định hình bộ lọc hành vi, giúp điều chỉnh giọng văn, độ sâu kiến thức và đối tượng hướng tới của model.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Tiktoken trả về số token cao hơn ước lượng `số từ / 0.75` khoảng 30% đến 50% tùy thuộc vào đoạn văn. Lý do là vì các tokenizer hiện tại (như cl100k_base hay o200k_base) được tối ưu hóa chủ yếu cho tiếng Anh. Tiếng Việt có dấu và là ngôn ngữ đơn lập, dẫn đến việc bộ tokenizer phải tách các từ có dấu thành nhiều sub-token hoặc mã byte nhỏ hơn, làm gia tăng lượng token phát sinh so với tiếng Anh không dấu.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các giao diện hội thoại thời gian thực (interactive chat UI), nơi phản hồi của LLM dài và mất nhiều thời gian sinh (latency cao). Việc hiển thị từng token giúp giảm Time-To-First-Token (TTFT), tạo cảm giác phản hồi tức thì và nâng cao trải nghiệm người dùng. Ngược lại, non-streaming phù hợp hơn cho các hệ thống xử lý ngầm (backend pipeline), gọi API trả về dữ liệu cấu trúc (JSON), hoặc khi ứng dụng cần nhận đủ toàn bộ phản hồi để kiểm tra cú pháp/phân tích dữ liệu trước khi xử lý tiếp.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp tăng thời gian chờ sau mỗi lần thất bại, cho phép hệ thống bị sự cố hoặc quá tải có khoảng nghỉ đủ dài để phục hồi. Nếu hàng nghìn client cùng retry với delay cố định (ví dụ 1 giây), toàn bộ các request sẽ đồng loạt đập lại vào server cùng một thời điểm (hiện tượng Thundering Herd problem), tiếp tục làm sập server và khiến hệ thống rơi vào vòng lặp nghẽn mạng liên tục.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona là Trợ lý nghiên cứu sinh học và công nghệ sinh học. System prompt: "Bạn là một trợ lý nghiên cứu khoa học chuyên sâu về Sinh học và Công nghệ sinh học. Hãy trả lời các câu hỏi một cách chính xác, dựa trên cơ sở khoa học, sử dụng thuật ngữ chuyên ngành chuẩn xác. Luôn trình bày câu trả lời ngắn gọn, sử dụng danh sách dạng gạch đầu dòng và trả lời hoàn toàn bằng tiếng Việt." Việc yêu cầu "trả lời ngắn gọn và dùng gạch đầu dòng" giúp tối ưu hóa số lượng output token (tiết kiệm chi phí API) và tăng tính dễ đọc. Lựa chọn chỉ định "hoàn toàn bằng tiếng Việt" giúp đảm bảo model không tự động phản hồi bằng tiếng Anh khi gặp các thuật ngữ khoa học quốc tế.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Trợ lý chỉ lưu được 3 lượt hội thoại gần nhất trong bộ nhớ đệm (context window) nên dễ bị quên ngữ cảnh khi thảo luận dài, đồng thời không có bộ nhớ dài hạn để lưu trữ thông tin cá nhân hóa của người dùng. Đề xuất cải thiện: Triển khai cơ chế Summarization Memory (Tóm tắt ngữ cảnh). Cách triển khai: Khi lịch sử chat vượt quá 3 lượt, hệ thống sẽ tự động gọi một LLM nhẹ (như gpt-4o-mini) để tóm tắt các đoạn hội thoại cũ thành một đoạn văn ngắn (summary_context), sau đó nối đoạn tóm tắt này vào system prompt của các lượt chat tiếp theo thay vì gửi toàn bộ raw history.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026