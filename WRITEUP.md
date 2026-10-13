# BÁO CÁO KỸ THUẬT VÀ PHÂN TÍCH CHUYÊN SÂU GPU FINOPS
**Dự án:** NimbusAI — GPU Cost Optimization & Infrastructure Efficiency  
**Tác giả:** FinOps Engineering Lead  
**Ngày thực hiện:** 13/10/2026  
**Trạng thái:** 11/11 Checks Passed (`verify.py`) | 15/15 Passed (`pytest`) | 5/5 Extensions Implemented  

---

## 1. TỔNG QUAN HIỆU QUẢ TỐI ƯU HÓA: BASELINE VS. OPTIMIZED

Dưới sự bùng nổ của các mô hình LLM, chi phí GPU của **NimbusAI** đã tăng đột biến do việc mua sắm hạ tầng theo phản xạ mà thiếu kiểm toán hiệu năng thực tế. Việc đo lường theo đơn vị truyền thống `$/GPU-giờ` tạo ra ảo giác chi phí: hai đội ngũ có thể trả cùng một mức giá thuê GPU mỗi giờ, nhưng đội ngũ tối ưu tốt hơn có thể phục vụ số lượng token gấp 5–10 lần. Do đó, toàn bộ bài lab này được chuẩn hóa theo đơn vị **`$/1M-token`** (đo lường đồng thời chi phí hạ tầng và hiệu quả sử dụng).

### Bảng số liệu tổng hợp Baseline vs. Optimized

| Chỉ số kinh tế | Trước tối ưu (Baseline) | Sau tối ưu (Optimized) | Mức cắt giảm (Savings) | Tỷ lệ giảm (%) |
|---|---|---|---|---|
| **Đơn giá phục vụ (`$/1M-token`)** | **$6.488** / 1M token | **$1.126** / 1M token | **-$5.362** / 1M token | **82.6%** |
| **Chi phí Inference ngày** | $48.87 / ngày | $8.48 / ngày | -$40.39 / ngày | **82.6%** |
| **Chi phí Inference quy tháng (30 ngày)** | $1,466 / tháng | $254 / tháng | -$1,212 / tháng | **82.6%** |
| **Chi phí Workload hạ tầng quy tháng** | $25,667 / tháng | $15,627 / tháng | -$10,040 / tháng | **39.1%** |
| **Lãng phí GPU để trống (Idle Waste)** | $600 / tháng | $0 / tháng (thu hồi) | -$600 / tháng | **100.0%** |
| **Lãng phí do cấu hình sai ("Util-Lie")** | $655 / tháng | $0 / tháng (right-size) | -$655 / tháng | **100.0%** |
| **TỔNG CHI PHÍ HÀNG THÁNG** | **$27,133 / tháng** | **$14,626 / tháng** | **-$12,507 / tháng** | **46.1%** |

> **Kết luận:** Tổng ngân sách GPU của NimbusAI giảm từ **$27,133/tháng** xuống còn **$14,626/tháng**, tương ứng tiết kiệm ròng **$12,507/tháng** (**46.1%** tổng ngân sách), vượt xa mục tiêu cam kết tối thiểu 40%.

---

## 2. PHÂN TÍCH CHI TIẾT TỪNG NHIỆM VỤ (MISSION 1 – MISSION 5)

### 2.1. Mission 1: Kiểm toán hiệu quả GPU & Vạch trần "GPU-Util Lie"

#### Kết quả thực nghiệm M1:
- Telemetry ghi nhận 11 cụm GPU phục vụ training và serving.
- Danh sách GPU bị đánh dấu "GPU-Util Lie" (`util >= 90%` nhưng `MFU < 30%`): **`gpu-h100-4`** và **`gpu-a10g-1`**.
- Chi phí GPU bỏ không (Idle waste): **$20.00/ngày** $\rightarrow$ **$600.00/tháng** (tập trung ở `gpu-h100-5` bị bỏ trống 8 giờ).

#### Trả lời các câu hỏi phân tích (Guide.md mục 4.3):
1. **GPU nào có `GPU-Util` cao nhất? MFU của nó là bao nhiêu?**  
   - `gpu-h100-4` đạt mức sử dụng clock `GPU-Util = 98.2%` (cao nhất hệ thống), nhưng chỉ số **MFU (Model FLOPs Utilization)** thực tế chỉ đạt **0.194 (19.4%)**.  
   - `gpu-a10g-1` cũng đạt `GPU-Util = 96.9%` nhưng MFU chỉ đạt **0.268 (26.8%)**.

2. **Tại sao `GPU-Util 98%` lại đi kèm với `MFU 20%`? Bản chất cơ chế phần cứng là gì?**  
   - Lệnh `nvidia-smi` báo cáo `GPU-Util %` dựa trên tỷ lệ thời gian một hoặc nhiều kernel CUDA đang active trên GPU trong cửa sổ lấy mẫu. Chỉ số này **chỉ đo thời gian hoạt động của clock (time-active clock), hoàn toàn không đo số phép tính FLOPs hữu ích thực sự sinh ra**.  
   - **Cơ chế nghẽn phần cứng (Hardware Stalls):**  
     * **Memory-Bound Stalls:** Trong các tác vụ decode token của LLM với batch size nhỏ (batch-1), cường độ số học (arithmetic intensity) chỉ đạt ~1–2 FLOPs/byte, nằm sâu bên dưới điểm gãy ridge point của H100 (~295 FLOP/byte theo Roofline model). Các Streaming Multiprocessor (SM) phải dừng chờ (stall) dữ liệu nạp từ bộ nhớ HBM3 qua bus bộ nhớ hơn 80% số chu kỳ xung nhịp.  
     * **Kernel Launch Overhead & Uncoalesced Memory Access:** Việc gọi nhiều kernel nhỏ liên tiếp khiến GPU bận rộn điều phối nhưng không khai thác được các Tensor Core FP16/BF16.  
   - **Tác động tài chính:** NimbusAI đang chi trả giá thuê $2.50/giờ cho H100 nhưng chỉ nhận được sức mạnh tính toán tương đương một GPU phân khúc thấp hơn.

3. **Lãng phí Idle tính ra bao nhiêu $/tháng? Chiếm bao nhiêu % chi phí?**  
   - `gpu-h100-5` có 8 giờ trong ngày ghi nhận `util < 10%` (hoàn toàn bỏ không). Với đơn giá on-demand của H100 là $2.50/giờ, công ty bị "đốt" $20.00/ngày, tương đương **$600/tháng**.  
   - Khoản lãng phí này chiếm **2.2%** tổng hóa đơn ban đầu và có thể cắt giảm ngay lập tức bằng chính sách auto-shutdown hoặc scale-to-zero.

---

### 2.2. Mission 2: Các đòn bẩy chi phí Inference

#### Kết quả thực nghiệm M2:
- Tổng số lượng request: **2,400 requests**, tổng số token xử lý: **7,533,027 tokens**.
- Chi phí Baseline (tất cả đẩy vào model Large, không cache, không batch): **$48.87/ngày** $\rightarrow$ **$6.488 / 1M-token**.
- Chi phí sau tối ưu hóa (Cascade + Prompt Cache + Batch): **$8.48/ngày** $\rightarrow$ **$1.126 / 1M-token** (**tiết kiệm 82.6%**).

#### Phân rã đóng góp của từng đòn bẩy (Decomposition Analysis):
- **Baseline (Naive):** $48.87/ngày (100%)
- **Chỉ áp dụng Cascade Routing:** Chi phí giảm còn **$11.48/ngày** (Giảm **76.5%**, tiết kiệm $37.39/ngày).
- **Áp dụng Cascade + Prompt Caching:** Chi phí giảm còn **$10.28/ngày** (Giảm thêm $1.20/ngày, lũy kế 79.0%).
- **Áp dụng Full Stack (Cascade + Caching + Batch API):** Chi phí giảm còn **$8.48/ngày** (Giảm thêm $1.80/ngày, lũy kế **82.6%**).

#### Trả lời các câu hỏi phân tích (Guide.md mục 5.4):
1. **Trong 3 đòn bẩy, đòn bẩy nào đóng góp savings lớn nhất? Tại sao?**  
   - **Cascade Routing đóng góp lớn nhất (chiếm tới 92.6% tổng số tiền tiết kiệm được của inference).**  
   - Lý do: Chênh lệch giá giữa model nhỏ ($0.20 input / $0.40 output trên 1M token) và model lớn ($3.00 input / $15.00 output trên 1M token) lên tới **15× – 37.5×**. Khi phần lớn các truy vấn thông thường (tra cứu, phân loại, tóm tắt cơ bản) được định tuyến về model nhỏ, chi phí nền tảng lập tức giảm hơn 76%.

2. **Tại sao `discount_stack(batch=True, cache_hit_frac=1.0) = 0.05`?**  
   - Các mức chiết khấu áp dụng theo **nguyên lý nhân dồn (multiplicative stacking)**:  
     * Prompt Caching chiết khấu 90% (chỉ thanh toán $10\% = 0.10$ đơn giá input đã cache).  
     * Batch API chiết khấu 50% ($0.50$ đơn giá).  
     * Tích lũy: $0.10 \times 0.50 = 0.050$ (nghĩa là chỉ trả **5%** chi phí ban đầu, tương đương **giảm 95%**).

3. **Khi nào KHÔNG NÊN dùng Batch API?**  
   - Không được dùng cho các tác vụ đòi hỏi tương tác thời gian thực (real-time chatbot, autocomplete, voice agent) nơi người dùng yêu cầu Time-to-First-Token (TTFT) dưới vài trăm mili-giây.  
   - Batch API thường có SLA hoàn thành trong vòng 12–24 giờ, nên chỉ thích hợp cho đánh giá benchmark offline, tổng hợp báo cáo đêm, embedding tài liệu hàng loạt và crawl dữ liệu định kỳ.

---

### 2.3. Mission 3: Chiến lược mua sắm GPU (Purchasing Strategy)

#### Kết quả thực nghiệm M3:
- Ngưỡng hòa vốn cam kết: `break-even utilization = 1 - 0.45 = 55%` (tương đương 13.2 giờ/ngày).
- Chi phí hạ tầng On-demand: **$25,667/tháng**.
- Chi phí hạ tầng sau phân bổ Tier tối ưu: **$15,627/tháng** (**tiết kiệm $10,040/tháng, tức 39.1%**).

#### Bảng phân bổ Tier cho 8 Job:
| Job ID | GPU Type | Số GPU | Giờ/ngày | Tính chất | Tier đề xuất | Chi phí On-demand | Chi phí Tối ưu | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| `job-train-llm` | H100 | 8 | 20 | Interruptible | **Spot** | $12,000 | $7,596 | Tiết kiệm $4,404/tháng nhờ spot checkpoint |
| `job-train-embed` | A100 | 4 | 10 | Interruptible | **Spot** | $2,148 | $1,393 | Chạy gián đoạn theo batch |
| `job-finetune` | H100 | 2 | 6 | Interruptible | **Spot** | $900 | $570 | Chạy ngắt quãng |
| `job-infer-chat` | A10G | 6 | 24 | Non-interruptible | **Reserved** | $4,320 | $2,592 | Duty cycle 100% $\ge$ 55% |
| `job-infer-rag` | A100 | 3 | 24 | Non-interruptible | **Reserved** | $3,866 | $2,160 | Duty cycle 100% $\ge$ 55% |
| `job-infer-search` | L4 | 4 | 24 | Non-interruptible | **Reserved** | $1,728 | $972 | Duty cycle 100% $\ge$ 55% |
| `job-dev-sandbox` | A10G | 2 | 8 | Interruptible | **Spot** | $480 | $203 | Dev môi trường thử nghiệm |
| `job-batch-eval` | H100 | 1 | 3 | Interruptible | **Spot** | $225 | $142 | Chạy vài giờ mỗi đêm |

#### Trả lời các câu hỏi phân tích (Guide.md mục 6.4):
1. **Job nào được đề xuất dùng Spot? Tại sao?**  
   - Các job: `job-train-llm`, `job-train-embed`, `job-finetune`, `job-dev-sandbox`, `job-batch-eval`.  
   - Lý do: Các job này có cờ `interruptible = 1` và không cần chạy liên tục 24/7 khẩn cấp. Mô hình đào tạo/tinh chỉnh hỗ trợ lưu checkpoint định kỳ, cho phép chịu được việc bị thu hồi instance mà vẫn khôi phục được trạng thái tính toán. Đổi lại, đơn giá spot giảm từ 40% đến 60% so với on-demand.

2. **Với Spot, tại sao "Effective Hours" lại cao hơn "Job Hours" thực tế?**  
   - Trong mô hình `spot_checkpoint_cost`:  
     $$\text{Effective Hours} = \text{Job Hours} \times (1 + \text{Overhead Checkpoint}) + \text{Thời gian Rework do gián đoạn}$$
     * **Checkpoint Overhead (3%):** GPU tốn chu kỳ I/O ghi trọng số model ra ổ cứng phân tán thay vì chỉ tính toán.  
     * **Rework Time:** Khi bị nhà cung cấp thu hồi máy chủ, tiến trình phải tính toán lại phần công việc tính từ checkpoint gần nhất (trung bình mất 0.5 giờ mỗi lần gián đoạn).  
   - Mặc dù tổng thời gian GPU tăng từ 720h lên 759.6h, mức giảm giá cực sâu của Spot vẫn mang lại khoản tiết kiệm thực tế ròng hơn 40%.

3. **Job nào nên xem xét lại giữa Reserved và On-demand?**  
   - Các job inference chạy liên tục 24/7 (`job-infer-chat`, `job-infer-rag`, `job-infer-search`) có duty cycle = 100%, vượt trội so với điểm hòa vốn 55% $\rightarrow$ **Bắt buộc dùng Reserved 1–3 năm**.  
   - Tuy nhiên, nếu một dự án mới đang trong giai đoạn thử nghiệm thị trường và dự kiến có thể bị hủy trong 2–3 tháng tới, việc cam kết Reserved 3 năm sẽ gây lãng phí tồn đọng (commitment lock-in waste). Khi đó, nên chọn Savings Plans linh hoạt hoặc On-demand.

---

### 2.4. Mission 4: Phân bổ chi phí (Cost Allocation) & Chuẩn FOCUS

#### Kết quả thực nghiệm M4:
- Tỷ lệ gắn thẻ hợp lệ (**Tag Coverage**): **92.0%** (vượt ngưỡng kiểm soát 80%).
- Trạng thái **Chargeback Ready**: **True** (đã sẵn sàng tính phí nội bộ).
- File xuất dữ liệu chuẩn **FOCUS 1.x**: `outputs/focus_export.csv` (chứa 50 bản ghi mẫu).

#### Phân bổ chi phí theo Team:
| Team | Chi phí ($/ngày) | Tỷ lệ (%) | Dự án chính |
|---|---|---|---|
| **assistant** | $2.59 | 30.6% | `chat` |
| **search** | $2.49 | 29.4% | `web-search` |
| **eval** | $1.79 | 21.1% | `nightly-eval` |
| **rag** | $1.60 | 18.9% | `doc-qa` |
| **Tổng cộng** | **$8.47 / ngày** | **100.0%** | |

#### Trả lời các câu hỏi phân tích (Guide.md mục 7.4):
1. **Team nào tốn nhiều chi phí nhất? Tỷ lệ là bao nhiêu?**  
   - Đội ngũ **assistant** tốn nhiều nhất ($2.59/ngày, chiếm 30.6%), bám sát ngay sau là **search** ($2.49/ngày, chiếm 29.4%). Hai đội này chiếm hơn 60% tổng lượng tiêu thụ inference hàng ngày.

2. **Tag Coverage là bao nhiêu %? Có đủ điều kiện để thực hiện Chargeback không?**  
   - Tag coverage đạt **92.0%** trên cả 2 chiều `team` và `project`.  
   - Theo thang trưởng thành FinOps (*Visibility $\rightarrow$ Showback $\rightarrow$ Chargeback*), ngưỡng an toàn để chuyển từ "thông báo chi phí" (Showback) sang "trừ tiền thực tế vào ngân sách bộ phận" (Chargeback) là $\ge 80\%$. Với mức 92%, NimbusAI hoàn toàn đủ điều kiện kích hoạt Chargeback mà không lo kiện cáo vì dữ liệu nhiễu.

3. **Tại sao chuẩn FOCUS (FinOps Open Cost and Usage Specification) lại quan trọng trong môi trường Multi-Cloud?**  
   - Mỗi nhà cung cấp dịch vụ đám mây (AWS, GCP, Azure, CoreWeave, Lambda Labs) có cấu trúc hóa đơn và tên gọi cột hoàn toàn khác nhau (ví dụ: `UnblendedCost`, `LineItemType`, `PricingPlan`).  
   - FOCUS chuẩn hóa tên cột (`BilledCost`, `ServiceCategory`, `ChargePeriodStart`, `ResourceId`) thành một lược đồ thống nhất, cho phép NimbusAI tổng hợp hóa đơn GPU từ nhiều nhà cung cấp vào một dashboard duy nhất mà không phải viết lại parser ETL riêng cho từng bên.

---

### 2.5. Mission 5: Báo cáo tối ưu hóa tổng thể & Tính bền vững

- Báo cáo chính thức được lưu tại `outputs/report.md` kèm biểu đồ phân bổ tiết kiệm `outputs/savings.png`.
- Tỷ trọng 4 đòn bẩy tiết kiệm hàng tháng:
  * **Purchasing (Spot/Reserved):** $10,040 (80.3%)
  * **Inference Optimization:** $1,212 (9.7%)
  * **Right-sizing Util-Lies:** $655 (5.2%)
  * **Kill Idle GPUs:** $600 (4.8%)
- Các chỉ số phát thải xanh:
  * Năng lượng trung bình mỗi query: **0.24 Wh**
  * Phát thải carbon trung bình mỗi query: **0.091 gCO2e**
  * Vùng sạch nhất và rẻ nhất: **`europe-north1`**

---

## 3. PHẦN MỞ RỘNG "YOUR TURN" (THỰC HIỆN ĐỦ 5 EXTENSIONS)

Để hoàn thiện toàn diện năng lực FinOps theo chuẩn của Rubric (Mục D), toàn bộ 5 phần mở rộng đã được nghiên cứu và triển khai thành công vào mã nguồn:

### 3.1. Extension 1: Nâng cấp hàm `recommend_tier()` thông minh (`finops/pricing.py`)
- **Vấn đề của thuật toán cũ:** Chỉ dựa đơn thuần vào `hours_per_day >= 13.2h` để khuyến nghị Reserved 3 năm, dễ dẫn đến rủi ro kẹt cam kết nếu dự án ngắn hạn, hoặc chọn Spot cho các dòng GPU có độ sẵn sàng thấp.
- **Cải tiến triển khai:**
  1. Thêm tham số `job_days`: Với các dự án ngắn hạn dưới 14 ngày (như sprint thử nghiệm hoặc dev sandbox ngắn), hàm từ chối cam kết dài hạn và giữ lại tier `on_demand`.
  2. Tích hợp bảng rủi ro thu hồi theo dòng GPU (`GPU_INTERRUPT_RISK`): Đảm bảo các GPU cao cấp như H100/H200 ưu tiên cấu hình checkpoint tối ưu trước khi nhận diện Spot.
  3. Duy trì độ tương thích ngược 100% với toàn bộ 15 bài unit test ban đầu.

### 3.2. Extension 2: Phân tích Right-sizing theo MBU & $/GB-VRAM (`missions/m1_efficiency_audit.py`)
- **Phương pháp tiếp cận:** Với các workload inference decode bị memory-bound, sức mạnh FLOPs không phải là nút thắt mà là dung lượng VRAM và băng thông HBM.
- **Kết quả đo lường từ Catalog:**
  | GPU Type | On-demand ($/hr) | HBM (GB) | Peak BW (TB/s) | Chi phí $/GB-VRAM | Chi phí $/TB/s BW |
  |---|---|---|---|---|---|
  | **MI300X** | $1.95 | 192 | 5.30 | **$0.0102** | **$0.3679** |
  | **A100** | $1.79 | 80 | 2.00 | **$0.0224** | **$0.8950** |
  | **B200** | $5.09 | 192 | 8.00 | **$0.0265** | **$0.6362** |
  | **H200** | $3.95 | 141 | 4.80 | **$0.0280** | **$0.8229** |
  | **H100** | $2.50 | 80 | 3.35 | **$0.0312** | **$0.7463** |
  | **L4** | $0.80 | 24 | 0.30 | **$0.0333** | **$2.6667** |
  | **A10G** | $1.00 | 24 | 0.60 | **$0.0417** | **$1.6667** |
- **Đề xuất thực thi:**
  * `gpu-h100-4` (MBU 0.207) và `gpu-h100-5` (MBU 0.271) chuyển sang **A100** $\rightarrow$ Tiết kiệm **$852.00/tháng** (giảm 28.4%).
  * `gpu-a100-0` (MBU 0.276) và `gpu-a100-1` (MBU 0.247) chuyển sang **A10G** $\rightarrow$ Tiết kiệm **$1,137.60/tháng** (giảm 44.1%).
  * **Tổng tiềm năng tiết kiệm Right-sizing theo MBU:** **$1,989.60/tháng**.

### 3.3. Extension 3: Xác thực hiệu quả Caching qua hàm `cache_is_worth_it()` (`finops/pricing.py`)
- **Bản chất kinh tế:** Prompt caching không phải lúc nào cũng có lợi; các nhà cung cấp đám mây thu phí ghi bộ nhớ cache (cache write) và phí lưu trữ theo giờ. Caching chỉ có lãi khi tổng số tiền tiết kiệm từ các lượt đọc lặp lại bù đắp được chi phí ghi ban đầu.
- **Công thức điểm hòa vốn:**
  $$\text{Break-even Reads} = \frac{\text{Write Cost per 1M}}{\text{Base Price} \times (1 - \text{Read Discount})} = \frac{1.0}{1.0 \times 0.90} \approx \mathbf{1.11 \text{ lần đọc}}$$
- **Đo lường trên dữ liệu NimbusAI:** Với tỷ lệ tái sử dụng trung bình đạt ~2.5 lần đọc trên mỗi prefix trong tập log, hàm `cache_is_worth_it()` trả về `True` cho cả hai tier Small và Large, xác nhận chính sách Prompt Caching mang lại lợi ích tài chính thực tế ròng.

### 3.4. Extension 4: Ngân sách Reasoning & Cắt giảm Năng lượng (`missions/m2_inference_levers.py`)
- **Phát hiện chấn động:**
  | Loại truy vấn | Số Request | % Request | Số Token | % Token | Chi phí ($/ngày) | % Chi phí | Điện năng (Wh) | % Điện năng |
  |---|---|---|---|---|---|---|---|---|
  | **Standard** | 2,199 | 91.6% | 6,291,871 | 83.5% | $7.09 | 83.5% | 1,887.6 Wh | **6.0%** |
  | **Reasoning** | 201 | **8.4%** | 1,241,156 | 16.5% | $1.40 | 16.5% | 29,787.7 Wh | **94.0%** |
- **Insight then chốt:** Các truy vấn reasoning chỉ chiếm **8.4% tổng số request** nhưng lại thiêu đốt tới **94.0% tổng năng lượng điện inference** do hệ số nhân suy luận ~80×!
- **Đề xuất kiểm soát (Gated Routing):** Thiết lập ngưỡng phân loại (Task Complexity Router) — chỉ kích hoạt reasoning khi điểm tin cậy (confidence score) của model nhỏ dưới 0.70. Nếu giảm 50% lưu lượng reasoning không cần thiết, NimbusAI tiết kiệm được **14,894 Wh/ngày (~446.8 kWh/tháng)** và **$20.95/tháng** chi phí inference trực tiếp.

### 3.5. Extension 5: Carbon-Aware Scheduling cho các tác vụ Training (`missions/m3_purchasing.py`)
- **Mô hình hóa năng lượng:** 5 job training/batch có cờ `interruptible = 1` tiêu thụ tổng cộng **4,227.0 kWh điện mỗi tháng**.
- **So sánh 5 vùng điện toán toán cầu:**
  | Vùng đám mây | Giá điện ($/kWh) | Cường độ Carbon (gCO2/kWh) | Tiền điện ($/tháng) | Phát thải CO2 (kg/tháng) | % Giảm phát thải | Đánh giá |
  |---|---|---|---|---|---|---|
  | **europe-north1** (Na Uy) | $0.090 | **30** | $380.43 | **126.8** | **-92.1%** | **Sạch nhất thế giới (Thủy điện)** |
  | **us-east-wa** (Washington) | **$0.055** | 90 | **$232.49** | 380.4 | -76.3% | **Chi phí điện rẻ nhất** |
  | **us-west-2** (Oregon) | $0.070 | 120 | $295.89 | 507.2 | -68.4% | Cân bằng Bờ Tây |
  | **us-east-1** (Virginia - Gốc) | $0.120 | 380 | $507.24 | 1,606.3 | 0.0% | Vùng mặc định đắt và ô nhiễm |
  | **europe-central2** (Ba Lan) | $0.180 | 660 | $760.86 | 2,789.8 | +73.7% | Ô nhiễm nhất (Nhiệt điện than) |
- **Khuyến nghị điều phối:** Chuyển toàn bộ các job đào tạo batch sang `europe-north1` giúp NimbusAI cắt giảm ngay **1,479.5 kg CO2e/tháng (giảm 92.1%)**, hoặc sang `us-east-wa` để tiết kiệm **$274.75/tháng** tiền điện. Tác vụ training không nhạy cảm với độ trễ mạng (latency-insensitive) nên việc đặt cụm tính toán xa người dùng hoàn toàn không ảnh hưởng đến trải nghiệm sản phẩm.

---

## 4. BA HÀNH ĐỘNG HÀNG ĐẦU DÀNH CHO FINOPS LEAD TẠI NIMBUSAI

Nếu đảm nhận vai trò FinOps Lead tại NimbusAI, 3 hành động tiên quyết theo thứ tự ưu tiên ROI trong 30 ngày đầu tiên là:

1. **Khóa ngay cam kết Reserved Instances 3 năm cho các dịch vụ Inference 24/7 và chuyển Training sang Spot:**
   - *Hành động:* Ký hợp đồng cam kết cho 6x A10G, 3x A100 và 4x L4 đang chạy 24/7; cấu hình tự động lưu checkpoint mỗi 30 phút trên Kubernetes cho các job training để chạy trên Spot pool.
   - *Hiệu quả:* Đem lại ngay lập tức **$10,040/tháng tiền mặt tiết kiệm** (chiếm 80% tổng savings tiềm năng) với nỗ lực tái cấu trúc code bằng 0.

2. **Triển khai Gateway Cascade Routing và Gated Reasoning cho toàn bộ API LLM:**
   - *Hành động:* Cài đặt proxy định tuyến thông minh (như LiteLLM proxy) ở tầng API gateway. 90% truy vấn đơn giản tự động trỏ về model Small; kích hoạt Prompt Caching trên các system prompt dài; chặn việc gọi model reasoning tùy tiện trừ khi task vượt qua bộ lọc độ phức tạp.
   - *Hiệu quả:* Hạ giá thành đơn vị từ **$6.488 xuống $1.126 trên 1 triệu token (giảm 82.6%)**, đồng thời cắt giảm hơn 90% lượng điện tiêu thụ lãng phí của các truy vấn suy luận.

3. **Thiết lập rào cản tự động hóa kiểm soát hạ tầng (Guardrails & Automated Right-sizing):**
   - *Hành động:* Cài đặt kịch bản CloudWatch/Prometheus tự động tắt các GPU có `util < 10%` sau 30 phút không hoạt động; hạ cấp các máy chủ bị cảnh báo "GPU-Util Lie" (`gpu-h100-4`) xuống dòng GPU A100/A10G tương ứng.
   - *Hiệu quả:* Xóa sổ hoàn toàn **$1,255/tháng** thất thoát do GPU chạy không tải và cấp phát sai cấu hình, hoàn thiện quy chế gắn nhãn để kích hoạt Chargeback minh bạch 100%.

---
**Tài liệu đính kèm kiểm chứng:**
- Báo cáo tóm tắt: [`outputs/report.md`](file:///c:/Bo/VinAI/LAB/261013_LAB25/K4-L3-DAY25-BuiLeGiaHuy-2A202602607-GpuFinOps/outputs/report.md)
- Biểu đồ tiết kiệm: [`outputs/savings.png`](file:///c:/Bo/VinAI/LAB/261013_LAB25/K4-L3-DAY25-BuiLeGiaHuy-2A202602607-GpuFinOps/outputs/savings.png)
- Hóa đơn chuẩn FOCUS: [`outputs/focus_export.csv`](file:///c:/Bo/VinAI/LAB/261013_LAB25/K4-L3-DAY25-BuiLeGiaHuy-2A202602607-GpuFinOps/outputs/focus_export.csv)
