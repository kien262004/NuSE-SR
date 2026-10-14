# Kế hoạch nghiên cứu và thực nghiệm: Null-Space PCA cho Efficient Image Super-Resolution

**Phiên bản:** 1.0  
**Ngày:** 14/10/2026  
**Trạng thái:** Kế hoạch thực hiện; chưa phải kết quả thực nghiệm trên DIV2K  
**Phạm vi khởi đầu:** SR ×4, ảnh RGB, degradation **non-overlapping average pooling**, học PCA theo từng block và từng kênh; sau đó nghiên cứu khả năng dự đoán từ LR và hiệu quả edge inference.

> **Nguồn nền tảng:** `PCA_Null_Space_SR_Theory.docx` (các mệnh đề về phép chiếu, data consistency và phân rã lỗi), cùng mã thử nghiệm `rnd_nullspace_pca_sr.py` được xây dựng trong cuộc trao đổi. Những nội dung về *predictability-aware subspace*, giảm CNN architecture search và edge deployment dưới đây là **giả thuyết/đề xuất cần thực nghiệm**, không phải kết quả đã được chứng minh bởi tài liệu gốc.

---

## 1. Mục tiêu nghiên cứu

### 1.1. Câu hỏi chính

1. Với toán tử LR quan sát cố định \(A\), có thể tạo đúng phép chiếu \(P=I-A^\dagger A\) và học một PCA basis nhỏ \(U_k\subseteq\ker(A)\) từ **HR training data** không?
2. PCA giữ được bao nhiêu năng lượng của null-space residual với mỗi rank \(k\)? Sai số giới hạn thấp nhất (*oracle truncation*) là bao nhiêu?
3. Những PCA coefficients có **dự đoán được từ LR** hay không? PCA thông thường có giữ những hướng nhiều phương sai nhưng khó dự đoán không?
4. Giảm từ \(s^2-1\) null-space coefficients xuống \(k\) có cải thiện **latency, RAM, tổng tham số và PSNR–latency Pareto frontier** không?
5. Những thống kê null-space có giúp **cắt giảm không gian lựa chọn channel/kernel/depth** của RLFN, SPAN hoặc PLKSR trước full training không?

### 1.2. Mục tiêu có thể kiểm chứng

- Xác minh trên ảnh nhỏ và ảnh thật: \(AP\approx0\), \(AU_k\approx0\), \(A\hat x\approx y\), và đẳng thức phân rã lỗi.
- Thu được bảng **eigenvalue spectrum / explained variance / oracle PSNR** cho \(k\in\{1,2,4,8,12,15\}\) tại SR ×4.
- Đánh giá riêng **PCA truncation error** và **coefficient prediction error**; tuyệt đối không gộp chúng thành một kết luận về khả năng nén.
- So sánh **PCA vs DCT (bỏ DC) vs random orthonormal null basis vs full-rank null-space output**.
- Chỉ công bố lợi ích Efficient SR nếu có **measured latency và peak RAM**, không suy ra tốc độ từ số coefficients hoặc FLOPs đơn thuần.

### 1.3. Không nằm trong phạm vi của pha đầu

- Không áp dụng ngay cho bicubic/real-world degradation rồi tuyên bố exact data consistency.
- Không coi PCA rank \(k\) là số channels tối thiểu của CNN; mối liên hệ đó cần được kiểm nghiệm riêng.
- Không coi low-dimensional PCA null space là ý tưởng hoàn toàn mới; phải đối chiếu các hướng PCA-SR, null-space GAN và learned null-space projection đã có.

---

## 2. Mô hình toán học và các giả thiết

### 2.1. Forward operator

Với một channel và block HR không chồng lấn có kích thước \(s\times s\), đặt \(q=s^2\):

\[
 x_b\in\mathbb R^q,\qquad y_b=A_bx_b,\qquad
 A_b=\frac1q\mathbf 1^\top\in\mathbb R^{1\times q}.
\]

Pseudoinverse Moore–Penrose là:

\[
 A_b^\dagger=\mathbf1\in\mathbb R^{q\times1}.
\]

Điều này tương ứng:

- \(A\): `avg_pool2d(kernel_size=s, stride=s)`.
- \(A^\dagger\): `repeat_interleave(s)` trên mỗi trục không gian.
- Cần \(H_{HR},W_{HR}\) chia hết cho \(s\); RGB xử lý riêng từng channel.

### 2.2. Tính projector \(P\)

\[
 P_b=I_q-A_b^\dagger A_b=I_q-\frac1q\mathbf1\mathbf1^\top.
\]

Với mọi block:

\[
 z_b=P_bx_b=x_b-\operatorname{mean}(x_b)\mathbf1.
\]

Các tính chất bắt buộc kiểm tra:

\[
 P_b^\top=P_b,\quad P_b^2=P_b,\quad A_bP_b=0,
 \quad\operatorname{rank}(P_b)=q-1.
\]

Tại **×4**, \(q=16\), null-space có **15 chiều trên mỗi block, mỗi channel**. Với toàn ảnh, \(P\) là phép chiếu blockwise; **không cần dựng ma trận lớn**.

> **Quan trọng:** \(P\) được xác định **chỉ bởi \(A\)**. Dataset **không được dùng để tính \(P\)**; dataset chỉ cung cấp mẫu \(z_i=P x_i\) để ước lượng mean/covariance và PCA.

### 2.3. Tính mean và covariance từ HR training blocks

Với \(T\) block residuals cùng một kênh:

\[
 z_i=P_bx_i\in\mathbb R^q,\qquad
 \mu=\frac1T\sum_{i=1}^{T}z_i.
\]

\[
 C=\frac1T\sum_{i=1}^{T}(z_i-\mu)(z_i-\mu)^\top
   =\frac1T\sum_i z_iz_i^\top-\mu\mu^\top.
\]

Cần ước lượng **riêng từng kênh RGB** ở baseline: \(C\in\mathbb R^{3\times16\times16}\). Một vector \(z_i\) có tổng phần tử bằng 0 nhưng \(\mu\) không nhất thiết là vector toàn 0; nó vẫn thỏa \(A_b\mu=0\).

**Streaming covariance** (không giữ toàn bộ block trong RAM):

```python
q = scale * scale
sum_z  = torch.zeros(3, q, dtype=torch.float64)
sum_zz = torch.zeros(3, q, q, dtype=torch.float64)
count = 0

for hr in train_hr_loader:             # [B,3,H,W], normalized [0,1]
    blocks = to_block_patches(hr.double(), scale)
    flat = blocks.permute(1, 0, 3, 4, 2).reshape(3, -1, q)
    z = flat - flat.mean(dim=-1, keepdim=True)
    sum_z += z.sum(dim=1)
    sum_zz += torch.einsum('cni,cnj->cij', z, z)
    count += z.shape[1]

mu = sum_z / count
cov = sum_zz / count - torch.einsum('ci,cj->cij', mu, mu)
```

Sau đó thực hiện chiếu lại \(C\leftarrow P_bCP_b\) và đối xứng hóa \((C+C^\top)/2\) để ổn định số học; phép chiếu này **không thay đổi covariance lý tưởng**.

### 2.4. PCA basis và giới hạn rank

Giải:

\[
 Cu_j=\lambda_j u_j,\qquad
 \lambda_1\ge\dots\ge\lambda_q\ge0.
\]

Chọn \(U_k=[u_1,\ldots,u_k]\) ứng với \(k\) eigenvalues dương lớn nhất. Khi đó:

\[
 A_bU_k=0,\qquad U_k^\top U_k=I_k.
\]

Với \(s=4\): \(1\le k\le15\), miễn có đủ eigenvectors hợp lệ. Thêm vào đó, sample covariance có:

\[
 \operatorname{rank}(C)\le\min(T-1,q-1).
\]

Phải kiểm tra eigenvalue spectrum trên **validation data** để tránh kết luận sai về intrinsic dimension chỉ từ tập train.

---

## 3. Mô hình reconstruction và các bảo đảm

### 3.1. Decoder PCA

CNN nhận LR \(y\), sinh \(3k\) coefficient maps \(c_\theta(y)\) ở độ phân giải LR (tại mỗi vị trí LR có \(k\) coefficients cho mỗi channel RGB). Với từng block:

\[
 \boxed{\hat x_b=A_b^\dagger y_b+\mu+U_kc_\theta(y)_b.}
\]

Vì \(A_b\mu=A_bU_k=0\) và \(A_bA_b^\dagger=1\):

\[
 \boxed{A_b\hat x_b=y_b.}
\]

Đây là **exact data consistency về toán học** đối với non-overlapping average pooling, không nhiễu và matching forward model, kể cả khi predictor chưa được train. Trong FP32 chỉ kỳ vọng chính xác đến sai số số học.

### 3.2. Phân rã reconstruction error

Đặt:

\[
 z_b=P_bx_b,\qquad
 c_b^*=U_k^\top(z_b-\mu),\qquad
 r_{\perp,b}=(I-U_kU_k^\top)(z_b-\mu).
\]

Với basis trực chuẩn:

\[
 \boxed{
 \|x_b-\hat x_b\|_2^2=
 \|r_{\perp,b}\|_2^2+
 \|c_b^*-c_\theta(y)_b\|_2^2.
 }
\]

- **Truncation error:** phần năng lượng bị bỏ vì \(k\) nhỏ, không thể được bù bởi CNN nếu giữ cố định decoder.
- **Prediction error:** phần do CNN dự đoán sai coefficients từ LR.
- Theo PCA, \(\mathbb E\|r_\perp\|^2=\sum_{j>k}\lambda_j\) trên phân phối dùng để tính covariance, không tự động đúng bằng giá trị validation/test.
- Đẳng thức áp dụng cho MSE/L2 bình phương; nếu dùng L1 hoặc clamp đầu ra, không được coi cùng công thức còn đúng nguyên vẹn.

### 3.3. Đánh giá approximation/numerical consistency

Nếu basis không còn hoàn toàn thuộc null space:

\[
 \|A\hat x-y\|_2
 \le\|A\mu\|_2+\|AU_k\|_2\|c_\theta(y)\|_2
\]

(khi \(AA^\dagger y=y\)). Kiểm tra cả `max_abs` lẫn relative error; không chỉ nhìn PSNR.

### 3.4. Giới hạn phát biểu

- Data consistency **không** chứng minh \(\hat x=x_{GT}\).
- Việc fit PCA trên HR **training** là supervised learning hợp lệ, **không được dùng HR validation/test** để fit basis, chọn threshold hoặc tính covariance.
- Việc dự đoán \(k\) coefficients thay cho \(s^2\) pixel không tự động giảm FLOPs của backbone; decoder synthesis và memory access vẫn có chi phí.

---

## 4. Thiết kế dữ liệu và quy trình chống leakage

### 4.1. Dataset và cách tạo LR

- **Train:** DIV2K train HR (800 ảnh); **validation:** DIV2K val (100 ảnh), chỉ để đánh giá và chọn hyperparameters.
- Scale đầu tiên: **×4**, RGB 8-bit chuyển `float32 [0,1]`.
- Dùng HR crops \(256\times256\), bảo đảm chia hết cho 4; \(y=A(x)\) bằng `avg_pool2d`.
- Giữ đúng sự căn chỉnh HR↔LR, tránh resize thêm trước hoặc sau `A`.
- Augmentation: flip ngang/dọc, rotate 90°; dùng cho train, nhưng chỉ tính covariance trên training distribution đã được quy định rõ. Báo cáo có/không dùng augmentation khi PCA fit.
- Chia seed cố định, lưu danh sách tên ảnh, crop sampling và cấu hình preprocessing.

> **Không sử dụng** cặp LR bicubic có sẵn làm ground truth cho data consistency average pooling. Nếu muốn so sánh các nghiên cứu SR ×4 bicubic, đó phải là một protocol **riêng**, với \(A\) hoặc phát biểu consistency tương ứng.

### 4.2. Hai tập dữ liệu cho hai phép kiểm định khác nhau

- **PCA fitting set:** chỉ HR train → \(P\) → residual blocks → \(\mu,C,U_k\).
- **Coefficient prediction set:** train CNN với pairs \((y,c^*)\) hoặc \((y,x)\) từ train HR.
- **Validation/test:** basis cố định; chỉ chạy encoding oracle trên HR validation để **đánh giá upper bound** (không dùng trong inference).

### 4.3. Tracking dữ liệu

Lưu `preprocess_config.yaml`, random seeds, split manifest, scale, crop size, RGB range, A definition và hash checkpoint/basis. Đây là phần bắt buộc để kiểm tra lại experiment.

---

## 5. Thí nghiệm theo các pha và tiêu chí hoàn thành

### Pha 0 — Mathematical & numerical sanity checks (2–3 ngày)

**Dữ liệu:** vectors ngẫu nhiên, ảnh nhỏ 8×8/16×16 và synthetic patches.

**Việc làm:**

1. Tính \(A_b\), \(A_b^\dagger\), \(P_b\) **tường minh** cho scale ×2; xác minh \(P_b=P_b^\top=P_b^2\), \(A_bP_b=0\).
2. So sánh kết quả tường minh với `x - Ap(A(x))` trên mọi channels.
3. Tính PCA trên synthetic HR patches và kiểm tra \(AU_k\), \(U_k^\top U_k\).
4. Sinh hệ số CNN ngẫu nhiên, giải mã, kiểm tra \(A\hat x-y\).
5. Xác minh **từng ảnh** có `total_squared_error ≈ truncation_squared_error + coefficient_squared_error`.

**Chỉ số / ngưỡng đề xuất:**

- FP64: `max|A(Px)| < 1e-10` trên test nhỏ hợp điều kiện.
- FP32: `max|A(x_hat)-y| < 1e-5` với tensor `[0,1]` trên test hiện tại.
- Relative difference của đẳng thức lỗi `< 1e-4` (có epsilon tránh chia 0).

**Deliverable:** `tests/test_operator.py`, `tests/test_pca.py`, `results/sanity_check.json`.

**Go/No-go:** Nếu phép chiếu và data consistency sai thì **không chuyển pha**, kiểm tra pixel ordering, `pixel_unshuffle/pixel_shuffle`, basis orientation và dtype.

### Pha 1 — PCA spectrum và oracle upper bound (tuần 1–2)

**Việc làm:**

1. Thu null-space patches từ DIV2K train bằng streaming moments FP64.
2. Tính \(\mu_c,C_c,\lambda_{c,j},U_{k,c}\) riêng theo RGB channels.
3. Quét \(k\in\{1,2,4,8,12,15\}\).
4. Trên validation: đo `explained_variance(k)`, `truncation_MSE(k)`, `oracle_PSNR(k)`.
5. So sánh basis giữa những subset train khác nhau để đánh giá độ ổn định.
6. Kiểm tra eigenvectors có tổng phần tử gần 0; covariance có eigenvalue gần 0 ứng với constant direction.

**Công thức:**

\[
 \operatorname{EVR}(k)=\frac{\sum_{j\le k}\lambda_j}{\sum_{j}\lambda_j},
 \qquad E_{\mathrm{trunc}}(k)=\sum_{j>k}\lambda_j.
\]

**Deliverable:** `basis/avgpool_x4_rgb.pt`, `results/pca_spectrum.csv`, đồ thị eigenvalue/EVR và oracle PSNR.

**Go/No-go:** Nếu \(k\ll15\) gây oracle PSNR thấp, không nên đầu tư CNN lớn để dự đoán basis rank đó. Nếu kết quả rất nhạy với sampling, cần tăng sample diversity và xác định lại thiết kế fit PCA.

### Pha 2 — Predictability của PCA coefficients (tuần 3–4)

**Việc làm:**

1. Dùng một CNN LR-domain nhỏ dự đoán coefficient maps \(c_\theta(y)\).
2. Tại mỗi \(k\), dùng **cùng backbone width, training steps, optimizer, crops, augmentations và seeds**.
3. Trước tiên tối ưu MSE để giữ liên hệ với phân rã sai số; có thể train trực tiếp `MSE(coeff_pred, coeff_oracle)`.
4. Đo đồng thời `coefficient_MSE`, `truncation_MSE`, `output_MSE`, RGB PSNR, `A(x_hat)-y`.
5. Vẽ ba đường theo \(k\): oracle PSNR, actual PSNR, coefficient prediction MSE.
6. Kiểm tra kết quả bằng ít nhất 3 seeds cho các cấu hình khả quan.

**Baselines:** `k=15` (full local null-space), `k=8`, `k=4`, và trực tiếp dự đoán HR correction rồi chiếu bằng \(P\) (*full-output projected residual*).

**Go/No-go:** Nếu oracle PSNR tăng khi \(k\) tăng nhưng actual PSNR không tăng, đây là bằng chứng **prediction bottleneck**; không suy luận rằng dữ liệu chỉ có intrinsic rank nhỏ.

### Pha 3 — So sánh PCA với các basis khác (tuần 5)

Tạo cùng số lượng coefficients \(k\) và cùng predictor để so sánh:

| Basis | Xây dựng | Mục đích đối chứng |
|---|---|---|
| PCA on \(Px\) | Top eigenvectors của covariance HR null residual | Baseline đề xuất |
| DCT, bỏ DC | DCT zero-mean basis, chọn \(k\) hướng | Prior tần số có cấu trúc và có thể tổng hợp hiệu quả |
| Random null orthobasis | Sample basis rồi trực chuẩn hóa trong \(\ker(A_b)\) | Kiểm tra lợi ích đặc thù của PCA |
| Full null basis (\(k=15\)) | Cơ sở trực chuẩn bất kỳ của \(\ker(A_b)\) | Không có truncation error |
| Pixel residual + \(P\) | CNN sinh 16 giá trị/block, project xuống null space | Không ép low-rank decoder |

**Go/No-go:** Nếu PCA không vượt DCT rõ ràng về PSNR–latency hoặc rank thấp, không nên gắn novelty với PCA riêng lẻ; có thể nghiên cứu **predictability-guided** basis.

### Pha 4 — Predictability-aware subspace (tuần 6–7; nghiên cứu mở rộng)

**Giả thuyết:** PCA trên \(Z=PX\) giữ những hướng có variance lớn; điều ta thực sự cần là những hướng **có thể dự đoán từ LR**.

Đặt \(m(y)=\mathbb E[Z-\mu\mid Y=y]\). Với reconstruction tối ưu theo MSE và basis rank \(k\), subspace lý tưởng được xác định từ:

\[
 C_{\mathrm{predictable}}=\mathbb E[m(Y)m(Y)^\top].
\]

**Thực hiện:**

1. Train teacher dự đoán full null-space residual từ LR bằng **train split**.
2. Project teacher prediction qua \(P\), trừ training mean và tính covariance của predictions.
3. PCA trên covariance **teacher-predicted residuals**, tạo basis \(U_k^{pred}\).
4. So sánh với HR null-PCA trên same \(k\), same student backbone và same budget.
5. Kiểm tra teacher fit/basis selection trên split held-out trong train để giảm nguy cơ overfit.
6. Báo cáo kết quả teacher approximation; không gọi đây là true conditional mean nếu chưa chứng minh.

**Go/No-go:** Chỉ giữ hướng này nếu tăng actual PSNR hoặc giảm số \(k\) cần thiết ở cùng PSNR mà không tăng chi phí deploy. Teacher pretraining/search cost phải được tính vào chi phí nghiên cứu.

### Pha 5 — Efficient CNN và edge benchmark (tuần 8+)

**Mục tiêu:** Biết liệu PCA-rank reduction thực sự giảm runtime, hay chỉ giảm output channels.

**Việc làm:**

1. Giữ `A`, dataset và losses không đổi; thay backbone hệ số bằng RLFN, SPAN hoặc PLKSR-tiny đã chỉnh head.
2. Quét \(k\in\{4,8,12,15\}\) cùng các width \(\in\{16,24,32,48,64\}\); ưu tiên candidate đã qua screening.
3. Ghi MACs, params, model size, peak activation memory, peak RAM.
4. Đo measured latency trên thiết bị cố định, batch 1, FP32, theo hai kịch bản: **CNN-only** và **CNN + PCA decode + output**.
5. Dùng ít nhất 50 warmup và 200 measured iterations ×5 runs; báo cáo median, p95 và độ lệch.
6. Với SPAN/PlainUSR, thực hiện **deploy fusion một lần trước benchmark**, không fuse mỗi `forward()`.
7. Vẽ PSNR–latency và PSNR–RAM Pareto frontiers; báo cáo chi phí fitting PCA và train riêng với runtime deploy.

**So sánh công bằng:** Baseline RLFN/SPAN/PLKSR phải được train trên **cùng LR = average pooling(HR)**. Không so sánh trực tiếp PSNR với benchmark công bố dùng bicubic degradation.

**Go/No-go:** Nếu PCA decoding hoặc buffer memory triệt tiêu lợi ích từ output head nhỏ hơn, chọn PCA làm **công cụ phân tích/regularizer offline** thay vì một inference decoder bắt buộc.

---

## 6. Bảng thí nghiệm tối thiểu

| ID | Nhóm | Cấu hình | Metric quyết định | Ưu tiên |
|---|---|---|---|---|
| E00 | Sanity | ×2 toy matrix | \(P^2-P\), \(AP\) | P0 |
| E01 | Sanity | ×4 random HR | \(A\hat x-y\) | P0 |
| E02 | Sanity | ×4 oracle coefficient | Decomposition equality | P0 |
| E03 | Spectrum | PCA RGB train | Eigenvalues, effective rank | P0 |
| E04 | Rank | \(k=1,2,4,8,12,15\) | Oracle MSE/PSNR | P0 |
| E05 | Prediction | Small CNN each \(k\) | Actual PSNR, coefficient MSE | P0 |
| E06 | Basis | PCA vs DCT vs random | PSNR–latency | P0 |
| E07 | Full-rank | \(k=15\) / full residual+\(P\) | Prediction error | P0 |
| E08 | Statistical stability | Different train subsets | Basis angle, val oracle MSE | P1 |
| E09 | Predictability | Teacher-derived basis | Actual PSNR for same \(k\) | P1 |
| E10 | Width allocation | \(k\) × width sweep | PSNR–latency–RAM | P1 |
| E11 | Backbones | RLFN/SPAN/PLKSR-tiny | Edge latency, peak memory | P1 |
| E12 | Degradation mismatch | AvgPool basis vs bicubic LR | DC residual, PSNR drop | P2 |

**Đối chứng cốt lõi:** E06–E07 phải hiện diện; nếu không sẽ không biết cải thiện đến từ PCA, null-space constraint hay đơn thuần do mô hình head nhỏ.

---

## 7. Chỉ số và cách ghi kết quả

### 7.1. Chất lượng ảnh

- `RGB_PSNR` trên `[0,1]`, báo cáo rõ full-image hay crop/shave (tất cả thí nghiệm phải dùng một cách thống nhất).
- `RGB_SSIM` (tùy chọn; ghi implementation).
- `oracle_PSNR(k)`: decoder nhận hệ số \(c^*\) từ **HR ground truth**, chỉ để phân tích, **không phải mô hình inference**.
- `actual_PSNR(k)`: CNN chỉ nhận LR.
- `truncation_MSE(k)` và `coefficient_prediction_MSE(k)`.

### 7.2. Độ đúng toán học

\[
 e_P=\|P^2-P\|_F,\quad
 e_{AP}=\|AP\|_F,\quad
 e_U=\|AU_k\|_F,
\]

\[
 e_{DC}=\|A\hat x-y\|_\infty,
\quad
 e_{decomp}=\left|\|x-\hat x\|_2^2-E_{trunc}-E_{pred}\right|.
\]

Trên ảnh lớn kiểm tra bằng matrix-free implementation và normalize theo số phần tử khi cần. `P` tường minh chỉ dùng trên toy size.

### 7.3. Hiệu năng

- `params`, `MACs`, `checkpoint_MB`, `PCA_basis_MB`.
- `CNN_only_latency_ms`, `decode_latency_ms`, `E2E_latency_ms`, `latency_p95_ms`.
- `peak_RAM_MB`, backend, CPU/GPU, precision, input shape, batch size, warmup/iterations.
- `search_GPU_hours` cho giai đoạn search nếu nghiên cứu NAS/pruning.

**CSV tối thiểu:**

```text
experiment_id,seed,scale,degradation,basis,rank,cnn_width,
val_rgb_psnr,oracle_rgb_psnr,truncation_mse,coefficient_mse,
dc_max_abs,params,macs,latency_e2e_ms,latency_p95_ms,
peak_ram_mb,backend,device,notes
```

---

## 8. Cấu trúc repository đề xuất

```text
nullspace-pca-sr/
├── README.md
├── docs/
│   ├── math_derivation.md
│   └── experiment_plan.md
├── configs/
│   ├── data_avgpool_x4.yaml
│   ├── pca_x4.yaml
│   └── train_coeff_x4.yaml
├── src/
│   ├── operators.py             # A, A_dagger, P
│   ├── patches.py               # block extraction & pixel shuffle
│   ├── pca_fit.py               # streaming mean/cov/eigensolver
│   ├── pca_decode.py            # U_k coefficient synthesis
│   ├── train.py
│   ├── eval.py
│   ├── metrics.py
│   └── benchmark_edge.py
├── models/
│   ├── small_cnn.py
│   ├── rlfn_coeff.py
│   ├── span_coeff.py
│   └── plksr_coeff.py
├── tests/
│   ├── test_operator.py
│   ├── test_pca.py
│   └── test_error_decomposition.py
├── basis/
├── results/
├── figures/
└── scripts/
    ├── fit_pca.sh
    ├── train_rank_sweep.sh
    └── benchmark.sh
```

**Mã khởi đầu đã có:** `rnd_nullspace_pca_sr.py`, bao gồm `MeanSRLinearOperator`, `BlockNullSpacePCA`, `NullSpacePCASR`, và `demonstration()`.

Chạy sanity demo (môi trường đã cài PyTorch):

```bash
python rnd_nullspace_pca_sr.py
```

Chỉ số demo trên synthetic data **không** thay cho thí nghiệm DIV2K.

---

## 9. Timeline đề xuất 8 tuần

| Tuần | Hoạt động | Deliverable / tiêu chí qua pha |
|---|---|---|
| 1 | Operator, projector, toy proofs, unit tests | Kiểm tra đạt \(AP\approx0\), \(A\hat x\approx y\) |
| 2 | Fit covariance/PCA trên DIV2K train | Spectrum, basis lưu được, oracle curves |
| 3 | Train coefficient CNN cho các rank \(k\) | Actual PSNR vs oracle PSNR |
| 4 | Multi-seed, error decomposition, kiểm tra generalization | Báo cáo prediction bottleneck |
| 5 | PCA/DCT/random/full-null baselines | So sánh công bằng trên cùng backbone |
| 6 | Predictability-guided basis prototype | Kiểm tra giả thuyết so với PCA |
| 7 | Width sweep và đổi sang RLFN/SPAN/PLKSR | Candidate PSNR–MAC Pareto |
| 8 | Benchmark latency/RAM trên edge + báo cáo | PSNR–latency–RAM Pareto, quyết định hướng paper |

Nếu nguồn lực hạn chế, ưu tiên hoàn thành **tuần 1–5** trước; predictive basis và architecture search chỉ nên mở khi đã có baseline mạnh, có thể tái lập.

---

## 10. Tiêu chí kết luận và điều kiện dừng nghiên cứu

### Tình huống A — PCA rank thấp giữ được oracle PSNR nhưng CNN dự đoán kém

**Kết luận:** Residual có khả năng nén nhưng chưa chắc dự đoán được từ LR. Chuyển trọng tâm sang **predictability-aware subspace**, distillation, hoặc thay loss/predictor. Không tăng backbone vô hạn mà không phân tích lỗi.

### Tình huống B — PCA rank thấp gây truncation error cao

**Kết luận:** Block PCA quá hạn chế. Thử DCT/full null basis, hoặc mở rộng từ per-block độc lập sang structured basis có ngữ cảnh; phải thiết kế lại ràng buộc consistency nếu có trộn giữa blocks.

### Tình huống C — Giảm coefficient head nhưng latency E2E không giảm

**Kết luận:** Lợi ích tính toán không được xác nhận. Xem thời gian synthesis, memory movement, operator fusion; có thể dùng PCA **chỉ ở training/search stage**.

### Tình huống D — Có cải thiện PSNR–latency và search cost

**Kết luận:** Mới có cơ sở để nghiên cứu một phương pháp **hardware-aware capacity allocation** dựa trên null-space statistics. Phải báo cáo baseline NAS/pruning và paper gần nhất trước khi tuyên bố novelty.

---

## 11. Checklist triển khai

- [ ] Cố định \(A\) = block average pooling và scale ×4; ghi cấu hình.
- [ ] Xác nhận \(A^\dagger\) = pixel repetition, không dùng bicubic làm pseudoinverse.
- [ ] Xác nhận RGB channel ordering và `pixel_unshuffle`/`pixel_shuffle` inverse đúng.
- [ ] Unit tests cho \(P^2=P\), \(AP=0\), \(AU_k=0\), consistency và phân rã MSE.
- [ ] Fit covariance từ **train only** bằng `float64` accumulators.
- [ ] Lưu \(\mu,C,U_k,\lambda\), sample count, code version, random seeds.
- [ ] Vẽ PCA eigen-spectrum, explained variance và oracle PSNR theo \(k\).
- [ ] Train coefficient predictors cùng compute budget.
- [ ] So sánh PCA với DCT/random/full null-space baselines.
- [ ] Tính riêng truncation/prediction errors cho cả train và validation.
- [ ] Kiểm tra 3 seeds và độ ổn định basis.
- [ ] Benchmark E2E latency và peak RAM, không chỉ MACs.
- [ ] Nếu mở rộng sang bicubic, xây mới \(A,A^\dagger,P\) hoặc ghi rõ consistency không còn exact.
- [ ] Đối chiếu literature PCA-SR, GAN null-space learning và learned null-space projection trước khi viết novelty claims.

---

## 12. Kết luận nghiên cứu

Điểm xuất phát **đã có bảo đảm toán học** là:

\[
 A\ \longrightarrow\ P=I-A^\dagger A
 \ \longrightarrow\ Z=PX
 \ \longrightarrow\ C=\operatorname{Cov}(Z)
 \ \longrightarrow\ U_k\subset\ker(A).
\]

Khi kết hợp với decoder \(\hat x=A^\dagger y+\mu+U_kc_\theta(y)\), ta giữ được **data consistency** và có **exact MSE error decomposition** trong mô hình đã chọn.

Phần nghiên cứu **chưa được bảo đảm** là:

\[
 \text{Low PCA rank}
 \ \not\Rightarrow\ 
 \text{Easy coefficient prediction}
 \ \not\Rightarrow\ 
 \text{Fast CNN inference}.
\]

Do vậy, quyết định khoa học quan trọng nhất là đo **oracle truncation + prediction error + E2E cost** cùng lúc. Chỉ sau bước đó mới có đủ bằng chứng để nối PCA/null-space theory với mục tiêu giảm không gian tìm kiếm của Efficient SR.

### Tài liệu và mã nền tảng

- Tài liệu nội bộ: `PCA_Null_Space_SR_Theory.docx`.
- Prototype: `rnd_nullspace_pca_sr.py`.
- *GAN Prior Based Null-Space Learning for Consistent Super-Resolution* (AAAI 2023): https://ojs.aaai.org/index.php/AAAI/article/view/25372
- *Non-Linear Projections of the Null-Space for Imaging Inverse Problems* (NeurIPS 2025): https://proceedings.neurips.cc/paper_files/paper/2025/hash/acb1891d79e617134ed604084ebbc919-Abstract-Conference.html
- *Once-for-All* (ICLR 2020): https://arxiv.org/abs/1908.09791
- *ProxylessNAS* (ICLR 2019): https://arxiv.org/abs/1812.00332
