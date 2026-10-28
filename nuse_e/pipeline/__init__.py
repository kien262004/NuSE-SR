"""NuSE-E: toàn bộ quy trình thí nghiệm, chạy từ dòng lệnh (thay cho notebook).

Mỗi bước đọc đầu vào và ghi kết quả ra thư mục output, nên chạy lại một bước bất kỳ không cần chạy lại các bước
trước (khác notebook, nơi khởi động lại kernel làm mất mọi biến). Thứ tự các bước:

    setup     tìm dữ liệu, tạo cache ảnh, kiểm checkpoint cũ, (tuỳ chọn) chép kết quả phiên trước
    stats     thống kê PCA null space + suy kiến trúc cho mọi biến thể
    test      test chấp nhận (UᵀU = I, A(SR) = y, co giãn theo K, ...)
    screen    sàng tham số và latency trước khi train
    train     train mọi run còn thiếu (một tiến trình mỗi GPU), rồi pha B trên LR bicubic
    curves    hình đường học
    select    chọn biến thể tốt nhất mỗi loại LR
    evaluate  PSNR, SSIM, kênh Y, LPIPS, FID trên DIV2K valid (và benchmark nếu có)
    cost      tham số, GFLOPs, latency
    report    bảng tổng hợp, kiểm ràng buộc, report.md
    figures   hình so sánh
    all       chạy lần lượt mọi bước trên

Mỗi bước nằm trong một module của package này: prepare (setup, stats), checks (test, screen),
training (train), selection (select), evaluation (evaluate, cost), report (report),
figures (curves, figures). Bối cảnh dùng chung ở context.

Ví dụ: python -m nuse_e.pipeline setup --train-hr <DIV2K_train_HR> --val-hr <DIV2K_valid_HR>
       python -m nuse_e.pipeline all
Xem nuse_e/README.md.
"""
