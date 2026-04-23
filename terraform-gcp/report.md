# Báo cáo LAB 16 — Phương án CPU (LightGBM thay GPU)

## Lý do sử dụng CPU thay GPU
Tài khoản GCP mới bị khóa quota GPU (NVIDIA T4 = 0) theo mặc định.
Yêu cầu tăng quota không tìm được trong IAM Quotas do Compute Engine API
chưa hiển thị đủ quota mới. Thay vì bỏ qua lab, chuyển sang Phần 7
(phương án dự phòng hợp lệ, được chấm tương đương).

## Hạ tầng đã triển khai
- Cloud Provider : Google Cloud Platform (GCP)
- Instance type  : n2-standard-8 (8 vCPU, 32 GB RAM) — không cần quota GPU
- Region / Zone  : us-central1 / us-central1-a
- Network        : Private VPC + Cloud NAT + External HTTP Load Balancer
- Workload       : LightGBM (gradient boosting) — Credit Card Fraud Detection
- API port       : 8000 (Flask) — health check tại /health, results tại /v1/benchmark/result

## Kết quả benchmark

| Metric               | Kết quả           |
|----------------------|-------------------|
| Instance type        | n2-standard-8 CPU |
| Dataset              | 50,000 rows (synthetic, fraud rate 0.174%) |
| Data load time       | 0.031s            |
| Training time        | 0.285s            |
| Best iteration       | 9                 |
| AUC-ROC              | 0.5447            |
| Accuracy             | 99.76%            |
| Inference (1 row)    | 0.615 ms          |
| Inference (1000 rows)| 0.859 ms          |

## So sánh CPU vs GPU

| Tiêu chí             | GPU (T4)          | CPU (n2-standard-8) |
|----------------------|-------------------|----------------------|
| Quota yêu cầu        | Cần xin duyệt     | Không cần            |
| Chi phí/giờ          | ~$0.54            | ~$0.43               |
| Thời gian deploy     | ~5 phút           | ~5 phút              |
| Phù hợp workload     | Deep Learning/LLM | Gradient Boosting/ML |
| Training time (lab)  | N/A               | 0.285s               |

## Cold Start Time
- Thời gian terraform apply hoàn tất  : ~5 phút
- Thời gian cài packages + chạy server: ~3 phút
- Tổng cold start đến API /health OK  : ~8 phút (mục tiêu < 15 phút ✅)

## Cleanup
terraform destroy đã chạy thành công — 0 resources còn lại.
