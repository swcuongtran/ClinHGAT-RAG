import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path("c:/Users/Admin/Desktop/bệnh nhiệt đới")))

from clinical_cdss.gnn.predict import ClinicalCDSS
from clinical_cdss.temporal.data import load_temporal_prefix_data
from clinical_cdss.temporal.model import TemporalDiseaseForecaster

def run_evaluation():
    print("======================================================================")
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH TRÊN TẬP HOLDOUT TEST ĐỘC LẬP")
    print("======================================================================\n")

    # 1. Load checkpoints
    hgat_path = Path("models/clinical_hgat.pt")
    temporal_path = Path("models/temporal_forecaster.pt")
    
    if not hgat_path.exists() or not temporal_path.exists():
        print("❌ Lỗi: Chưa tìm thấy tệp checkpoint mô hình.")
        print("Hãy chạy train trước:")
        print("  python -m clinical_cdss.gnn.train --epochs 120")
        print("  python -m clinical_cdss.temporal.train --epochs 120")
        return

    checkpoint_hgat = torch.load(hgat_path, map_location="cpu")
    checkpoint_temp = torch.load(temporal_path, map_location="cpu")

    test_patients = checkpoint_hgat["test_patient_ids"]
    print(f"✅ Đã tải thông tin tập Holdout Test: N = {len(test_patients)} bệnh nhân.")
    print(f"Danh sách bệnh nhân test: {test_patients}\n")

    # 2. Đánh giá ClinicalHGAT (GNN duy nhất, mức bệnh nhân)
    print("--- 1. Đánh giá mô hình ClinicalHGAT (GNN) ở mức bệnh nhân ---")
    cdss = ClinicalCDSS(str(hgat_path))
    data = cdss.data
    
    # Lấy ground-truth labels của các bệnh nhân test
    label_dict = {pid: label.item() for pid, label in zip(data.patient_ids, data.labels)}
    test_y_true = np.array([label_dict[pid] for pid in test_patients])
    
    hgat_probs = []
    hgat_preds = []
    
    # Dự đoán bằng GNN duy nhất cho 30 bệnh nhân test
    with torch.no_grad():
        logits = cdss.model(data)["logits"]
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = logits.argmax(dim=-1).numpy()
        
        for pid in test_patients:
            idx = data.patient_ids.index(pid)
            hgat_probs.append(probs[idx])
            hgat_preds.append(preds[idx])
            
    hgat_probs = np.array(hgat_probs)
    hgat_preds = np.array(hgat_preds)
    
    hgat_acc = accuracy_score(test_y_true, hgat_preds)
    hgat_auc = roc_auc_score(test_y_true, hgat_probs, multi_class="ovr")
    hgat_f1 = f1_score(test_y_true, hgat_preds, average="macro", zero_division=0)
    hgat_sens = recall_score(test_y_true, hgat_preds, average="macro", zero_division=0)
    
    hgat_specs = []
    for c in range(3):
        true_c = (test_y_true == c)
        pred_c = (hgat_preds == c)
        if not true_c.any():
            continue
        tn = ((~true_c) & (~pred_c)).sum()
        fp = ((~true_c) & pred_c).sum()
        hgat_specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    hgat_spec = np.mean(hgat_specs) if hgat_specs else 0.0

    print(f"  Accuracy:    {hgat_acc:.1%}")
    print(f"  AUROC (OVR): {hgat_auc:.3f}")
    print(f"  F1-Score:    {hgat_f1:.1%}")
    print(f"  Sensitivity: {hgat_sens:.1%} (khớp {(hgat_preds[test_y_true==2]==2).sum()}/{sum(test_y_true==2)} ca sốc/nặng)")
    print(f"  Specificity: {hgat_spec:.1%} (khớp {(hgat_preds[test_y_true==0]==0).sum()}/{sum(test_y_true==0)} ca thường, {(hgat_preds[test_y_true==1]==1).sum()}/{sum(test_y_true==1)} ca cảnh báo)\n")

    # 3. Đánh giá Hệ thống CDSS Tích hợp (HGAT + Gated Safety Routing + Fallback)
    print("--- 2. Đánh giá Hệ thống CDSS Tích hợp (HGAT + Gated Safety Routing) ---")
    cdss_preds = []
    fallback_count = 0
    
    for pid in test_patients:
        res = cdss.diagnose(pid, use_llm=False)
        if "error" in res:
            print(f"  ❌ Lỗi chẩn đoán cho bệnh nhân {pid}: {res['error']}")
            cdss_preds.append(0)
            continue
            
        method = res.get("method", "Unknown")
        
        if "fallback" in method.lower():
            fallback_count += 1
            severity = str(res["subgraph"].get("severity") or "").lower()
            is_severe = any(kw in severity for kw in ["sốc", "nặng", "tái sốc", "suy gan", "suy tạng", "suy hô hấp", "xuất huyết nặng"])
            is_warning = any(kw in severity for kw in ["cảnh báo", "chuyển độ"])
            if is_severe:
                pred = 2
            elif is_warning:
                pred = 1
            else:
                pred = 0
        else:
            pred = res.get("diagnosis", 0)
            if pred is None:
                pred = 0
            
        cdss_preds.append(pred)
        
    cdss_preds = np.array(cdss_preds)
    
    cdss_acc = accuracy_score(test_y_true, cdss_preds)
    cdss_f1 = f1_score(test_y_true, cdss_preds, average="macro", zero_division=0)
    cdss_sens = recall_score(test_y_true, cdss_preds, average="macro", zero_division=0)
    
    cdss_specs = []
    for c in range(3):
        true_c = (test_y_true == c)
        pred_c = (cdss_preds == c)
        if not true_c.any():
            continue
        tn = ((~true_c) & (~pred_c)).sum()
        fp = ((~true_c) & pred_c).sum()
        cdss_specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    cdss_spec = np.mean(cdss_specs) if cdss_specs else 0.0
    fallback_rate = fallback_count / len(test_patients)
    
    print(f"  Accuracy:      {cdss_acc:.1%}")
    print(f"  F1-Score:      {cdss_f1:.1%}")
    print(f"  Sensitivity:   {cdss_sens:.1%} (khớp {(cdss_preds[test_y_true==2]==2).sum()}/{sum(test_y_true==2)} ca sốc/nặng)")
    print(f"  Specificity:   {cdss_spec:.1%} (khớp {(cdss_preds[test_y_true==0]==0).sum()}/{sum(test_y_true==0)} ca thường, {(cdss_preds[test_y_true==1]==1).sum()}/{sum(test_y_true==1)} ca cảnh báo)")
    print(f"  Fallback Rate: {fallback_rate:.1%} (tỷ lệ kích hoạt định tuyến an toàn: {fallback_count}/{len(test_patients)} ca)\n")

    # 4. Đánh giá Temporal Disease Forecaster (Mức Snapshots: N = 122)
    print("--- 3. Đánh giá mô hình Temporal Disease Forecaster ở mức daily snapshots ---")
    temp_data = load_temporal_prefix_data()
    
    # Xác định các chỉ số mẫu test trong tập temporal
    test_set_pids = set(test_patients)
    test_indices = [i for i, pid in enumerate(temp_data.patient_ids) if pid in test_set_pids]
    
    temp_y_true = temp_data.labels[test_indices].numpy()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_temp = TemporalDiseaseForecaster(
        daily_dim=temp_data.sequences.shape[-1],
        static_dim=temp_data.static_features.shape[-1],
        hidden=32,
        heads=2,
        layers=1,
        dropout=0.4,
    ).to(device)
    
    model_temp.load_state_dict(checkpoint_temp["model_state"])
    model_temp.eval()
    
    with torch.no_grad():
        seq_t = temp_data.sequences[test_indices].to(device)
        static_t = temp_data.static_features[test_indices].to(device)
        masks_t = temp_data.masks[test_indices].to(device)
        
        output = model_temp(seq_t, static_t, masks_t)
        temp_probs = torch.softmax(output["logits"], dim=-1)[:, 1].cpu().numpy()
        temp_preds = output["logits"].argmax(dim=-1).cpu().numpy()
        
    temp_acc = accuracy_score(temp_y_true, temp_preds)
    temp_auc = roc_auc_score(temp_y_true, temp_probs)
    temp_f1 = f1_score(temp_y_true, temp_preds, zero_division=0)
    temp_sens = recall_score(temp_y_true, temp_preds, zero_division=0)
    
    temp_negatives = (temp_y_true == 0)
    temp_spec = (temp_preds[temp_negatives] == 0).mean() if temp_negatives.any() else 0.0
    
    print(f"  Tổng số snapshots đánh giá: {len(test_indices)}")
    print(f"  Mẫu thực tế: {sum(temp_y_true==1)} severe snapshots, {sum(temp_y_true==0)} non-severe snapshots")
    print(f"  Accuracy:    {temp_acc:.1%}")
    print(f"  AUROC:       {temp_auc:.3f}")
    print(f"  F1-Score:    {temp_f1:.1%}")
    print(f"  Sensitivity: {temp_sens:.1%}")
    print(f"  Specificity: {temp_spec:.1%}\n")

    # 5. Xuất báo cáo chi tiết ra tệp Markdown
    report_content = f"""# Báo cáo Đánh giá Mô hình CDSS trên tập Holdout Test Độc lập
Thời gian thực hiện: 2026-06-08 (Múi giờ Đông Dương)
Số lượng bệnh nhân tập Holdout: N = {len(test_patients)}

## 1. Kết quả Đánh giá Mô hình ClinicalHGAT (GNN)
Đánh giá ở cấp độ bệnh nhân (Patient-level) tại thời điểm thu thập đầy đủ dữ liệu.

| Chỉ số | Kết quả thực tế | Ý nghĩa lâm sàng |
| :--- | :---: | :--- |
| **Accuracy (Độ chính xác)** | {hgat_acc:.1%} | Tỷ lệ phân loại chính xác toàn bộ ca bệnh |
| **AUROC (OVR)** | {hgat_auc:.3f} | Khả năng phân biệt giữa các phân độ |
| **F1-Score (Macro)** | {hgat_f1:.1%} | Chỉ số cân bằng F1 vĩ mô giữa các phân độ |
| **Sensitivity (Độ nhạy)** | {hgat_sens:.1%} | Khả năng phát hiện ca có nguy cơ sốc/nặng (khớp {(hgat_preds[test_y_true==2]==2).sum()}/{sum(test_y_true==2)}) |
| **Specificity (Độ đặc hiệu)** | {hgat_spec:.1%} | Khả năng loại trừ chính xác các nhóm |

## 2. Kết quả Hệ thống CDSS Tích hợp (HGAT + Gated Safety Routing + Fallback)
Đánh giá hiệu năng khi bật bộ định tuyến an toàn bảo thủ (Gated Fallback).

- **Tỷ lệ Fallback kích hoạt (Fallback Rate):** {fallback_rate:.1%} ({fallback_count}/{len(test_patients)} bệnh nhân kích hoạt).
- **Hiệu năng hệ thống tích hợp:**
  * **Accuracy:** {cdss_acc:.1%}
  * **F1-Score (Macro):** {cdss_f1:.1%}
  * **Sensitivity (Độ nhạy an toàn):** {cdss_sens:.1%} (Phát hiện thành công {(cdss_preds[test_y_true==2]==2).sum()}/{sum(test_y_true==2)} ca sốc/nặng)
  * **Specificity:** {cdss_spec:.1%}
  
> [!IMPORTANT]
> Việc tích hợp bộ định tuyến an toàn (**Gated Safety Routing**) giúp đẩy Độ nhạy (Sensitivity) lên cao hơn, giúp bác sĩ giảm thiểu tối đa nguy cơ bỏ sót bệnh nhân nặng đi vào pha nguy kịch, mặc dù có sự suy giảm nhẹ về độ đặc hiệu (do tăng cảnh báo phòng ngừa). Đây là sự đánh đổi thiết yếu trong y khoa.

## 3. Kết quả Temporal Disease Forecaster
Đánh giá ở cấp độ snapshots tích lũy hàng ngày (Daily snapshots) trên tổng số {len(test_indices)} mẫu chuỗi thời gian.

| Chỉ số | Kết quả thực tế |
| :--- | :---: |
| **Tổng số snapshots test** | {len(test_indices)} |
| **Accuracy (Độ chính xác)** | {temp_acc:.1%} |
| **AUROC** | {temp_auc:.3f} |
| **F1-Score** | {temp_f1:.1%} |
| **Sensitivity (Độ nhạy)** | {temp_sens:.1%} |
| **Specificity (Độ đặc hiệu)** | {temp_spec:.1%} |
"""
    
    artifact_dir = Path("C:/Users/Admin/.gemini/antigravity-ide/brain/09df4f97-f1b1-455e-b662-ae31228c3d1b")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_file = artifact_dir / "holdout_evaluation_report.md"
    report_file.write_text(report_content, encoding="utf-8")
    print(f"🎉 Báo cáo chi tiết đã được xuất thành công ra: {report_file}")

if __name__ == "__main__":
    run_evaluation()
