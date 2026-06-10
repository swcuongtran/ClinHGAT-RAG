import time
import pandas as pd
from clinical_cdss.rag.engine import MedicalGraphRAG
from clinical_cdss.core.database import Neo4jConnection

def run_benchmark(num_samples=10, output_file="data/benchmark_results.csv"):
    print(f"🚀 Bắt đầu chạy Benchmark trên {num_samples} bệnh nhân ngẫu nhiên...")
    
    # 1. Khởi tạo Engine
    print("⏳ Đang khởi tạo MedicalGraphRAG (Load LLM)...")
    engine = MedicalGraphRAG()
    db = Neo4jConnection()
    
    # 2. Lấy danh sách bệnh nhân ngẫu nhiên từ Neo4j
    query = """
    MATCH (p:Patient)
    RETURN p.id AS id
    ORDER BY rand()
    LIMIT $limit
    """
    patients_data = db.execute_query(query, {"limit": num_samples})
    patient_ids = [record["id"] for record in patients_data]
    db.close()
    
    if not patient_ids:
        print("⚠️ Không tìm thấy bệnh nhân nào trong CSDL Neo4j.")
        return

    results = []
    
    # 3. Chạy từng bệnh nhân
    for idx, pid in enumerate(patient_ids, 1):
        print(f"\n[{idx}/{num_samples}] Đang đánh giá bệnh nhân: {pid}...")
        
        # --- Đo thời gian Retrieval ---
        t0 = time.time()
        context_data = engine.retrieve_context(pid)
        retrieval_time = time.time() - t0
        
        if "error" in context_data:
            print(f"  ❌ Lỗi lấy dữ liệu: {context_data['error']}")
            continue
            
        context_length = len(context_data['context_str'])
        # Lấy coverage_score từ context nếu có
        coverage_score = context_data.get('coverage_score', 0.0)
        print(f"  ✅ Kéo Graph xong: {retrieval_time:.2f} giây | Coverage: {round(coverage_score*100,1)}% | Context: {context_length} ký tự")
        
        # --- Đo thời gian Sinh LLM ---
        t1 = time.time()
        report = engine.generate_response(context_data)
        llm_time = time.time() - t1
        
        print(f"  ✅ Sinh báo cáo xong: {llm_time:.2f} giây")
        
        # Lưu kết quả
        results.append({
            "Patient_ID": pid,
            "Symptoms": context_data["symptoms_text"],
            "Coverage_Score_pct": round(coverage_score * 100, 1),
            "Retrieval_Time_s": round(retrieval_time, 2),
            "LLM_Generation_Time_s": round(llm_time, 2),
            "Context_Length_chars": context_length,
            "Generated_Report": report
        })

        
    # 4. Xuất ra file CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n🎉 Hoàn tất Benchmark! Đã lưu kết quả tại: {output_file}")
        
        # In tóm tắt
        avg_retrieval = df["Retrieval_Time_s"].mean()
        avg_llm = df["LLM_Generation_Time_s"].mean()
        print(f"📊 Thời gian Retrieval (Kéo Graph) trung bình: {avg_retrieval:.2f} s")
        print(f"📊 Thời gian LLM Sinh báo cáo trung bình: {avg_llm:.2f} s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Medical Graph RAG Benchmark")
    parser.add_argument("--samples", type=int, default=5, help="Số lượng bệnh nhân cần benchmark")
    parser.add_argument("--out", type=str, default="data/benchmark_results.csv", help="Đường dẫn file lưu kết quả")
    args = parser.parse_args()
    
    run_benchmark(num_samples=args.samples, output_file=args.out)
