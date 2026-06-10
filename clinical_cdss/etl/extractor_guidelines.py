import fitz
import json
import re
from typing import List
from llama_cpp import Llama, LlamaGrammar
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- QUẢN LÝ MÔ HÌNH TOÀN CỤC ---
llm_local = None

def get_llm():
    global llm_local
    if llm_local is None:
        print("⏳ Đang load Qwen 7B vào GPU...")
        llm_local = Llama(
            model_path=str(PROJECT_ROOT / "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf"),
            n_ctx=8192, 
            n_gpu_layers=-1, 
            n_threads=4, 
            verbose=False
        )
        print("✅ Load model thành công!")
    return llm_local

def extract_text_chunks_from_pdf(pdf_path, start_page, end_page, chunk_size=4, overlap=2):
    chunks = []
    actual_start, actual_end = start_page - 1, end_page - 1
    doc = fitz.open(pdf_path)
    actual_end = min(actual_end, len(doc) - 1)
    
    step = chunk_size - overlap
    if step <= 0: step = 1
    
    for i in range(actual_start, actual_end + 1, step):
        text = ""
        end_idx = min(i + chunk_size, actual_end + 1)
        for j in range(i, end_idx):
            text += doc[j].get_text("text") + "\n"
        chunks.append((i + 1, end_idx, text))
        if end_idx >= actual_end + 1:
            break
            
    doc.close()
    return chunks

# --- 2. NGỮ PHÁP LLAMAGRAMMAR HỖ TRỢ MẢNG (LIST) ---
json_grammar_str = r'''
root ::= "{" ws "\"rules\"" ws ":" ws "[" ws rulelist "]" ws "}"
rulelist ::= rule (ws "," ws rule)* | ""
rule ::= "{" ws "\"phase\"" ws ":" ws string ws "," ws "\"severity\"" ws ":" ws string ws "," ws "\"clinical_signs\"" ws ":" ws stringlist ws "," ws "\"lab_tests\"" ws ":" ws stringlist ws "," ws "\"treatments\"" ws ":" ws stringlist ws "," ws "\"contraindications\"" ws ":" ws stringlist ws "}"
stringlist ::= "[" ws stringitems "]"
stringitems ::= string (ws "," ws string)* | ""
string ::= "\"" ([^"] | "\\\"")* "\""
ws ::= [ \t\n]*
'''
grammar = LlamaGrammar.from_string(json_grammar_str)

def _repair_truncated_json(text: str) -> str:
    """Cố gắng sửa JSON bị cắt ngang do Token Overflow bằng cách thêm dấu đóng ngoặc còn thiếu."""
    # Đếm số dấu mở vs đóng
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    
    # Nếu chuỗi kết thúc giữa chừng trong một string, xóa phần thừa đó
    # Tìm dấu nháy kép cuối cùng không bị escape
    last_complete_pos = max(text.rfind('",'), text.rfind('"\n'), text.rfind('"]'), text.rfind('"}'))
    if last_complete_pos > 0:
        text = text[:last_complete_pos + 1]
    
    # Đóng tất cả mảng và object còn đang mở
    missing_brackets = max(0, open_brackets - close_brackets)
    missing_braces = max(0, open_braces - close_braces)
    text = text + ']' * missing_brackets + '}' * missing_braces
    return text

def extract_graph_rules_with_local_llm(raw_text):
    llm = get_llm()
    # Dọn dẹp ký tự điều khiển ẩn (chỉ loại bỏ mã ASCII dưới 32 trừ tab và newline) để GIỮ LẠI TIẾNG VIỆT
    clean_text = re.sub(r'[\x00-\x08\x0B-\x1F\x7F]', ' ', raw_text)
    
    # Giới hạn độ dài văn bản đầu vào để tránh Token Overflow (max ~2500 ký tự)
    # Tương đương ~700-800 token, để lại không gian cho prompt + JSON output
    MAX_INPUT_CHARS = 3000
    if len(clean_text) > MAX_INPUT_CHARS:
        clean_text = clean_text[:MAX_INPUT_CHARS] + "\n...[Văn bản đã bị cắt để tránh tràn ngữ cảnh]..."
    
    prompt = f"""Trích xuất Bệnh cảnh Lâm sàng Sốt xuất huyết (Clinical Scenarios) từ văn bản.
- Gộp các thông tin thành từng Khối Quy tắc dựa theo Mức độ và Giai đoạn.
- Nếu không có thông tin ở một mảng (ví dụ không có chống chỉ định), hãy để mảng rỗng [].
- Tuyệt đối không bịa thêm thông tin ngoài văn bản.
- Giữ mỗi chuỗi trong mảng NGẮN GỌN (dưới 100 ký tự).

Ví dụ Output mong muốn:
{{"rules": [{{"phase": "Giai đoạn nguy hiểm", "severity": "Sốc Dengue", "clinical_signs": ["Mạch nhanh nhỏ", "Huyết áp kẹt"], "lab_tests": ["HCT tăng", "Tiểu cầu giảm"], "treatments": ["Truyền Ringer Lactate 5-7ml/kg/h"], "contraindications": ["Không dùng Aspirin"]}}]}}

VĂN BẢN CẦN PHÂN TÍCH:
{clean_text}

JSON TRẢ VỀ:"""
    try:
        response = llm(
            prompt,
            max_tokens=3000,
            temperature=0.0,
            grammar=grammar
        )
        response_text = response['choices'][0]['text']
        
        match = re.search(r'\{.*', response_text, re.DOTALL)
        if not match:
            print("   ⚠️ Không tìm thấy JSON hợp lệ trong kết quả trả về của LLM.")
            return []
        
        raw_json = match.group(0)
        
        # Thử parse bình thường trước
        try:
            data = json.loads(raw_json, strict=False)
        except json.JSONDecodeError:
            # Nếu lỗi (bị cắt ngang), thử tự sửa JSON
            print("   🔧 Phát hiện JSON bị cắt - đang thử Auto-Repair...")
            repaired = _repair_truncated_json(raw_json)
            try:
                data = json.loads(repaired, strict=False)
                print("   ✅ Auto-Repair thành công!")
            except json.JSONDecodeError as e2:
                print(f"   ⚠️ Auto-Repair thất bại: {e2}")
                return []
        
        extracted_rules = []
        for item in data.get('rules', []):
            has_data = any([
                item.get('clinical_signs'), item.get('lab_tests'), 
                item.get('treatments'), item.get('contraindications')
            ])
            if has_data:
                extracted_rules.append(item)
                
        return extracted_rules
    except Exception as e:
        print(f"   ⚠️ Lỗi không xác định: {e}")
        return []

def extract_guidelines_to_draft(pdf_path="data/Huong-dan-chan-doan-va-dieu-tri.pdf", start_page=207, end_page=278, chunk_size=2, overlap=1, output_file="data/draft_guidelines.json"):
    if not os.path.exists(pdf_path):
        print(f"⚠️ Không tìm thấy file PDF tại {pdf_path}")
        return

    chunks = extract_text_chunks_from_pdf(pdf_path, start_page, end_page, chunk_size=chunk_size, overlap=overlap)
    all_rules = []
    
    print(f"📚 Bắt đầu trích xuất {len(chunks)} phần tài liệu từ {pdf_path}...\n")
    for idx, (start, end, raw_text) in enumerate(chunks):
        if len(raw_text.strip()) < 50: continue
        print(f"\n▶️ Đang xử lý Trang {start} - {end}...")
        
        rules = extract_graph_rules_with_local_llm(raw_text)
        if rules:
            all_rules.extend(rules)
            print(f"   💡 [THÀNH CÔNG] Thu hoạch được {len(rules)} Bệnh cảnh lâm sàng!")
        else:
            print("   ℹ️ [THÔNG TIN] Không tìm thấy Bệnh cảnh rõ ràng ở đoạn này.")
            
    if all_rules:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_rules, f, ensure_ascii=False, indent=4)
        print(f"\n🎉 HOÀN TẤT TRÍCH XUẤT! Vui lòng mở file '{output_file}' để kiểm duyệt (Human-in-the-loop) trước khi nạp đồ thị.")

if __name__ == "__main__":
    extract_guidelines_to_draft()
