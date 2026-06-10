from pathlib import Path
import os

from clinical_cdss.rag.engine import MedicalGraphRAG
from clinical_cdss.etl import patients as etl_patients
from clinical_cdss.core.database import Neo4jConnection

HGAT_MODEL = Path("models/clinical_hgat.pt")


def initialize_system():
    print("\n--- KHOI TAO HE THONG KNOWLEDGE GRAPH ---")

    # 0. Khoi tao Vector Index cho DB truoc khi nap data
    db = Neo4jConnection()
    db.init_schema()
    db.close()

    print("LUU Y: Phac do y te yeu cau kiem duyet bang tay.")
    print("Hay chac chan da chay 'python -m clinical_cdss.etl.extractor_guidelines'")
    print("va duyet file 'data/draft_guidelines.json' truoc khi chay he thong nay.")

    from clinical_cdss.etl import guidelines as loader_guidelines
    if hasattr(loader_guidelines, "load_extracted_rules_to_graph"):
        loader_guidelines.load_extracted_rules_to_graph()
        loader_guidelines.index_guideline_chunks()
    else:
        print("loader_guidelines.py chua co ham load_extracted_rules_to_graph().")

    data_path = "data/NghiencuuHFLC (1) - NghiencuuHFLC (1).csv"
    if os.path.exists(data_path):
        etl_patients.load_patients_to_graph(data_path)
    else:
        print(f"Khong tim thay file CSV tai {data_path}")


def chat_interface():
    print("\n--- KHOI DONG CDSS ---")

    # Dung HGAT neu model da duoc train, fallback sang RAG neu chua
    cdss = None
    if HGAT_MODEL.exists():
        try:
            from clinical_cdss.gnn.predict import ClinicalCDSS
            cdss = ClinicalCDSS(str(HGAT_MODEL))
            print(f"HGAT model loaded: {HGAT_MODEL}")
        except Exception as exc:
            print(f"HGAT load that bai, dung RAG: {exc}")
            cdss = None
    else:
        print(f"Chua co HGAT model ({HGAT_MODEL}). Dung RAG Coverage Score.")
        print("De train: python -m clinical_cdss.gnn.train --epochs 120 --patience 20")

    rag = MedicalGraphRAG()
    print("\nHe thong da san sang! (Go 'exit' de thoat)")

    while True:
        patient_name = input("\nNhap ID/Ten benh nhan can danh gia: ").strip()
        if patient_name.lower() in ["exit", "quit"]:
            break
        if not patient_name:
            continue

        print("Dang phan tich...")
        try:
            if cdss is not None:
                result = cdss.diagnose(patient_name, use_llm=False)
                if "error" in result:
                    print(f"Loi: {result['error']}")
                else:
                    print(f"\nPhuong phap: {result['method']} | Do tin cay: {result['confidence']:.1%}")
                    print("-" * 60)
                    print(result["report"])
                    print("-" * 60)
            else:
                answer = rag.ask(patient_name, use_llm=False)
                print("\nBAO CAO DANH GIA (Coverage Score RAG):")
                print("-" * 60)
                print(answer)
                print("-" * 60)
        except Exception as exc:
            print(f"Loi: {exc}")


if __name__ == "__main__":
    # De build lai DB theo kien truc moi:
    #   1. Vao Neo4j xoa trang: MATCH (n) DETACH DELETE n
    #   2. Bo comment dong initialize_system() ben duoi
    #   3. Chay lai script nay

    # initialize_system()
    chat_interface()
