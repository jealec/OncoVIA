import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.vision_pipeline import scan_all_patients

def create_augmented_docs(xlsx_path, imaging_map):
    df = pd.read_excel(xlsx_path, engine='openpyxl')
    df.fillna({'Clinical information data (Pseudo reports)': "", 'AccessionNumber': "UNKNOWN"}, inplace=True)
    
    documents = []
    for _, row in df.iterrows():
        report_text = str(row['Clinical information data (Pseudo reports)']).strip()
        acc_num = str(row['AccessionNumber']).strip()
        
        if not report_text: continue

        metadata = {
            "PatientID": str(row['PatientID']).strip(),
            "AccessionNumber": acc_num,
        }
        
        if acc_num in imaging_map:
            img_info = imaging_map[acc_num]
            metadata["date"] = img_info["date"]
            metadata["anatomical_locations"] = img_info["anatomical_locations"]
            metadata["source_seg_path"] = img_info["seg_file_path"]
        else:
            metadata["date"] = "NOT_FOUND_IN_FOLDERS"
            metadata["anatomical_locations"] = "N/A"

        documents.append(Document(page_content=report_text, metadata=metadata))
        
    return documents

def build_database(dataset_path, excel_file, chroma_dir):
    print("--- STEP 1: SCANNING ALL DATASET FOLDERS (With Checkpoints) ---")
    all_imaging_metadata = scan_all_patients(dataset_path, checkpoint_file="imaging_checkpoint.json")

    print("\n--- STEP 2: MERGING EXCEL WITH FOLDER DATA ---")
    final_docs = create_augmented_docs(excel_file, all_imaging_metadata)

    print(f"\n--- STEP 3: CHUNKING {len(final_docs)} DOCUMENTS ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150)
    chunks = text_splitter.split_documents(final_docs)

    print("\n--- STEP 4: BUILDING CHROMA DB IN BATCHES ---")
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings", model_kwargs={'device': 'cpu'})
    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

    BATCH_SIZE = 150
    total_chunks = len(chunks)

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        db.add_documents(batch)
        print(f"   [DB SAVE] Inserted batch {i//BATCH_SIZE + 1} / {(total_chunks - 1)//BATCH_SIZE + 1}")

    print("\n✅ Process Complete. Database built safely.")

# Décommente cette section si tu veux recréer la DB en lançant ce script
# if __name__ == "__main__":
#     build_database("data/raw/dataset", "data/raw/Liste examen UNBOXED finaliseģe v2 (avec mesures).xlsx", "db/chroma_db_full_dataset")