import json
from collections.abc import Iterable
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.vision_pipeline import scan_patient

PROMPT = {
    "patient": """You are an expert thoracic radiologist specialized in oncologic imaging.
Your task is to synthesize the automated quantitative computer vision findings with the patient's retrieved historical reports in order to generate a professional, patient-oriented CT report.
The report is addressed to a patient so vulgarize and be comprehensive of the patient sentiment. We are talking about cancer here.

All disease evolution assessments must be aligned with RECIST 1.1 principles.  
However, RECIST categories must be translated into clear, non-technical language suitable for a patient.  

Structure the report strictly into the following sections:
1. WHY THE SCAN WAS PERFORMED  
2. WHAT WAS COMPARED  
3. CURRENT FINDINGS
4. HOW THE LESIONS CHANGED  
5. OVERALL CONCLUSION  
6. WHAT HAPPENS NEXT  

### PATIENT ID: {patient_id}
### CURRENT EXAM FINDINGS (Automated Extraction):
{current_findings}
### PRIOR HISTORY (Retrieved from Database):
{historical_context}""",

    "radiologist": """SYSTEM ROLE :
You are a thoracic radiologist AI generating a RECIST 1.1–compliant pulmonary oncologic CT report for clinicians.

────────────────────
RECIST 1.1 LOGIC
────────────────────
Per lesion:
Δmm = current_mm − ref_mm
Δ% = 100 × (current_mm − ref_mm) / ref_mm  (if ref_mm > 0)

────────────────────
OUTPUT STRUCTURE (FIXED ORDER)
────────────────────
1. Clinical Context
2. Technique
3. Comparison
4. Target Lesions
5. Non-Target Lesions
6. New Lesions
7. Global Tumor Burden
8. RECIST 1.1 Assessment
9. Additional Thoracic Findings
10. Impression
11. Recommendation

### PATIENT ID: {patient_id}
### CURRENT EXAM FINDINGS (Automated Extraction):
{current_findings}
### PRIOR HISTORY (Retrieved from Database):
{historical_context}"""
}

def query_patient_data(patient_id, user_query, db_directory="db/chroma_db_full_dataset"):
    print(f"Searching for Patient: {patient_id} | Query: '{user_query}'")
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings", model_kwargs={'device': 'cpu'})
    db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    
    results = db.similarity_search(query=user_query, k=3, filter={"PatientID": str(patient_id)})

    if not results:
        return []

    return results

def format_current_findings(cv_input):
    formatted_text = ""
    for acc_num, data in cv_input.items():
        date = data.get('date', 'UNKNOWN')
        formatted_text += f"\nExam Accession: {acc_num} (Date: {date})\n"
        anatomy_string = data.get('anatomical_locations', '[]')
        try:
            anatomy_list = json.loads(anatomy_string)
            if not anatomy_list:
                formatted_text += "- No significant tumors detected in this scan.\n"
            else:
                for finding in anatomy_list:
                    tumor_id = finding.get('Tumeur', 'Unknown')
                    diam = finding.get('Diametre_mm', 'Unknown')
                    vol = finding.get('Volume_cm3', 'Unknown')
                    loc = finding.get('Localisation_principale', 'Unknown')
                    formatted_text += f"- {tumor_id}: Diameter {diam} mm, Volume {vol} cm3, located in {loc}\n"
        except json.JSONDecodeError:
            formatted_text += "- Error reading anatomical data for this exam.\n"
    return formatted_text

def format_historical_context(retrieved_results):
    unique_docs = {}
    for result_list in retrieved_results:
        if result_list: 
            for doc in result_list:
                content = doc.page_content if not isinstance(doc, str) else doc
                metadata = doc.metadata if not isinstance(doc, str) else {"AccessionNumber": "Unknown"}
                if content not in unique_docs:
                    unique_docs[content] = metadata

    if not unique_docs:
        return "No prior relevant reports found in the database."

    history_text = ""
    for content, metadata in unique_docs.items():
        acc = metadata.get('AccessionNumber', 'Unknown')
        history_text += f"--- Prior Record (Accession: {acc}) ---\n"
        
        historical_anatomy_str = metadata.get('anatomical_locations', '[]')
        if historical_anatomy_str not in ["N/A", "[]", ""]:
            try:
                hist_anatomy_list = json.loads(historical_anatomy_str)
                history_text += "Prior Extracted Measurements (Automated):\n"
                for finding in hist_anatomy_list:
                    history_text += f"  - {finding.get('Tumeur')}: Diameter {finding.get('Diametre_mm')} mm, Volume {finding.get('Volume_cm3')} cm3\n"
            except json.JSONDecodeError:
                pass
                
        legacy_size = metadata.get('lesion_size_mm', 'Unknown')
        if legacy_size != 'Unknown':
            history_text += f"Prior Target Lesion Size (Manual): {legacy_size} mm\n"
            
        history_text += f"Report Excerpt:\n{content}\n\n"
    return history_text

def generate_search_queries(global_imaging_data, chat_model):
    print("Agent: Parsing imaging data and generating queries...")
    all_queries = []
    
    for accession_num, data in global_imaging_data.items():
        anatomy_string = data.get('anatomical_locations', '[]')
        try:
            anatomy_list = json.loads(anatomy_string)
        except json.JSONDecodeError:
            continue
            
        if not anatomy_list: continue
            
        clean_locations = []
        for finding in anatomy_list:
            loc = finding.get('Localisation_principale', 'Unknown location')
            if loc != 'Unknown location' and loc not in clean_locations:
                clean_locations.append(loc)
                
        locations_text = "\n".join([f"- {loc}" for loc in clean_locations])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical data extraction assistant. 
Your task is to generate short, natural language search queries to find prior radiology reports mentioning specific anatomical locations.
Output ONLY the search queries, one per line."""),
            ("human", "Anatomical Locations Found:\n{locations_text}")
        ])
        
        chain = prompt | chat_model
        response = chain.invoke({"locations_text": locations_text})
        queries = [q.strip('- *').strip() for q in response.content.split('\n') if q.strip()]
        all_queries.extend(queries)
        
    return list(set(all_queries))

def generate_final_radiology_report(patientID, folder_path, users):
    print(f"\n================ STARTING PIPELINE FOR {patientID} ================\n")
    
    # 1. Vision Extraction
    cv_input = scan_patient(folder_path) 
    current_findings_text = format_current_findings(cv_input)
    
    # 2. Queries Generation
    mistral_medium = ChatMistralAI(model="mistral-medium-latest", temperature=0.0)
    search_queries = generate_search_queries(cv_input, mistral_medium)
    
    # 3. RAG Retrieval
    retrieved_results = []
    for q in search_queries:
        retrieved_results.append(query_patient_data(patient_id=patientID, user_query=q))
    historical_text = format_historical_context(retrieved_results)

    # 4. Final Generation
    mistral_large = ChatMistralAI(model="mistral-large-latest", temperature=0.1)
    
    reports = []
    users_list = users if isinstance(users, Iterable) and not isinstance(users, str) else [users]
    
    for u in users_list:
        report_prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT.get(u, PROMPT["patient"])),
            ("human", "### PATIENT ID: {patient_id}\n### CURRENT EXAM FINDINGS:\n{current_findings}\n### PRIOR HISTORY:\n{historical_context}")
        ])
        chain = report_prompt | mistral_large
        res = chain.invoke({
            "patient_id": patientID,
            "current_findings": current_findings_text,
            "historical_context": historical_text
        })
        reports.append(res.content)
        
    return reports if len(reports) > 1 else reports[0]