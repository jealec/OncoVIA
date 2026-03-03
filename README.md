# Oncovia AI 🫁 
**Multimodal RAG Pipeline for Oncology (Computer Vision + LLM Agents)**

**Live Interface / Front-end:** [https://oncovia-ai-flow.lovable.app/](https://oncovia-ai-flow.lovable.app/)

> ⚠️ **Disclaimer:** This project is a Proof of Concept (POC) developed during a hackathon. Due to strict medical confidentiality (HIPAA/GDPR), all patient data, DICOM files, and medical reports shown in the interface or used in this public repository are **100% synthetic/fake**. No real patient data is exposed.

##  Overview
Oncovia AI is an end-to-end medical pipeline that bridges 3D spatial data (CT scans) and longitudinal clinical history (radiology reports) to automate RECIST 1.1 tumor tracking. 

Instead of relying solely on text, our system extracts quantitative data directly from medical imaging and uses an **Agentic RAG architecture** powered by **Mistral AI** to generate highly accurate, patient-specific follow-up reports.

## ⚙️ How it Works (The Pipeline)

1. **3D Vision Extraction (Computer Vision):** Processes DICOM files using `TotalSegmentator` to generate anatomical masks. It automatically extracts tumor locations (e.g., *Right Upper Lobe*), volumes (cm³), and diameters (mm) with pre-trained models and our custom logic.
2. **Medical Vector Database:** Historical patient reports are embedded using `NeuML/pubmedbert-base-embeddings` and stored securely in a local `ChromaDB`, strictly filtered by Patient ID.
3. **Agentic Retrieval (`mistral-medium`):** An AI agent takes the raw JSON anatomical data from the current scan and dynamically generates natural language queries to retrieve relevant historical lesions from the vector DB.
4. **Dual-Target Synthesis (`mistral-large`):** The system synthesizes the extracted visual data with the retrieved historical text to generate two distinct reports:
   - 🩺 **For the Radiologist:** A highly technical, RECIST 1.1 compliant report.
   - 🫂 **For the Patient:** An empathetic, vulgarized, and non-alarmist summary. 

## 🛠️ Tech Stack
* **LLMs:** Mistral API (`mistral-medium-latest`, `mistral-large-latest`)
* **Framework:** LangChain
* **Vector DB:** ChromaDB
* **Computer Vision:** `pydicom`, `nibabel`, `TotalSegmentator`
* **Embeddings:** HuggingFace (`pubmedbert`)
* **Frontend:** Lovable

## Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/your-username/oncovia-ai.git](https://github.com/your-username/oncovia-ai.git)
cd oncovia-ai
pip install -r requirements.txt
pip install dcm_seg_nodules-1.0.0-py3-none-any.whl # for the tumor extractor 
```
#### How the GE Healthcare extractor works
```bash
from dcm_seg_nodules import extract_seg
seg_path = extract_seg("input/PATIENT_FOLDER", output_dir="results")
print(f"SEG saved to: {seg_path}")
```


### 2. Environment Variables
Create a `.env` file at the root of the project and add your Mistral API key:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 3. Run the Pipeline
*(Note: You must add your own synthetic DICOM data in a `data/` folder to run the pipeline locally).*

```bash
python src/main.py
```

##  Team & Collaborators
**🥇 1st Place Winners - Unboxed Hackathon**
* [Ahmed Loughzali](https://www.linkedin.com/in/ahmed-loughzali-15a0b7257/)
* [Ayoub Tarek](https://www.linkedin.com/in/ayoub-tarek-5283b8320/)  
* [Gabriel Cheval](https://www.linkedin.com/in/gabriel-cheval-49ab4130b/) 
* [Jeanne Leclerc](https://www.linkedin.com/in/jeanne-leclerc16/) 
* [Adrien Schumacher](https://www.linkedin.com/in/adrien-schumacher-021710331/) 

## 🤝 Acknowledgments
This project was built during the intense 30-hour **Unboxed Hackathon** hosted at **Centrale Lyon**. 

A special thanks to the incredible partners who guided the MedTech challenges and made this project possible: **GE HealthCare** for their clinical expertise and trust, **Mistral AI** for providing the powerful models that run our reasoning engine, and **Lovable** for the frontend support. 

## 🔮 Perspectives
* **Containerization:** Dockerize the end-to-end pipeline for plug-and-play deployment in clinical IT environments.
* **On-Premise Deployment:** Transition to running local models to guarantee 100% data privacy and comply with strict medical regulations, removing the need for external internet access.
* **UI Integration:** Package the Lovable frontend and the Python backend into a unified deployable service.