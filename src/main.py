import os
from dotenv import load_dotenv
from src.rag_agent import generate_final_radiology_report

load_dotenv()

if __name__ == "__main__":
    # Paramètres de test
    PATIENT_ID = "17A76C2A"
    FOLDER_PATH = "data/raw/17A76C2A_17A76C2A/20210405_TC_T_RAX/17A76C2A 17A76C2A/26721665 TC TRAX"
    
    try:
        final_report = generate_final_radiology_report(
            patientID=PATIENT_ID, 
            folder_path=FOLDER_PATH, 
            users=["patient", "radiologist"]
        )
        
        print("\n" + "="*50)
        print("RAPPORT PATIENT :")
        print("="*50)
        print(final_report[0])
        
        print("\n" + "="*50)
        print("RAPPORT RADIOLOGUE :")
        print("="*50)
        print(final_report[1])
        
    except Exception as e:
        print(f"Erreur lors de l'exécution du pipeline: {e}")