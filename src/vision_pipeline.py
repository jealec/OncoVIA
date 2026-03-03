import os
import re
import json
import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
from pathlib import Path
from dcm_seg_nodules import extract_seg 

LOBE_FILES = {
    "lung_upper_lobe_left.nii.gz": "Lobe supérieur gauche",
    "lung_lower_lobe_left.nii.gz": "Lobe inférieur gauche",
    "lung_upper_lobe_right.nii.gz": "Lobe supérieur droit",
    "lung_middle_lobe_right.nii.gz": "Lobe moyen droit",
    "lung_lower_lobe_right.nii.gz": "Lobe inférieur droit",
}
PERICARDIUM_FILE = "pericardium.nii.gz"

def run_totalsegmentator(input_dicom_dir, output_dir):
    out_path = Path(output_dir)
    if out_path.exists() and any(out_path.iterdir()):
        return 
    
    from totalsegmentator.python_api import totalsegmentator
    totalsegmentator(
        input=str(input_dicom_dir),
        output=str(output_dir),
        device="cpu", 
        fast=True,
    )

def load_tumor_mask(seg_path):
    ds = pydicom.dcmread(str(seg_path))
    pixel_data = ds.pixel_array
    
    segments_info = {}
    for seg in ds.SegmentSequence:
        segments_info[seg.SegmentNumber] = {"label": getattr(seg, "SegmentLabel", f"Segment_{seg.SegmentNumber}"), "frames": []}

    for frame_idx, fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
        seg_num = fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        z_pos = float(fg.PlanePositionSequence[0].ImagePositionPatient[2])
        segments_info[seg_num]["frames"].append((frame_idx, z_pos))

    fg0 = ds.PerFrameFunctionalGroupsSequence[0]
    orientation = fg0.PlaneOrientationSequence[0].ImageOrientationPatient
    pixel_spacing = fg0.PixelMeasuresSequence[0].PixelSpacing
    origin = fg0.PlanePositionSequence[0].ImagePositionPatient

    row_cosines = np.array([float(orientation[0]), float(orientation[1]), float(orientation[2])])
    col_cosines = np.array([float(orientation[3]), float(orientation[4]), float(orientation[5])])

    all_z = sorted(list(set(round(z, 4) for info in segments_info.values() for _, z in info["frames"])))
    slice_spacing = abs(all_z[1] - all_z[0]) if len(all_z) > 1 else float(getattr(fg0.PixelMeasuresSequence[0], "SliceThickness", 1.25))

    affine = np.eye(4)
    affine[0:3, 0] = row_cosines * float(pixel_spacing[1])
    affine[0:3, 1] = col_cosines * float(pixel_spacing[0])
    affine[0:3, 2] = np.cross(row_cosines, col_cosines) * slice_spacing
    affine[0:3, 3] = [float(origin[0]), float(origin[1]), float(all_z[0])]

    z_to_idx = {round(z, 4): i for i, z in enumerate(all_z)}
    
    tumor_volumes = {}
    for seg_num, info in segments_info.items():
        vol = np.zeros((ds.Rows, ds.Columns, len(all_z)), dtype=np.uint8)
        for frame_idx, z_pos in info["frames"]:
            vol[:, :, z_to_idx[round(z_pos, 4)]] = pixel_data[frame_idx]
        tumor_volumes[info["label"]] = vol

    return tumor_volumes, affine, all_z

def resample_mask_to_target(source_data, source_affine, target_affine, target_shape):
    transform = np.linalg.inv(source_affine) @ target_affine
    coords = np.mgrid[0:target_shape[0], 0:target_shape[1], 0:target_shape[2]].reshape(3, -1)
    source_coords = (transform @ np.vstack([coords, np.ones(coords.shape[1])]))[:3]
    source_idx = np.round(source_coords).astype(int)

    valid = ((source_idx[0] >= 0) & (source_idx[0] < source_data.shape[0]) &
             (source_idx[1] >= 0) & (source_idx[1] < source_data.shape[1]) &
             (source_idx[2] >= 0) & (source_idx[2] < source_data.shape[2]))

    result = np.zeros(np.prod(target_shape), dtype=np.uint8)
    result[valid] = source_data[source_idx[0, valid], source_idx[1, valid], source_idx[2, valid]]
    return result.reshape(target_shape)

def load_anatomical_masks(totalseg_dir, tumor_affine, tumor_shape):
    anatomy_volumes = {}
    files_to_load = dict(LOBE_FILES)
    files_to_load[PERICARDIUM_FILE] = "Péricarde"
    
    for filename, label in files_to_load.items():
        filepath = Path(totalseg_dir) / filename
        if filepath.exists():
            nii = nib.load(str(filepath))
            resampled = resample_mask_to_target(nii.get_fdata().astype(np.uint8), nii.affine, tumor_affine, tumor_shape)
            anatomy_volumes[label] = resampled
    return anatomy_volumes

def analyze_tumor_locations(tumor_volumes, anatomy_volumes, affine, pixel_spacing_mm, tumor_diameters=None):
    if tumor_diameters is None:
        tumor_diameters = {}
        
    voxel_volume_cm3 = np.prod(pixel_spacing_mm) / 1000.0
    results = []

    for tumor_name, tumor_vol in tumor_volumes.items():
        tumor_mask = tumor_vol > 0
        tumor_voxels = np.sum(tumor_mask)
        if tumor_voxels == 0: continue

        lobe_overlaps = {lobe: np.sum(tumor_mask & (vol > 0)) for lobe, vol in anatomy_volumes.items() if "Péricarde" not in lobe and np.sum(tumor_mask & (vol > 0)) > 0}
        main_lobe = max(lobe_overlaps, key=lobe_overlaps.get) if lobe_overlaps else "Hors lobes"
        
        pericardium_overlap = np.sum(tumor_mask & (anatomy_volumes.get("Péricarde", np.zeros(1)) > 0))
        pericardium_pct = (pericardium_overlap / tumor_voxels * 100) if tumor_voxels > 0 else 0

        lobes_desc = [f"{lobe} ({count / tumor_voxels * 100:.1f}%)" for lobe, count in sorted(lobe_overlaps.items(), key=lambda x: -x[1])]
        diameter = tumor_diameters.get(tumor_name, "Inconnu")

        results.append({
            "Tumeur": tumor_name,
            "Diametre_mm": diameter, 
            "Volume_cm3": round(tumor_voxels * voxel_volume_cm3, 2),
            "Localisation_principale": main_lobe,
            "Lobes_touches": " + ".join(lobes_desc) if lobes_desc else "Aucun",
            "Chevauche_plusieurs_lobes": "Oui" if len(lobe_overlaps) > 1 else "Non",
            "Contact_pericarde": "Oui" if pericardium_overlap > 0 else "Non",
            "pct_tumeur_sur_pericarde": round(pericardium_pct, 1),
        })
    return pd.DataFrame(results)

def scan_patient(patient_path_str):
    """Scans a single patient directory (used for current exam evaluation)"""
    patient_path = Path(patient_path_str)
    global_imaging_data = {}
    print(f"--> Scanning Patient: {patient_path.name}")
    for accession_dir in patient_path.rglob("CT*"):
        if accession_dir.is_dir():
            try:
                seg_path_obj, seg_summary = extract_seg(str(accession_dir), output_dir="results")
                acc_match = re.search(r"Accession Number:\s*(\d+)", seg_summary)
                
                if acc_match:
                    acc = acc_match.group(1)
                    if acc in global_imaging_data:
                        continue
                        
                    date_match = re.search(r"(\d{8})", str(accession_dir))
                    extracted_diameters = []
                    for line in seg_summary.splitlines():
                        diam_match = re.match(r"-\s*(.+?):\s*diameter\s*([\d.]+)\s*mm", line.strip(), re.IGNORECASE)
                        if diam_match:
                            extracted_diameters.append(float(diam_match.group(2)))
                    
                    totalseg_out = accession_dir / "totalseg_output"
                    run_totalsegmentator(str(accession_dir), str(totalseg_out))
                    
                    tumor_vols, affine, z_positions = load_tumor_mask(str(seg_path_obj))
                    anatomy_vols = load_anatomical_masks(str(totalseg_out), affine, list(tumor_vols.values())[0].shape)
                    
                    tumor_diameters = {}
                    tumor_keys = sorted(list(tumor_vols.keys()))
                    for i, d in enumerate(extracted_diameters):
                        if i < len(tumor_keys):
                            tumor_diameters[tumor_keys[i]] = d
                    
                    fg0 = pydicom.dcmread(str(seg_path_obj)).PerFrameFunctionalGroupsSequence[0]
                    pm = fg0.PixelMeasuresSequence[0]
                    slice_spacing = abs(z_positions[1] - z_positions[0]) if len(z_positions) > 1 else 1.25
                    spacing_mm = [float(pm.PixelSpacing[0]), float(pm.PixelSpacing[1]), slice_spacing]
                    
                    df_loc = analyze_tumor_locations(tumor_vols, anatomy_vols, affine, spacing_mm, tumor_diameters)
                    
                    global_imaging_data[acc] = {
                        "date": date_match.group(1) if date_match else "UNKNOWN",
                        "anatomical_locations": df_loc.to_json(orient="records") if not df_loc.empty else "[]",
                        "seg_file_path": str(seg_path_obj),
                        "tumor_diameters": tumor_diameters 
                    }
            except Exception as e:
                print(f"   [ERROR] Failed processing {accession_dir.name}: {e}")
                    
    return global_imaging_data

def scan_all_patients(dataset_root_path, checkpoint_file="imaging_checkpoint.json"):
    """Scans the entire dataset for DB construction"""
    dataset_root = Path(dataset_root_path)
    global_imaging_data = {}
    
    if os.path.exists(checkpoint_file):
        print(f"--> Loading existing progress from {checkpoint_file}...")
        try:
            with open(checkpoint_file, 'r') as f:``
                global_imaging_data = json.load(f)
        except json.JSONDecodeError:
            print("   [WARNING] Checkpoint file corrupted. Starting fresh.")

    patient_folders = [f for f in dataset_root.iterdir() if f.is_dir()]

    for patient_path in patient_folders:
        patient_updated = False
        patient_data = scan_patient(str(patient_path))
        
        for acc, data in patient_data.items():
            if acc not in global_imaging_data:
                global_imaging_data[acc] = data
                patient_updated = True
                print(f"   [OK] Accession {acc} fully processed.")
        
        if patient_updated:
            with open(checkpoint_file, 'w') as f:
                json.dump(global_imaging_data, f, indent=4)
            print(f"   [SAVE] Checkpoint updated to disk for {patient_path.name}.")
                    
    return global_imaging_data