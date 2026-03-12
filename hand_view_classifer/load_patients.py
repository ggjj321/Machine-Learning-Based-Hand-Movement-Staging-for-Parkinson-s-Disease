#!/usr/bin/env python3
"""
Script to load CSV file and skeleton sequences to create PDPatient objects.
Reads patient data from CSV and matches with corresponding left/right hand skeleton sequences.
"""

import os
import re
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from patient_structure import PDPatient


class PatientLoader:
    """Loads patient data from CSV and skeleton sequences to create PDPatient objects."""
    
    def __init__(self, csv_path: str, skeleton_base_dir: str):
        """
        Initialize the PatientLoader.
        
        Args:
            csv_path: Path to the CSV file containing patient information
            skeleton_base_dir: Base directory containing skeleton_sequences folder
        """
        self.csv_path = csv_path
        self.skeleton_base_dir = Path(skeleton_base_dir)
        self.patient_data = self._load_csv()
        
    def _load_csv(self) -> pd.DataFrame:
        """Load and preprocess the CSV file."""
        df = pd.read_csv(self.csv_path, encoding='utf-8')
        
        # Convert date format from "2025/03/03" to "20250303"
        df['date_str'] = pd.to_datetime(df['日期'], format='%Y/%m/%d', errors='coerce').dt.strftime('%Y%m%d')
        
        # Normalize case number (remove leading zeros and decimals)
        df['case_num_str'] = df['收案號'].apply(lambda x: str(int(float(x))) if pd.notna(x) else '')
        
        # Handle medication status (藥效: 0=off, 1=on)
        df['on_medication'] = df['藥效'].apply(lambda x: bool(x) if pd.notna(x) and x != '-' else False)
        
        return df
    
    def _parse_filename(self, filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse video filename to extract date, case number, and hand type.
        
        Args:
            filename: Filename like "2025-04-18 14:36:55_gesture_20250418_142843__128_左手旋轉_REC_xxx.pt"
            
        Returns:
            Tuple of (date_str, case_num_str, hand_type) where hand_type is '左手' or '右手'
        """
        # Extract date and case number using regex
        match = re.search(r'gesture_(\d{8})_(\d+)__', filename)
        if not match:
            return None, None, None
        
        date_str = match.group(1)  # e.g., "20250418"
        case_num = match.group(2)   # e.g., "142843"
        case_num_str = str(int(case_num))  # Remove leading zeros: "142843"
        
        # Extract hand type (左手 or 右手)
        if '左手' in filename:
            hand_type = '左手'
        elif '右手' in filename:
            hand_type = '右手'
        else:
            hand_type = None
            
        return date_str, case_num_str, hand_type
    
    def _find_skeleton_files(self) -> Dict[Tuple[str, str, str, str], str]:
        """
        Find all skeleton .pt files and organize them by (date, case_number, view, hand_type).
        
        Returns:
            Dictionary mapping (date_str, case_num_str, view, hand_type) -> file_path
        """
        skeleton_files = {}
        
        # Iterate through all views and stages
        for view in ['horizontal_view', 'top_down_view']:
            view_path = self.skeleton_base_dir / view
            if not view_path.exists():
                continue
                
            for stage_dir in view_path.iterdir():
                if not stage_dir.is_dir():
                    continue
                    
                for pt_file in stage_dir.glob('*.pt'):
                    date_str, case_num_str, hand_type = self._parse_filename(pt_file.name)
                    if date_str and case_num_str and hand_type:
                        key = (date_str, case_num_str, view, hand_type)
                        skeleton_files[key] = str(pt_file)
        
        return skeleton_files
    
    def load_patients(self, view: str = 'horizontal_view') -> List[PDPatient]:
        """
        Load all PDPatient objects for a specific view.
        
        Args:
            view: 'horizontal_view' or 'top_down_view'
            
        Returns:
            List of PDPatient objects
        """
        skeleton_files = self._find_skeleton_files()
        patients = []
        
        # Group by (date, case_number) to find matching left and right hand pairs
        patient_groups = {}
        
        for (date_str, case_num_str, file_view, hand_type), file_path in skeleton_files.items():
            if file_view != view:
                continue
                
            key = (date_str, case_num_str)
            if key not in patient_groups:
                patient_groups[key] = {}
            patient_groups[key][hand_type] = file_path
        
        # Create PDPatient objects for complete pairs (both left and right hands)
        for (date_str, case_num_str), hands in patient_groups.items():
            if '左手' not in hands or '右手' not in hands:
                print(f"Warning: Missing hand data for date={date_str}, case={case_num_str}")
                continue
            
            # Find matching patient data in CSV
            csv_match = self.patient_data[
                (self.patient_data['date_str'] == date_str) & 
                (self.patient_data['case_num_str'] == case_num_str)
            ]
            
            if csv_match.empty:
                print(f"Warning: No CSV data found for date={date_str}, case={case_num_str}")
                continue
            
            # Get patient information
            patient_row = csv_match.iloc[0]
            pd_stage = patient_row['PD 階段']
            
            # Handle missing or invalid PD stage
            if pd.isna(pd_stage) or pd_stage == '-':
                pd_stage = 0
            else:
                pd_stage = int(pd_stage)
            
            on_medication = patient_row['on_medication']
            patient_id = f"{date_str}_{case_num_str}"
            
            # Load skeleton data
            try:
                left_data = torch.load(hands['左手'])
                right_data = torch.load(hands['右手'])
                
                # Extract skeleton_sequence tensor from the saved dict
                left_trajectory = left_data['skeleton_sequence']
                right_trajectory = right_data['skeleton_sequence']
                
                # Create PDPatient object
                patient = PDPatient(
                    patient_id=patient_id,
                    pd_stage=pd_stage,
                    date=date_str,
                    on_medication=on_medication,
                    left_trajectory=left_trajectory,
                    right_trajectory=right_trajectory
                )
                
                patients.append(patient)
                
            except Exception as e:
                print(f"Error loading skeleton data for {patient_id}: {e}")
                continue
        
        return patients
    
    def load_all_patients(self) -> Dict[str, List[PDPatient]]:
        """
        Load all PDPatient objects for all views.
        
        Returns:
            Dictionary mapping view name -> list of PDPatient objects
        """
        return {
            'horizontal_view': self.load_patients('horizontal_view'),
            'top_down_view': self.load_patients('top_down_view')
        }


def main():
    """Example usage of the PatientLoader."""
    
    # Set paths
    csv_path = "收案_CAREs 20251009-加密 - deID.csv"
    skeleton_base_dir = "skeleton_sequences"
    
    # Create loader
    loader = PatientLoader(csv_path, skeleton_base_dir)
    
    # Load patients for horizontal view
    print("Loading patients for horizontal_view...")
    horizontal_patients = loader.load_patients('horizontal_view')
    print(f"Loaded {len(horizontal_patients)} patients for horizontal_view")
    
    # Load patients for top-down view
    print("\nLoading patients for top_down_view...")
    topdown_patients = loader.load_patients('top_down_view')
    print(f"Loaded {len(topdown_patients)} patients for top_down_view")
    
    # Show some examples
    if horizontal_patients:
        print("\n=== Example horizontal_view patients ===")
        for i, patient in enumerate(horizontal_patients[:3]):
            print(f"\n{i+1}. {patient}")
    
    if topdown_patients:
        print("\n=== Example top_down_view patients ===")
        for i, patient in enumerate(topdown_patients[:3]):
            print(f"\n{i+1}. {patient}")
    
    # Load all patients at once
    print("\n\n=== Loading all patients ===")
    all_patients = loader.load_all_patients()
    print(f"Total horizontal_view patients: {len(all_patients['horizontal_view'])}")
    print(f"Total top_down_view patients: {len(all_patients['top_down_view'])}")
    
    # Statistics by PD stage
    for view_name, patients in all_patients.items():
        print(f"\n{view_name} - PD Stage distribution:")
        stage_counts = {}
        for patient in patients:
            stage = patient.pd_stage
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        for stage in sorted(stage_counts.keys()):
            print(f"  Stage {stage}: {stage_counts[stage]} patients")


if __name__ == "__main__":
    main()
