#!/usr/bin/env python3
"""
將手部旋轉影片轉換為骨架序列並依照 PD 階段分類
支援處理所有病患資料、特定抽出 stage 0 (缺乏階段資料) 的病患，或處理未在病歷資料表中的獨立影片。
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import torch
import argparse
from pathlib import Path
from datetime import datetime
import re


class VideoToSkeletonProcessor:
    def __init__(self, csv_path, video_base_folder, output_base_folder, mode='all'):
        """
        初始化處理器
        
        Args:
            csv_path: CSV 檔案路徑
            video_base_folder: 包含 horizontal_view 和 top_down_view 的基礎資料夾
            output_base_folder: 輸出骨架序列的基礎資料夾
            mode: 'all' (處理所有), 'stage0_only' (僅處理PD為空)
        """
        self.csv_path = Path(csv_path) if csv_path else None
        self.video_base_folder = Path(video_base_folder) if video_base_folder else None
        self.output_base_folder = Path(output_base_folder) if output_base_folder else None
        self.mode = mode
        
        # 初始化 MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 僅在非 individual 模式讀取病人資料
        if self.mode != 'individual':
            self.patient_data = self._load_patient_data()
        else:
            self.patient_data = {}
        
    def _load_patient_data(self):
        """讀取 CSV 檔案並建立病人資料對應表"""
        if not self.csv_path or not self.csv_path.exists():
            print(f"錯誤: 找不到 CSV 檔案 {self.csv_path}")
            return {}
            
        print(f"讀取 CSV 檔案 ({self.csv_path})...")
        
        # 讀取 CSV
        df = pd.read_csv(self.csv_path, encoding='utf-8')
        
        # 建立對應字典：(日期, 收案號) -> PD階段
        patient_dict = {}
        
        for _, row in df.iterrows():
            date = row['日期']
            case_num = row['收案號']
            pd_stage = row['PD 階段']
            
            # 處理日期和收案號
            if pd.notna(date) and pd.notna(case_num):
                try:
                    # 處理日期格式 (2025/03/03 -> 20250303)
                    date = str(date).strip()
                    if '/' in date:
                        date_obj = datetime.strptime(date, '%Y/%m/%d')
                    elif '-' in date:
                        date_obj = datetime.strptime(date, '%Y-%m-%d')
                    else:
                        continue
                    
                    date_str = date_obj.strftime('%Y%m%d')
                    
                    # 處理收案號（移除小數點和前導零）
                    case_num_str = str(int(float(case_num)))
                    
                    # 儲存對應關係：(日期, 收案號) -> PD階段
                    key = (date_str, case_num_str)
                    
                    # 根據模式決定要存入什麼資料
                    if self.mode == 'stage0_only':
                        # 只有無效/空白/'-' 才會納入，其餘跳過
                        if pd.isna(pd_stage) or pd_stage == '-' or str(pd_stage).strip() == '':
                            patient_dict[key] = 0
                        else:
                            continue
                    else:
                        patient_dict[key] = pd_stage
                    
                except Exception as e:
                    print(f"  警告: 無法解析資料: {e}")
                    continue
        
        desc = "PD 階段為空的病患資料" if self.mode == 'stage0_only' else "筆病患資料"
        print(f"載入 {len(patient_dict)} 筆{desc}")
        return patient_dict
    
    def _parse_video_filename(self, filename):
        """解析影片檔名，提取日期和收案號"""
        pattern = r'gesture_(\d{8})_(\d+)__'
        match = re.search(pattern, filename)
        
        if match:
            date_str = match.group(1)
            case_num_str = str(int(match.group(2)))
            return (date_str, case_num_str)
        
        return None
    
    def _extract_skeleton_sequence(self, video_path):
        """從影片中提取手部骨架序列"""
        cap = cv2.VideoCapture(str(video_path))
        skeleton_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 轉換為 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 處理圖像
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                frame_landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in landmarks]
                skeleton_sequence.append(frame_landmarks)
            else:
                skeleton_sequence.append(None)
            
        cap.release()
        return skeleton_sequence
    
    def process_videos(self):
        """處理所有影片並生成骨架序列檔案 (針對 all 和 stage0_only 模式)"""
        if self.mode == 'individual':
            print("錯誤: individual 模式請呼叫 process_individual_videos 方法")
            return
            
        views = ['horizontal_view', 'top_down_view']
        
        stats = {
            'total_videos': 0,
            'processed': 0,
            'skipped_or_no_data': 0,
            'no_skeleton': 0,
            'by_stage': {}
        }
        
        for view in views:
            view_folder = self.video_base_folder / view
            
            if not view_folder.exists():
                print(f"警告: 資料夾不存在 {view_folder}")
                continue
            
            video_files = list(view_folder.glob("*.mp4"))
            print(f"\n處理 {view} 資料夾，共 {len(video_files)} 個影片")
            
            for idx, video_file in enumerate(video_files, 1):
                stats['total_videos'] += 1
                print(f"\n[{idx}/{len(video_files)}] 處理: {video_file.name}")
                
                parsed = self._parse_video_filename(video_file.name)
                if parsed is None:
                    print(f"  ✗ 無法解析檔名")
                    stats['skipped_or_no_data'] += 1
                    continue
                
                date_str, case_num = parsed
                print(f"  日期: {date_str}, 收案號: {case_num}")
                
                key = (date_str, case_num)
                if key not in self.patient_data:
                    if self.mode == 'stage0_only':
                        print(f"  - 跳過（非 stage_0 或找不到資料）")
                    else:
                        print(f"  ✗ 找不到對應的病人資料")
                    stats['skipped_or_no_data'] += 1
                    continue
                
                pd_stage = self.patient_data[key]
                
                if pd.isna(pd_stage) or pd_stage == '-' or str(pd_stage).strip() == '':
                    if self.mode == 'all':
                        print(f"  ! PD 階段資料空白或無效 ({pd_stage})，歸類為 stage 0")
                    pd_stage = 0
                
                try:
                    pd_stage = int(float(pd_stage))
                    pd_stage_str = f"stage_{pd_stage}"
                except:
                    print(f"  ✗ PD 階段格式錯誤: {pd_stage}")
                    stats['skipped_or_no_data'] += 1
                    continue
                
                print(f"  PD 階段: {pd_stage}")
                print(f"  提取骨架序列...")
                
                skeleton_sequence = self._extract_skeleton_sequence(video_file)
                valid_frames = sum(1 for frame in skeleton_sequence if frame is not None)
                print(f"  總幀數: {len(skeleton_sequence)}, 有效骨架: {valid_frames}")
                
                if valid_frames == 0:
                    print(f"  ✗ 沒有檢測到任何手部骨架")
                    stats['no_skeleton'] += 1
                    continue
                
                output_folder = self.output_base_folder / view / pd_stage_str
                output_folder.mkdir(parents=True, exist_ok=True)
                
                output_filename = video_file.stem + '.pt'
                output_path = output_folder / output_filename
                
                skeleton_array = []
                for frame in skeleton_sequence:
                    if frame is not None:
                        skeleton_array.append(frame)
                    else:
                        skeleton_array.append([[0.0, 0.0, 0.0]] * 21)
                
                skeleton_array = np.array(skeleton_array, dtype=np.float32)
                
                skeleton_data = {
                    'video_name': video_file.name,
                    'date': date_str,
                    'case_number': case_num,
                    'pd_stage': pd_stage,
                    'view': view,
                    'total_frames': len(skeleton_sequence),
                    'valid_frames': valid_frames,
                    'skeleton_sequence': torch.from_numpy(skeleton_array)
                }
                
                torch.save(skeleton_data, output_path)
                print(f"  ✓ 已儲存至: {output_path}")
                stats['processed'] += 1
                
                if pd_stage_str not in stats['by_stage']:
                    stats['by_stage'][pd_stage_str] = 0
                stats['by_stage'][pd_stage_str] += 1
        
        print("\n" + "=" * 70)
        print("處理完成！統計資訊：")
        print("=" * 70)
        print(f"總影片數: {stats['total_videos']}")
        print(f"成功處理: {stats['processed']}")
        print(f"跳過或無病患資料: {stats['skipped_or_no_data']}")
        print(f"無法提取骨架: {stats['no_skeleton']}")
        print(f"\n各 PD 階段分布:")
        for stage, count in sorted(stats['by_stage'].items()):
            print(f"  {stage}: {count} 個影片")
        print("=" * 70)
    
    def process_individual_videos(self, input_folder, output_folder):
        """處理個別影片資料夾中的影片，不需對照病人資料表"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        if not input_path.exists():
            print(f"錯誤: 輸入資料夾不存在 {input_path}")
            return
            
        output_path.mkdir(parents=True, exist_ok=True)
        video_files = list(input_path.glob("*.mp4"))
        print(f"\n處理個別影片資料夾: {input_path}")
        print(f"共 {len(video_files)} 個影片")
        
        processed_count = 0
        
        for idx, video_file in enumerate(video_files, 1):
            print(f"\n[{idx}/{len(video_files)}] 處理: {video_file.name}")
            
            parsed = self._parse_video_filename(video_file.name)
            if parsed:
                date_str, case_num = parsed
                print(f"  解析資訊 - 日期: {date_str}, 收案號: {case_num}")
            else:
                date_str, case_num = "unknown", "unknown"
                print(f"  無法解析檔名資訊")
            
            print(f"  提取骨架序列...")
            skeleton_sequence = self._extract_skeleton_sequence(video_file)
            
            valid_frames = sum(1 for frame in skeleton_sequence if frame is not None)
            print(f"  總幀數: {len(skeleton_sequence)}, 有效骨架: {valid_frames}")
            
            if valid_frames == 0:
                print(f"  ✗ 沒有檢測到任何手部骨架")
                continue
            
            output_filename = video_file.stem + '.pt'
            output_file_path = output_path / output_filename
            
            skeleton_array = []
            for frame in skeleton_sequence:
                if frame is not None:
                    skeleton_array.append(frame)
                else:
                    skeleton_array.append([[0.0, 0.0, 0.0]] * 21)
            
            skeleton_array = np.array(skeleton_array, dtype=np.float32)
            
            skeleton_data = {
                'video_name': video_file.name,
                'date': date_str,
                'case_number': case_num,
                'pd_stage': -1, 
                'view': 'individual',
                'total_frames': len(skeleton_sequence),
                'valid_frames': valid_frames,
                'skeleton_sequence': torch.from_numpy(skeleton_array)
            }
            
            torch.save(skeleton_data, output_file_path)
            print(f"  ✓ 已儲存至: {output_file_path}")
            processed_count += 1
            
        print(f"\n個別影片處理完成! 成功處理: {processed_count}/{len(video_files)}")

    def __del__(self):
        """清理資源"""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    parser = argparse.ArgumentParser(description="將手部旋轉影片轉換為骨架序列並依照 PD 階段分類")
    
    parser.add_argument('--mode', type=str, choices=['all', 'stage0_only', 'individual'], default='all',
                        help="執行模式。'all': 處理所有資料 (預設); 'stage0_only': 僅處理未分期(stage0)個案; 'individual': 解析個人測試無病歷影片。")
    parser.add_argument('--csv', type=str, default="收案_CAREs 20251009-加密 - deID.csv",
                        help="包含 PD 階段資料的 CSV 檔案路徑")
    parser.add_argument('--input_dir', type=str, default="classified_hand_videos",
                        help="已經藉由 classify_hand_view.py 分類好的基礎目錄 (僅適用於 all / stage0_only)")
    parser.add_argument('--output_dir', type=str, default="skeleton_sequences",
                        help="萃取後資料輸出的基礎目錄 (僅適用於 all / stage0_only)")
    
    parser.add_argument('--individual_in', type=str, default="Individual_handling_video",
                        help="要強制轉骨架的特定影片目錄 (僅適用 mode=individual)")
    parser.add_argument('--individual_out', type=str, default="Individual_handling_video_output",
                        help="特定影片輸出的 .pt 目錄 (僅適用 mode=individual)")
    
    args = parser.parse_args()

    print(f"啟動程式 | 模式: {args.mode}")

    if args.mode in ['all', 'stage0_only']:
        # 對應的 processor 初始化
        processor = VideoToSkeletonProcessor(args.csv, args.input_dir, args.output_dir, mode=args.mode)
        processor.process_videos()
        
        print(f"\n骨架序列已儲存至: {args.output_dir}/")
    else:
        # individual 模式
        processor = VideoToSkeletonProcessor(None, None, None, mode=args.mode)
        processor.process_individual_videos(args.individual_in, args.individual_out)
        print(f"\n骨架序列已儲存至: {args.individual_out}/")


if __name__ == "__main__":
    main()
