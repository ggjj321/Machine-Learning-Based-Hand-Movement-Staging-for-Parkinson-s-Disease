#!/usr/bin/env python3
"""
手部視角分類器
使用 MediaPipe 判斷手部旋轉影片的拍攝視角（從上往下 vs 水平）
"""

import cv2
import mediapipe as mp
import os
import shutil
from pathlib import Path
import numpy as np


class HandViewClassifier:
    def __init__(self, input_folder, output_base_folder):
        """
        初始化分類器
        
        Args:
            input_folder: 包含影片的資料夾路徑
            output_base_folder: 輸出分類結果的基礎資料夾
        """
        self.input_folder = Path(input_folder)
        self.output_base_folder = Path(output_base_folder)
        
        # 創建輸出資料夾
        self.top_down_folder = self.output_base_folder / "top_down_view"
        self.horizontal_folder = self.output_base_folder / "horizontal_view"
        self.top_down_folder.mkdir(parents=True, exist_ok=True)
        self.horizontal_folder.mkdir(parents=True, exist_ok=True)
        
        # 初始化 MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def calculate_hand_orientation(self, landmarks):
        """
        判斷中指(12)和手心(0)誰在上方
        Args:
            landmarks: MediaPipe 手部關鍵點
        Returns:
            bool: True 表示水平側拍(中指在上)，False 表示由上往下(手心在上)
        """
        middle_tip_y = landmarks[12].y
        palm_y = landmarks[0].y
        # y值小者在上方
        return middle_tip_y < palm_y
    
    def classify_view(self, video_path, sample_frames=10):
        """
        分類影片的拍攝視角
        基於前幾個 frame 判斷：
        - 從上往下：初始手部應該是手背（palm down）
        - 水平：初始手部應該是手心（palm up）
        
        Args:
            video_path: 影片檔案路徑
            sample_frames: 取樣的幀數（使用前幾幀）
            
        Returns:
            str: 'top_down' 或 'horizontal'
        """
        cap = cv2.VideoCapture(str(video_path))
        
        horizontal_count = 0  # 水平側拍次數(中指在上)
        top_down_count = 0    # 從上往下次數(手心在上)
        total_detected = 0
        frame_idx = 0
        max_frames_to_check = sample_frames * 3
        while cap.isOpened() and frame_idx < max_frames_to_check and total_detected < sample_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                is_horizontal = self.calculate_hand_orientation(landmarks)
                if is_horizontal:
                    horizontal_count += 1
                else:
                    top_down_count += 1
                total_detected += 1
            frame_idx += 1
        cap.release()
        if total_detected == 0:
            return None
        horizontal_ratio = horizontal_count / total_detected
        print(f"  檢測到 {total_detected} 個有效幀")
        print(f"  水平側拍: {horizontal_count} 次 ({horizontal_ratio*100:.1f}%)")
        print(f"  從上往下: {top_down_count} 次 ({(1-horizontal_ratio)*100:.1f}%)")
        # 判斷邏輯：大多數幀中指在上則水平，否則從上往下
        if horizontal_ratio > 0.5:
            return 'horizontal'
        else:
            return 'top_down'
    
    def process_videos(self):
        """
        處理資料夾中所有包含「旋轉」的影片
        """
        # 獲取所有包含「旋轉」的影片檔案
        video_files = [f for f in self.input_folder.glob("*.mp4") if "旋轉" in f.name]
        
        print(f"找到 {len(video_files)} 個包含「旋轉」的影片檔案")
        print(f"開始處理...\n")
        
        results = {
            'top_down': [],
            'horizontal': [],
            'failed': []
        }
        
        for idx, video_file in enumerate(video_files, 1):
            print(f"[{idx}/{len(video_files)}] 處理: {video_file.name}")
            
            try:
                view_type = self.classify_view(video_file)
                
                if view_type == 'top_down':
                    output_path = self.top_down_folder / video_file.name
                    shutil.copy2(video_file, output_path)
                    results['top_down'].append(video_file.name)
                    print(f"  ✓ 分類為: 從上往下視角\n")
                    
                elif view_type == 'horizontal':
                    output_path = self.horizontal_folder / video_file.name
                    shutil.copy2(video_file, output_path)
                    results['horizontal'].append(video_file.name)
                    print(f"  ✓ 分類為: 水平視角\n")
                    
                else:
                    results['failed'].append(video_file.name)
                    print(f"  ✗ 無法檢測到手部\n")
                    
            except Exception as e:
                results['failed'].append(video_file.name)
                print(f"  ✗ 處理失敗: {e}\n")
        
        # 輸出統計結果
        print("=" * 60)
        print("處理完成！")
        print(f"從上往下視角: {len(results['top_down'])} 個影片")
        print(f"水平視角: {len(results['horizontal'])} 個影片")
        print(f"處理失敗: {len(results['failed'])} 個影片")
        print("=" * 60)
        
        if results['failed']:
            print("\n處理失敗的檔案:")
            for filename in results['failed']:
                print(f"  - {filename}")
    
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    # 設定路徑
    dates = ["2025-07-07_to_2025-07-11", "2025-07-14_to_2025-07-18", "2025-07-21_to_2025-07-25", "2025-07-28_to_2025-08-01", "2025-08-04_to_2025-08-08", "2025-08-11_to_2025-08-15", "2025-08-18_to_2025-08-22", "2025-08-25_to_2025-08-29"]

    for date in dates:
        input_folder = f"right_hand_files_{date}"
        output_folder = f"classified_hand_videos"
        
        # 創建分類器並處理影片
        classifier = HandViewClassifier(input_folder, output_folder)
        classifier.process_videos()


if __name__ == "__main__":
    main()

