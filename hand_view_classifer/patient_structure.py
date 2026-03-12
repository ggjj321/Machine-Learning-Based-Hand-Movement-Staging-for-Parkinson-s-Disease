import os
import re
import pandas as pd
from typing import List, Optional
import torch

class PDPatient:
    """
    用於存放帕金森病（PD）患者數據的類別，包括PD分期、用藥狀態，以及左右手軌跡。
    左右手軌跡應為通過torch.load從.pt文件中讀取得到的torch.Tensor對象。
    """
    def __init__(
        self,
        patient_id: str,
        pd_stage: int,
        date: str,
        on_medication: bool,
        left_trajectory: torch.Tensor,
        right_trajectory: torch.Tensor
    ):
        self.patient_id = patient_id            # 患者ID（可選）
        self.pd_stage = pd_stage                # PD分期（例如1~5）
        self.date = date                        # 日期
        self.on_medication = on_medication      # 是否正在服藥
        self.left_trajectory = left_trajectory  # 左手軌跡 tensor
        self.right_trajectory = right_trajectory# 右手軌跡 tensor

    @classmethod
    def from_pt(
        cls,
        patient_id: str,
        pd_stage: int,
        date: str,
        on_medication: bool,
        left_pt_path: str,
        right_pt_path: str
    ) -> "PDPatient":
        """
        從.pt文件載入左右手軌跡並創建PDPatient實例。
        :param left_pt_path: 左手軌跡的.pt文件路徑
        :param right_pt_path: 右手軌跡的.pt文件路徑
        """
        left = torch.load(left_pt_path)
        right = torch.load(right_pt_path)
        return cls(patient_id, pd_stage, date, on_medication, left, right)

    def __repr__(self) -> str:
        return (
            f"PDPatient(patient_id={self.patient_id!r}, pd_stage={self.pd_stage}, "
            f"on_medication={self.on_medication}, "
            f"left_shape={tuple(self.left_trajectory.shape)}, "
            f"right_shape={tuple(self.right_trajectory.shape)})"
        )
