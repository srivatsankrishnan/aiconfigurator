# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd


def get_pareto_front(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    x = df[x_col].values.copy()
    y = df[y_col].values.copy()
    if not maximize_x:
        x = -x
    if not maximize_y:
        y = -y

    order = np.argsort(-x)
    mask = np.zeros(len(x), dtype=bool)
    best_y = -np.inf
    for idx in order:
        if y[idx] > best_y:
            mask[idx] = True
            best_y = y[idx]

    result = df.loc[mask].copy()
    return result.sort_values(x_col).reset_index(drop=True)


def filter_feasible(
    df: pd.DataFrame,
    feasibility_col: str = "feasibility",
    threshold: float = 0.5,
) -> pd.DataFrame:
    return df.loc[df[feasibility_col] >= threshold].reset_index(drop=True)
