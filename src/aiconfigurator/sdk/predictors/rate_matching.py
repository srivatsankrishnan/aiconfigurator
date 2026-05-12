# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd

PREFILL_DEGRADATION_FACTOR = 0.9
DECODE_DEGRADATION_FACTOR = 0.92
TTFT_CORRECTION_FACTOR = 1.8


def rate_match_batch(
    ttft_ms: np.ndarray,
    tpot_ms: np.ndarray,
    osl: np.ndarray,
    p_tp: np.ndarray,
    p_pp: np.ndarray,
    p_bs: np.ndarray,
    p_workers: np.ndarray,
    d_tp: np.ndarray,
    d_pp: np.ndarray,
    d_bs: np.ndarray,
    d_workers: np.ndarray,
    p_dp: np.ndarray | int = 1,
    d_dp: np.ndarray | int = 1,
    prefill_degradation: float = PREFILL_DEGRADATION_FACTOR,
    decode_degradation: float = DECODE_DEGRADATION_FACTOR,
    ttft_correction: float = TTFT_CORRECTION_FACTOR,
) -> pd.DataFrame:
    corrected_ttft = ttft_ms * ttft_correction
    decode_tokens = np.maximum(osl - 1, 0)
    prefill_time_s = corrected_ttft / 1000
    prefill_seq_s = np.where(prefill_time_s > 0, p_bs / prefill_time_s, 0.0)
    decode_time_s = tpot_ms * decode_tokens / 1000
    decode_seq_s = np.where(decode_time_s > 0, d_bs / decode_time_s, 0.0)
    seq_s = np.minimum(
        prefill_seq_s * p_workers * prefill_degradation,
        decode_seq_s * d_workers * decode_degradation,
    )
    num_total_gpus = p_tp * p_pp * p_dp * p_workers + d_tp * d_pp * d_dp * d_workers
    tokens_s = seq_s * osl
    tokens_s_gpu = np.where(num_total_gpus > 0, tokens_s / num_total_gpus, 0.0)
    seq_s_gpu = np.where(num_total_gpus > 0, seq_s / num_total_gpus, 0.0)
    request_latency = ttft_ms + tpot_ms * decode_tokens
    tokens_s_user = np.where(tpot_ms > 0, 1000.0 / tpot_ms, 0.0)
    concurrency = d_bs * d_workers
    prefill_seq_s_worker = np.where(prefill_time_s > 0, p_bs / prefill_time_s, 0.0)
    decode_seq_s_worker = np.where(tpot_ms > 0, 1000.0 / tpot_ms, 0.0)

    return pd.DataFrame({
        "ttft": ttft_ms,
        "tpot": tpot_ms,
        "request_latency": request_latency,
        "seq/s": seq_s,
        "seq/s/gpu": seq_s_gpu,
        "tokens/s": tokens_s,
        "tokens/s/gpu": tokens_s_gpu,
        "tokens/s/user": tokens_s_user,
        "(p)seq/s/worker": prefill_seq_s_worker,
        "(d)seq/s/worker": decode_seq_s_worker,
        "num_total_gpus": num_total_gpus,
        "concurrency": concurrency,
        "(p)tp": p_tp,
        "(p)pp": p_pp,
        "(p)dp": p_dp,
        "(p)bs": p_bs,
        "(p)workers": p_workers,
        "(d)tp": d_tp,
        "(d)pp": d_pp,
        "(d)dp": d_dp,
        "(d)bs": d_bs,
        "(d)workers": d_workers,
    })
