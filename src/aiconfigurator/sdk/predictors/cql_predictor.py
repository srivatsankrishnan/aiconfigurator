# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from aiconfigurator.sdk import common
from aiconfigurator.sdk.predictors.rate_matching import rate_match_batch

logger = logging.getLogger(__name__)

FAMILY_PREFIXES = {
    "meta-llama": "LLAMA",
    "Qwen": "QWEN",
    "deepseek-ai": "DEEPSEEK",
    "mistralai": "MIXTRAL",
}

# ---------------------------------------------------------------------------
# Feature encoding (self-contained, mirrors arch_features.py)
# ---------------------------------------------------------------------------

def _log_norm(x: float) -> float:
    return math.log1p(max(0, x or 0))


def _safe_ratio(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


@lru_cache(maxsize=64)
def _get_model_config_dict(model_path: str) -> dict:
    from aiconfigurator.sdk.utils import get_model_config_from_model_path

    parsed = get_model_config_from_model_path(model_path)
    keys = [
        "family", "layers", "n", "n_kv", "d",
        "hidden_size", "inter_size", "vocab", "context",
        "topk", "num_experts", "moe_inter_size", "extra_params",
    ]
    return {k: (parsed.get(k, 0) if parsed.get(k) is not None else 0) for k in keys}


@lru_cache(maxsize=64)
def _get_model_arch_features(model_path: str) -> np.ndarray:
    cfg = _get_model_config_dict(model_path)
    layers = cfg.get("layers", 0)
    hidden_size = cfg.get("hidden_size", 0)
    num_heads = cfg.get("n", 0)
    num_kv_heads = cfg.get("n_kv", 0)
    head_dim = cfg.get("d", 0)
    inter_size = cfg.get("inter_size", 0)
    vocab_size = cfg.get("vocab", 0)
    context_len = cfg.get("context", 0)
    num_experts = cfg.get("num_experts", 0)
    topk = cfg.get("topk", 0)
    moe_inter_size = cfg.get("moe_inter_size", 0)

    is_moe = 1.0 if num_experts > 0 else 0.0
    gqa_ratio = _safe_ratio(num_kv_heads, num_heads)
    ffn_ratio = _safe_ratio(inter_size, hidden_size)
    params_per_layer = (4 * hidden_size * hidden_size + 2 * hidden_size * inter_size) / 1e9
    total_dense_params = layers * params_per_layer

    return np.array([
        _log_norm(layers), _log_norm(hidden_size), _log_norm(num_heads),
        _log_norm(num_kv_heads), _log_norm(head_dim), _log_norm(inter_size),
        _log_norm(vocab_size), _log_norm(context_len), _log_norm(num_experts),
        _log_norm(topk), _log_norm(moe_inter_size), is_moe, gqa_ratio,
        ffn_ratio, _log_norm(total_dense_params),
    ], dtype=np.float32)


@lru_cache(maxsize=64)
def _get_quant_features(model_path: str) -> np.ndarray:
    from aiconfigurator.sdk import config as aic_config
    from aiconfigurator.sdk.models import get_model

    mc = aic_config.ModelConfig(tp_size=1, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=1)
    try:
        model = get_model(model_path, mc, "trtllm")
        c = model.config
        return np.array([
            c.gemm_quant_mode.value.memory, c.gemm_quant_mode.value.compute,
            c.kvcache_quant_mode.value.memory, c.kvcache_quant_mode.value.compute,
            c.fmha_quant_mode.value.memory, c.fmha_quant_mode.value.compute,
            c.moe_quant_mode.value.memory, c.moe_quant_mode.value.compute,
        ], dtype=np.float32)
    except Exception:
        return np.array([2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 1.0], dtype=np.float32)


@lru_cache(maxsize=16)
def _get_system_features(system: str, backend: str, version: str) -> np.ndarray:
    from aiconfigurator.sdk.perf_database import get_database

    db = get_database(system=system, backend=backend, version=version)
    if db is None:
        return np.zeros(9, dtype=np.float32)
    spec = db.system_spec
    gpu = spec.get("gpu", {})
    node = spec.get("node", {})
    return np.array([
        _log_norm(gpu.get("mem_bw", 0)), _log_norm(gpu.get("mem_capacity", 0)),
        _log_norm(gpu.get("float16_tc_flops", 0)),
        _log_norm(gpu.get("fp8_tc_flops", gpu.get("float16_tc_flops", 0))),
        _log_norm(gpu.get("power", 0)), _log_norm(gpu.get("sm_version", 0)),
        _log_norm(node.get("num_gpus_per_node", 0)),
        _log_norm(node.get("intra_node_bw", 0)),
        _log_norm(node.get("inter_node_bw", 0)),
    ], dtype=np.float32)


def _get_interaction_features(model_path: str, system: str, backend: str, version: str) -> np.ndarray:
    from aiconfigurator.sdk.perf_database import get_database

    cfg = _get_model_config_dict(model_path)
    quant = _get_quant_features(model_path)
    layers = cfg.get("layers", 0)
    hidden_size = cfg.get("hidden_size", 0)
    num_kv_heads = cfg.get("n_kv", 0)
    head_dim = cfg.get("d", 0)
    inter_size = cfg.get("inter_size", 0)
    num_experts = cfg.get("num_experts", 0)
    topk = cfg.get("topk", 0)

    params_per_layer = (4 * hidden_size * hidden_size + 2 * hidden_size * inter_size) / 1e9
    total_dense_params = layers * params_per_layer
    gemm_mem, kv_mem = float(quant[0]), float(quant[2])

    fp8_flops, mem_bw = 0.0, 0.0
    db = get_database(system=system, backend=backend, version=version)
    if db is not None:
        gpu = db.system_spec.get("gpu", {})
        fp8_flops = gpu.get("fp8_tc_flops", gpu.get("float16_tc_flops", 0))
        mem_bw = gpu.get("mem_bw", 0)

    return np.array([
        gemm_mem, kv_mem,
        _log_norm(total_dense_params * gemm_mem),
        _log_norm(num_kv_heads * head_dim * kv_mem),
        _safe_ratio(topk, max(num_experts, 1)),
        _log_norm(_safe_ratio(fp8_flops, max(mem_bw, 1.0))),
    ], dtype=np.float32)


def _encode_state(model_path: str, system: str, backend: str, version: str,
                  isl: int, osl: int, feature_version: int) -> np.ndarray:
    arch = _get_model_arch_features(model_path)
    quant = _get_quant_features(model_path)
    sys_feat = _get_system_features(system, backend, version)
    request = np.array([_log_norm(isl), _log_norm(osl)], dtype=np.float32)
    if feature_version == 1:
        return np.concatenate([arch, quant, sys_feat, request])
    interact = _get_interaction_features(model_path, system, backend, version)
    return np.concatenate([arch, quant, sys_feat, request, interact])


def _encode_action(p_tp, p_pp, p_bs, p_workers, d_tp, d_pp, d_bs, d_workers) -> np.ndarray:
    return np.array([
        _log_norm(p_tp), _log_norm(p_pp), _log_norm(p_bs), _log_norm(p_workers),
        _log_norm(d_tp), _log_norm(d_pp), _log_norm(d_bs), _log_norm(d_workers),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Neural network (inference-only, mirrors cql_agent.py network architectures)
# ---------------------------------------------------------------------------

def _build_residual_network(state_dim: int, action_dim: int,
                            hidden_dim: int = 512, num_blocks: int = 4,
                            dropout: float = 0.05, multi_head: bool = True):
    import torch.nn as nn

    def _residual_block(dim):
        return nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )

    def _output_head(dim):
        return nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2),
            nn.GELU(), nn.Linear(dim // 2, 1),
        )

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.GELU(),
            )
            self.blocks = nn.ModuleList([
                _residual_block(hidden_dim) for _ in range(num_blocks)
            ])
            if multi_head:
                self.ttft_head = _output_head(hidden_dim)
                self.tpot_head = _output_head(hidden_dim)
                self.feasibility_head = _output_head(hidden_dim)
                self.head = self.ttft_head
            else:
                self.head = _output_head(hidden_dim)

        def _encode(self, x):
            h = self.input_proj(x)
            for block in self.blocks:
                h = h + block(h)
            return h

        def forward(self, x):
            return self.head(self._encode(x))

        def forward_triple(self, x):
            h = self._encode(x)
            if hasattr(self, "ttft_head"):
                return self.ttft_head(h), self.tpot_head(h), self.feasibility_head(h)
            out = self.head(h)
            return out, out, out

    return _Net()


def _load_network(checkpoint_path: str | Path, device: str = "cpu"):
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dims = ckpt.get("dims", {})
    state_dim = ckpt.get("state_dim") or dims.get("state_dim")
    action_dim = ckpt.get("action_dim") or dims.get("action_dim")
    hidden_dim = ckpt.get("hidden_dim") or dims.get("hidden_dim", 512)
    num_blocks = ckpt.get("num_blocks") or dims.get("num_blocks", 4)
    net_state = ckpt.get("q_net_state_dict") or ckpt.get("model", {})

    multi_head = any(k.startswith("ttft_head") for k in net_state)
    net = _build_residual_network(state_dim, action_dim, hidden_dim, num_blocks,
                                  multi_head=multi_head)
    net.load_state_dict(net_state)
    net.to(device)
    net.eval()

    feature_version = 1 if state_dim == 34 else 2
    return net, state_dim, action_dim, feature_version, device


# ---------------------------------------------------------------------------
# CQLPredictor — public API
# ---------------------------------------------------------------------------

class CQLPredictor:
    def __init__(self, net, state_dim: int, action_dim: int,
                 feature_version: int, device: str,
                 calibration: dict | None = None) -> None:
        self._net = net
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._feature_version = feature_version
        self._device = device
        self._calibration = calibration or {}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str = "cpu") -> CQLPredictor:
        net, state_dim, action_dim, feat_ver, device = _load_network(checkpoint_path, device)
        cal = None
        cal_path = Path(checkpoint_path).parent / "calibration_params.json"
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            logger.info("Loaded CQL calibration from %s", cal_path)
        return cls(net, state_dim, action_dim, feat_ver, device, cal)

    def _get_family(self, model_path: str) -> str:
        prefix = model_path.split("/")[0]
        return FAMILY_PREFIXES.get(prefix, "OTHER")

    def _get_cal(self, model_path: str) -> dict:
        if not self._calibration:
            return {}
        if model_path in self._calibration:
            return self._calibration[model_path]
        family = self._get_family(model_path)
        family_key = f"_family_{family}"
        if family_key in self._calibration:
            return self._calibration[family_key]
        return self._calibration.get("_default", {})

    def _predict_log_batch(self, states: np.ndarray, actions: np.ndarray) -> dict:
        import torch

        with torch.no_grad():
            s = torch.as_tensor(states, dtype=torch.float32, device=self._device)
            a = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
            sa = torch.cat([s, a], dim=-1)
            ttft_log, tpot_log, feas_logit = self._net.forward_triple(sa)
            return {
                "ttft_log": ttft_log.squeeze(-1).cpu().numpy(),
                "tpot_log": tpot_log.squeeze(-1).cpu().numpy(),
                "feasibility": torch.sigmoid(feas_logit).squeeze(-1).cpu().numpy(),
            }

    def _calibrate_batch(self, raw_log: np.ndarray, scale: float, bias: float) -> np.ndarray:
        calibrated = scale * raw_log + bias
        return np.maximum(np.expm1(calibrated), 0.0)

    def predict_single(
        self, model_path: str, system: str, backend: str, version: str,
        isl: int, osl: int,
        p_tp: int, p_pp: int, p_bs: int, p_workers: int,
        d_tp: int, d_pp: int, d_bs: int, d_workers: int,
    ) -> dict:
        state = _encode_state(model_path, system, backend, version, isl, osl, self._feature_version)
        action = _encode_action(p_tp, p_pp, p_bs, p_workers, d_tp, d_pp, d_bs, d_workers)
        preds = self._predict_log_batch(state[np.newaxis], action[np.newaxis])
        cal = self._get_cal(model_path)
        if cal:
            ttft_arr = self._calibrate_batch(
                preds["ttft_log"], cal.get("ttft_scale", 1.0), cal.get("ttft_bias", 0.0),
            )
            tpot_arr = self._calibrate_batch(
                preds["tpot_log"], cal.get("tpot_scale", 1.0), cal.get("tpot_bias", 0.0),
            )
            ttft = float(ttft_arr[0])
            tpot = float(tpot_arr[0])
        else:
            ttft = float(max(np.expm1(preds["ttft_log"][0]), 0.0))
            tpot = float(max(np.expm1(preds["tpot_log"][0]), 0.0))
        feasibility = float(preds["feasibility"][0])
        return {"ttft": ttft, "tpot": tpot, "feasibility": feasibility}

    def predict_batch(
        self, model_path: str, system: str, backend: str, version: str,
        isl: int, osl: int, configs: list[dict],
    ) -> pd.DataFrame:
        n = len(configs)
        state = _encode_state(model_path, system, backend, version, isl, osl, self._feature_version)
        states = np.tile(state, (n, 1))
        actions = np.stack([
            _encode_action(
                c["p_tp"], c["p_pp"], c["p_bs"], c["p_workers"],
                c["d_tp"], c["d_pp"], c["d_bs"], c["d_workers"],
            ) for c in configs
        ])

        preds = self._predict_log_batch(states, actions)
        cal = self._get_cal(model_path)
        if cal:
            ttft_ms = self._calibrate_batch(preds["ttft_log"], cal.get("ttft_scale", 1.0), cal.get("ttft_bias", 0.0))
            tpot_ms = self._calibrate_batch(preds["tpot_log"], cal.get("tpot_scale", 1.0), cal.get("tpot_bias", 0.0))
        else:
            ttft_ms = np.expm1(np.maximum(preds["ttft_log"], 0))
            tpot_ms = np.expm1(np.maximum(preds["tpot_log"], 0))

        config_arrays = {
            "p_tp": np.array([c["p_tp"] for c in configs]),
            "p_pp": np.array([c["p_pp"] for c in configs]),
            "p_bs": np.array([c["p_bs"] for c in configs]),
            "p_workers": np.array([c["p_workers"] for c in configs]),
            "p_dp": np.array([c.get("p_dp", 1) for c in configs]),
            "d_tp": np.array([c["d_tp"] for c in configs]),
            "d_pp": np.array([c["d_pp"] for c in configs]),
            "d_bs": np.array([c["d_bs"] for c in configs]),
            "d_workers": np.array([c["d_workers"] for c in configs]),
            "d_dp": np.array([c.get("d_dp", 1) for c in configs]),
        }

        df = rate_match_batch(
            ttft_ms=ttft_ms, tpot_ms=tpot_ms,
            osl=np.full(n, osl), **config_arrays,
        )
        df["feasibility"] = preds["feasibility"]
        df["model"] = model_path
        df["isl"] = isl
        df["osl"] = osl

        for col in common.ColumnsDisagg:
            if col not in df.columns:
                df[col] = None

        return df

    def predict_single_as_summary(
        self, model_path: str, system: str, backend: str, version: str,
        isl: int, osl: int,
        p_tp: int, p_pp: int, p_bs: int, p_workers: int,
        d_tp: int, d_pp: int, d_bs: int, d_workers: int,
    ):
        from aiconfigurator.sdk.config import RuntimeConfig
        from aiconfigurator.sdk.inference_summary import InferenceSummary

        preds = self.predict_single(
            model_path, system, backend, version, isl, osl,
            p_tp, p_pp, p_bs, p_workers, d_tp, d_pp, d_bs, d_workers,
        )
        ttft, tpot = preds["ttft"], preds["tpot"]
        feasibility = preds["feasibility"]

        num_total_gpus = p_tp * p_pp * p_workers + d_tp * d_pp * d_workers
        request_latency = ttft + tpot * max(osl - 1, 0)
        tokens_s_user = 1000.0 / tpot if tpot > 0 else 0.0
        prefill_seq_s = 1000.0 / ttft if ttft > 0 else 0.0
        seq_s = min(prefill_seq_s * p_workers, tokens_s_user * d_workers)
        tokens_s = seq_s * osl
        tokens_s_gpu = tokens_s / num_total_gpus if num_total_gpus > 0 else 0.0
        seq_s_gpu = seq_s / num_total_gpus if num_total_gpus > 0 else 0.0

        row = dict.fromkeys(common.ColumnsDisagg)
        row.update({
            "model": model_path, "isl": isl, "osl": osl, "prefix": 0,
            "(p)bs": p_bs, "(p)global_bs": p_bs * p_workers, "(p)workers": p_workers,
            "(d)bs": d_bs, "(d)global_bs": d_bs * d_workers, "(d)workers": d_workers,
            "ttft": ttft, "tpot": tpot, "request_latency": request_latency,
            "seq/s": seq_s, "seq/s/gpu": seq_s_gpu,
            "tokens/s": tokens_s, "tokens/s/gpu": tokens_s_gpu,
            "tokens/s/user": tokens_s_user,
            "(p)seq/s/worker": prefill_seq_s, "(d)seq/s/worker": tokens_s_user,
            "num_total_gpus": num_total_gpus,
            "(p)tp": p_tp, "(p)pp": p_pp, "(p)dp": 1,
            "(d)tp": d_tp, "(d)pp": d_pp, "(d)dp": 1,
            "(p)backend": backend, "(p)version": version, "(p)system": system,
            "(d)backend": backend, "(d)version": version, "(d)system": system,
        })

        rc = RuntimeConfig(isl=isl, osl=osl)
        summary = InferenceSummary(rc)
        summary._summary_df = pd.DataFrame([row])
        summary._is_oom = feasibility < 0.5
        return summary
