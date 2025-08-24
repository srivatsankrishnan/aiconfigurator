from __future__ import annotations

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, Any, List, Tuple

from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk import common
from aiconfigurator.sdk.inference_session import DisaggInferenceSession, InferenceSession


def _to_enum(enum_cls, value_or_name):
    """
    Convert a string or enum value to the corresponding enum instance.
    If value_or_name is already an enum instance, return it as-is.
    """
    if isinstance(value_or_name, enum_cls):
        return value_or_name
    if isinstance(value_or_name, str):
        return enum_cls[value_or_name]
    return enum_cls(value_or_name)


def predict_ifb_single(
    *,
    model_name: str,
    system: str,
    backend: str = common.BackendName.trtllm.value,
    version: str = "0.20.0",
    # runtime
    isl: int,
    osl: int,
    batch_size: int,
    ctx_tokens: int,
    # parallel config
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    moe_tp: int = 1,
    moe_ep: int = 1,
    # quantization
    gemm_quant_mode: common.GEMMQuantMode | str = common.GEMMQuantMode.float16,
    moe_quant_mode: common.MoEQuantMode | str = common.MoEQuantMode.float16,
    kvcache_quant_mode: common.KVCacheQuantMode | str = common.KVCacheQuantMode.float16,
    fmha_quant_mode: common.FMHAQuantMode | str = common.FMHAQuantMode.float16,
    comm_quant_mode: common.CommQuantMode | str = common.CommQuantMode.half,
    # features
    nextn: int = 0,
    nextn_accept_rates: Optional[list[float]] = None,
    # advanced model options
    overwrite_num_layers: int = 0,
) -> Dict[str, Any]:
    """
    Predict metrics for a single IFB configuration using the existing backend math.

    Inputs must be single values; this function does not sweep or plot.

    Returns a dict containing at least: ttft_ms, tpot_ms, tokens_per_s_per_gpu, tokens_per_s_per_user.
    """
    # Build database and backend
    database = get_database(system=system, backend=backend, version=version)
    backend_impl = get_backend(backend)

    # Build model configuration
    mc = ModelConfig(
        tp_size=tp,
        pp_size=pp,
        attention_dp_size=dp,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        gemm_quant_mode=_to_enum(common.GEMMQuantMode, gemm_quant_mode),
        moe_quant_mode=_to_enum(common.MoEQuantMode, moe_quant_mode),
        kvcache_quant_mode=_to_enum(common.KVCacheQuantMode, kvcache_quant_mode),
        fmha_quant_mode=_to_enum(common.FMHAQuantMode, fmha_quant_mode),
        comm_quant_mode=_to_enum(common.CommQuantMode, comm_quant_mode),
        nextn=nextn,
        nextn_accept_rates=nextn_accept_rates,
        overwrite_num_layers=overwrite_num_layers,
    )
    model = get_model(model_name=model_name, model_config=mc)

    # Runtime configuration
    rc = RuntimeConfig(batch_size=batch_size, isl=isl, osl=osl)

    # Run IFB once for this single setup (no sweeps)
    summary = backend_impl.run_ifb(model=model, database=database, runtime_config=rc, ctx_tokens=ctx_tokens)
    df = summary.get_summary_df()
    if df is None or df.empty:
        return {
            "oom": summary.check_oom(),
        }

    row = df.iloc[0]
    result = {
        "ttft_ms": float(row["ttft"]),
        "tpot_ms": float(row["tpot"]),
        "tokens_per_s_per_gpu": float(row["tokens/s/gpu"]),
        "tokens_per_s_per_user": float(row["tokens/s/user"]),
        # Helpful extras
        "seq_per_s_per_gpu": float(row["seq/s/gpu"]),
        "tokens_per_s_total": float(row["tokens/s"]),
        "seq_per_s_total": float(row["seq/s"]),
        "concurrency": float(row["concurrency"]),
        "num_total_gpus": int(row["num_total_gpus"]),
        "global_batch_size": int(row["global_bs"]),
        "batch_size": int(row["bs"]),
        "backend": str(row["backend"]),
        "version": str(row["version"]),
        "system": str(row["system"]),
        "oom": bool(summary.check_oom()),
    }
    return result


def predict_ifb_from_sla_single_parallel(
    *,
    model_name: str,
    system: str,
    backend: str = common.BackendName.trtllm.value,
    version: str = "0.20.0",
    # runtime
    isl: int,
    osl: int,
    ttft_ms: float,
    tpot_ms: float,
    # parallel config (fixed)
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    moe_tp: int = 1,
    moe_ep: int = 1,
    # quantization
    gemm_quant_mode: common.GEMMQuantMode | str = common.GEMMQuantMode.float16,
    moe_quant_mode: common.MoEQuantMode | str = common.MoEQuantMode.float16,
    kvcache_quant_mode: common.KVCacheQuantMode | str = common.KVCacheQuantMode.float16,
    fmha_quant_mode: common.FMHAQuantMode | str = common.FMHAQuantMode.float16,
    comm_quant_mode: common.CommQuantMode | str = common.CommQuantMode.half,
    # sweep limits (kept modest)
    top_k: int = 1,
    max_batch_size: int = 512,
    ctx_stride: int = 512,
    # features
    nextn: int = 0,
    nextn_accept_rates: Optional[list[float]] = None,
    # advanced model options
    overwrite_num_layers: int = 0,
) -> Dict[str, Any]:
    """
    Predict metrics that meet a given SLA (ttft, tpot) for a fixed parallel config.
    This wraps the backend's constraint search but returns only core metrics.
    """
    # Build database and backend
    database = get_database(system=system, backend=backend, version=version)
    backend_impl = get_backend(backend)

    # Build model configuration
    mc = ModelConfig(
        tp_size=tp,
        pp_size=pp,
        attention_dp_size=dp,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        gemm_quant_mode=_to_enum(common.GEMMQuantMode, gemm_quant_mode),
        moe_quant_mode=_to_enum(common.MoEQuantMode, moe_quant_mode),
        kvcache_quant_mode=_to_enum(common.KVCacheQuantMode, kvcache_quant_mode),
        fmha_quant_mode=_to_enum(common.FMHAQuantMode, fmha_quant_mode),
        comm_quant_mode=_to_enum(common.CommQuantMode, comm_quant_mode),
        nextn=nextn,
        nextn_accept_rates=nextn_accept_rates,
        overwrite_num_layers=overwrite_num_layers,
    )
    model = get_model(model_name=model_name, model_config=mc)

    # SLA Runtime configuration
    rc = RuntimeConfig(isl=isl, osl=osl, ttft=ttft_ms, tpot=tpot_ms)

    summary = backend_impl.find_best_ifb_result_under_constraints(
        model=model,
        database=database,
        runtime_config=rc,
        top_k=top_k,
        max_batch_size=max_batch_size,
        ctx_stride=ctx_stride,
    )
    df = summary.get_summary_df()
    if df is None or df.empty:
        return {
            "oom": summary.check_oom(),
        }

    row = df.iloc[0]
    return {
        "ttft_ms": float(row["ttft"]),
        "tpot_ms": float(row["tpot"]),
        "tokens_per_s_per_gpu": float(row["tokens/s/gpu"]),
        "tokens_per_s_per_user": float(row["tokens/s/user"]),
        "seq_per_s_per_gpu": float(row["seq/s/gpu"]),
        "tokens_per_s_total": float(row["tokens/s"]),
        "seq_per_s_total": float(row["seq/s"]),
        "concurrency": float(row["concurrency"]),
        "num_total_gpus": int(row["num_total_gpus"]),
        "global_batch_size": int(row["global_bs"]),
        "batch_size": int(row["bs"]),
        "backend": str(row["backend"]),
        "version": str(row["version"]),
        "system": str(row["system"]),
        "oom": bool(summary.check_oom()),
    }


def predict_disagg_single(
    *,
    model_name: str,
    system: str,
    backend: str = common.BackendName.trtllm.value,
    version: str = "0.20.0",
    # runtime
    isl: int,
    osl: int,
    # prefill worker config
    p_tp: int, p_pp: int, p_dp: int,
    p_bs: int, p_workers: int,
    # decode worker config
    d_tp: int, d_pp: int, d_dp: int,
    d_bs: int, d_workers: int,
    # quantization (can be same for both)
    gemm_quant_mode: common.GEMMQuantMode | str = common.GEMMQuantMode.float16,
    moe_quant_mode: common.MoEQuantMode | str = common.MoEQuantMode.float16,
    kvcache_quant_mode: common.KVCacheQuantMode | str = common.KVCacheQuantMode.float16,
    fmha_quant_mode: common.FMHAQuantMode | str = common.FMHAQuantMode.float16,
    comm_quant_mode: common.CommQuantMode | str = common.CommQuantMode.half,
    # features
    nextn: int = 0,
    nextn_accept_rates: Optional[list[float]] = None,
    overwrite_num_layers: int = 0,
    # correction scales
    prefill_correction_scale: float = 1.0,
    decode_correction_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Predict metrics for a single disaggregated configuration (explicit prefill/decode workers).
    Returns core metrics from the composed system.
    """
    # Databases and backends
    prefill_db = get_database(system=system, backend=backend, version=version)
    decode_db = get_database(system=system, backend=backend, version=version)
    prefill_backend = get_backend(backend)
    decode_backend = get_backend(backend)

    # Model configs
    p_mc = ModelConfig(
        tp_size=p_tp, pp_size=p_pp, attention_dp_size=p_dp,
        gemm_quant_mode=_to_enum(common.GEMMQuantMode, gemm_quant_mode),
        moe_quant_mode=_to_enum(common.MoEQuantMode, moe_quant_mode),
        kvcache_quant_mode=_to_enum(common.KVCacheQuantMode, kvcache_quant_mode),
        fmha_quant_mode=_to_enum(common.FMHAQuantMode, fmha_quant_mode),
        comm_quant_mode=_to_enum(common.CommQuantMode, comm_quant_mode),
        moe_tp_size=1, moe_ep_size=1,
        nextn=nextn, nextn_accept_rates=nextn_accept_rates,
        overwrite_num_layers=overwrite_num_layers,
    )
    d_mc = ModelConfig(
        tp_size=d_tp, pp_size=d_pp, attention_dp_size=d_dp,
        gemm_quant_mode=_to_enum(common.GEMMQuantMode, gemm_quant_mode),
        moe_quant_mode=_to_enum(common.MoEQuantMode, moe_quant_mode),
        kvcache_quant_mode=_to_enum(common.KVCacheQuantMode, kvcache_quant_mode),
        fmha_quant_mode=_to_enum(common.FMHAQuantMode, fmha_quant_mode),
        comm_quant_mode=_to_enum(common.CommQuantMode, comm_quant_mode),
        moe_tp_size=1, moe_ep_size=1,
        nextn=nextn, nextn_accept_rates=nextn_accept_rates,
        overwrite_num_layers=overwrite_num_layers,
    )

    # Runtime
    rc_prefill = RuntimeConfig(batch_size=p_bs, isl=isl, osl=osl)
    rc_decode = RuntimeConfig(batch_size=d_bs, isl=isl, osl=osl)

    # Run static prefill and decode
    prefill_model = get_model(model_name, p_mc)
    decode_model = get_model(model_name, d_mc)
    prefill_sess = InferenceSession(prefill_model, prefill_db, prefill_backend)
    decode_sess = InferenceSession(decode_model, decode_db, decode_backend)

    prefill_summary = prefill_sess.run_static(mode='static_ctx', runtime_config=rc_prefill)
    decode_summary = decode_sess.run_static(mode='static_gen', runtime_config=rc_decode)

    if prefill_summary.check_oom() or decode_summary.check_oom():
        return {"oom": True}

    p_df = prefill_summary.get_summary_df()
    d_df = decode_summary.get_summary_df()
    if p_df is None or p_df.empty or d_df is None or d_df.empty:
        return {"oom": True}

    p = p_df.iloc[0]
    d = d_df.iloc[0]

    # Compose disagg metrics consistent with DisaggInferenceSession
    seq_s_prefill = float(p['seq/s']) * p_workers * prefill_correction_scale
    seq_s_decode = float(d['seq/s']) * d_workers * decode_correction_scale
    seq_s = min(seq_s_prefill, seq_s_decode)

    prefill_gpus = int(p['pp']) * int(p['tp']) * int(p['dp'])
    decode_gpus = int(d['pp']) * int(d['tp']) * int(d['dp'])
    total_gpus = prefill_gpus * p_workers + decode_gpus * d_workers

    seq_s_gpu = seq_s / total_gpus if total_gpus > 0 else 0.0

    ttft = float(p['ttft'])
    tpot = float(d['tpot'])

    tokens_s = seq_s * osl
    tokens_s_gpu = tokens_s / total_gpus if total_gpus > 0 else 0.0
    tokens_s_user = float(d['tokens/s/user'])

    concurrency = float(d['concurrency']) * d_workers

    return {
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "tokens_per_s_per_gpu": tokens_s_gpu,
        "tokens_per_s_per_user": tokens_s_user,
        "seq_per_s_per_gpu": seq_s_gpu,
        "tokens_per_s_total": tokens_s,
        "seq_per_s_total": seq_s,
        "concurrency": concurrency,
        "num_total_gpus": int(total_gpus),
        "prefill_workers": int(p_workers),
        "decode_workers": int(d_workers),
        "prefill_bs": int(p['bs']),
        "decode_bs": int(d['bs']),
        "oom": False,
    }


def predict_disagg_from_sla(
    *,
    model_name: str,
    system: str,
    backend: str = common.BackendName.trtllm.value,
    version: str = "0.20.0",
    # runtime
    isl: int,
    osl: int,
    ttft_ms: float,
    tpot_ms: float,
    # prefill search space (parallel configs)
    prefill_parallel_config_list: List[Tuple[int, int, int, int, int]],
    # decode search space (parallel configs)
    decode_parallel_config_list: List[Tuple[int, int, int, int, int]],
    # worker and replica constraints
    prefill_max_num_tokens: int = 16384,
    decode_max_num_tokens: int = 512,
    prefill_num_worker_list: Optional[List[int]] = None,
    decode_num_worker_list: Optional[List[int]] = None,
    num_gpu_list: Optional[List[int]] = None,
    # quantization
    gemm_quant_mode: common.GEMMQuantMode | str = common.GEMMQuantMode.float16,
    moe_quant_mode: common.MoEQuantMode | str = common.MoEQuantMode.float16,
    kvcache_quant_mode: common.KVCacheQuantMode | str = common.KVCacheQuantMode.float16,
    fmha_quant_mode: common.FMHAQuantMode | str = common.FMHAQuantMode.float16,
    comm_quant_mode: common.CommQuantMode | str = common.CommQuantMode.half,
    # features
    nextn: int = 0,
    nextn_accept_rates: Optional[list[float]] = None,
    overwrite_num_layers: int = 0,
    # correction scales
    prefill_correction_scale: float = 1.0,
    decode_correction_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    SLA-based disaggregated predictor using DisaggInferenceSession.find_best_disagg_result_under_constraints.
    """
    prefill_db = get_database(system=system, backend=backend, version=version)
    decode_db = get_database(system=system, backend=backend, version=version)
    prefill_backend = get_backend(backend)
    decode_backend = get_backend(backend)

    p_mc = ModelConfig(
        gemm_quant_mode=_to_enum(common.GEMMQuantMode, gemm_quant_mode),
        moe_quant_mode=_to_enum(common.MoEQuantMode, moe_quant_mode),
        kvcache_quant_mode=_to_enum(common.KVCacheQuantMode, kvcache_quant_mode),
        fmha_quant_mode=_to_enum(common.FMHAQuantMode, fmha_quant_mode),
        comm_quant_mode=_to_enum(common.CommQuantMode, comm_quant_mode),
        nextn=nextn, nextn_accept_rates=nextn_accept_rates, overwrite_num_layers=overwrite_num_layers,
    )
    d_mc = ModelConfig(
        gemm_quant_mode=_to_enum(common.GEMMQuantMode, gemm_quant_mode),
        moe_quant_mode=_to_enum(common.MoEQuantMode, moe_quant_mode),
        kvcache_quant_mode=_to_enum(common.KVCacheQuantMode, kvcache_quant_mode),
        fmha_quant_mode=_to_enum(common.FMHAQuantMode, fmha_quant_mode),
        comm_quant_mode=_to_enum(common.CommQuantMode, comm_quant_mode),
        nextn=nextn, nextn_accept_rates=nextn_accept_rates, overwrite_num_layers=overwrite_num_layers,
    )

    rc = RuntimeConfig(isl=isl, osl=osl, ttft=ttft_ms, tpot=tpot_ms)

    sess = DisaggInferenceSession(prefill_db, prefill_backend, decode_db, decode_backend)
    sess.set_correction_scales(prefill_correction_scale, decode_correction_scale)

    summary = sess.find_best_disagg_result_under_constraints(
        model_name=model_name,
        runtime_config=rc,
        prefill_model_config=p_mc,
        prefill_parallel_config_list=prefill_parallel_config_list,
        prefill_max_num_tokens=prefill_max_num_tokens,
        prefill_num_worker_list=prefill_num_worker_list,
        decode_model_config=d_mc,
        decode_parallel_config_list=decode_parallel_config_list,
        decode_max_num_tokens=decode_max_num_tokens,
        decode_num_worker_list=decode_num_worker_list,
        num_gpu_list=num_gpu_list,
    )

    df = summary.get_summary_df() if summary is not None else None
    if df is None or df.empty:
        return {"oom": True}

    row = df.iloc[0]
    return {
        "ttft_ms": float(row["ttft"]),
        "tpot_ms": float(row["tpot"]),
        "tokens_per_s_per_gpu": float(row["tokens/s/gpu"]),
        "tokens_per_s_per_user": float(row["tokens/s/user"]),
        "seq_per_s_per_gpu": float(row["seq/s/gpu"]),
        "tokens_per_s_total": float(row["tokens/s"]),
        "seq_per_s_total": float(row["seq/s"]),
        "concurrency": float(row["concurrency"]),
        "num_total_gpus": int(row["num_total_gpus"]),
        "prefill_workers": int(row["(p)workers"]),
        "decode_workers": int(row["(d)workers"]),
        "prefill_bs": int(row["(p)bs"]),
        "decode_bs": int(row["(d)bs"]),
        "oom": False,
    } 