from aiconfigurator.sdk.simple_predictor import predict_disagg_single

# Disagg example matching Rank-1 configuration from docs
# Model: QWEN3_32B
# System: h200_sxm, Backend: trtllm, Version: 1.0.0rc3
# ISL:OSL = 4000:500
# Prefill: 4 workers, tp1 pp1 dp1, bs=1 (1 GPU/worker)
# Decode:  1 worker,  tp4 pp1 dp1, bs=60 (4 GPUs/worker)
# gpus/replica = 4*1 + 1*4 = 8

out = predict_disagg_single(
    model_name="QWEN3_32B",
    system="h200_sxm",
    backend="trtllm",
    version="1.0.0rc3",
    isl=4000,
    osl=500,
    p_tp=1, p_pp=1, p_dp=1,
    p_bs=1, p_workers=4,
    d_tp=4, d_pp=1, d_dp=1,
    d_bs=60, d_workers=1,
    gemm_quant_mode="fp8",
    kvcache_quant_mode="fp8",
    fmha_quant_mode="fp8",
    moe_quant_mode="float16",
    comm_quant_mode="half",
)
print(out)

# Optional replicas for 512 GPUs
replicas = 512 // out["num_total_gpus"]
print({"replicas": replicas, "total_gpus_used": replicas * out["num_total_gpus"]}) 