# DeepSeek-V4.1-Flash on 8× MI355X: DP8/EP8 with AITER MegaMoE and EPLB

```bash
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_MORI_EP_V2=0 MORI_SHMEM_HEAP_SIZE=48G \
       VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=True AITER_TRITON_LOG_LEVEL=ERROR OMP_NUM_THREADS=1 \
       VLLM_ENGINE_READY_TIMEOUT_S=3600

vllm serve deepseek-ai/DeepSeek-V4.1-Flash --served-model-name dsv41 --port 8000 \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
  --moe-backend aiter_mega_moe --all2all-backend mori \
  --engram-config '{"cpu_offload":false,"embedding_across_dp":true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --attention-config '{"indexer_kv_dtype":"mxfp4"}' --block-size 128 \
  --max-num-batched-tokens 16384 --max-model-len 131072 --max-cudagraph-capture-size 1024 \
  --gpu-memory-utilization 0.85 --language-model-only \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --enable-eplb --eplb-config '{"window_size":1000,"step_interval":2000}'
```
