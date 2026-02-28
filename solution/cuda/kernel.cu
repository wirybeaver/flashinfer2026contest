/*
 * Track A CUDA implementation (correctness-first) with TVM FFI export.
 *
 * This implementation prioritizes semantic parity with the Triton reference:
 * - DeepSeek routing (sigmoid + bias, top2/group, top groups, top-k experts)
 * - FP8 block-scale dequant
 * - GEMM1 + SwiGLU + GEMM2 + weighted accumulation
 *
 * It is intentionally naive for now and serves as a baseline before optimization.
 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace {

using tvm::ffi::TensorView;

constexpr int kHiddenSize = 7168;
constexpr int kIntermediateSize = 2048;
constexpr int kNumExperts = 256;
constexpr int kTopK = 8;
constexpr int kNGroup = 8;
constexpr int kTopKGroup = 4;
constexpr int kBlockSize = 128;
constexpr int kLocalNumExperts = 32;
constexpr int kNumHiddenBlocks = kHiddenSize / kBlockSize;          // 56
constexpr int kNumIntermediateBlocks = kIntermediateSize / kBlockSize;  // 16
constexpr int kNumGemm1OutBlocks = (2 * kIntermediateSize) / kBlockSize;  // 32
constexpr int kExpertsPerGroup = kNumExperts / kNGroup;             // 32

__device__ __forceinline__ float sigmoidf_fast(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float sigmoid_accurate_dev(float x) {
  return 0.5f * tanhf(0.5f * x) + 0.5f;
}

__device__ __forceinline__ float siluf(float x) { return x * sigmoidf_fast(x); }

__device__ __forceinline__ float fp8e4m3_to_float(uint8_t v) {
  __nv_fp8_e4m3 x;
  x.__x = static_cast<__nv_fp8_storage_t>(v);
  return static_cast<float>(x);
}

__global__ void routing_deepseek_kernel(const float* logits, const __nv_bfloat16* bias,
                                        float routed_scale, int seq_len, int32_t* topk_idx_out,
                                        float* topk_w_out) {
  int t = blockIdx.x;
  int e = threadIdx.x;
  if (t >= seq_len || e >= kNumExperts) return;

  __shared__ float s_sigmoid[kNumExperts];
  __shared__ float s_bias[kNumExperts];
  __shared__ float s_group_score[kNGroup];
  __shared__ int s_group_top[kTopKGroup];
  __shared__ int s_topk_idx[kTopK];
  __shared__ float s_topk_w[kTopK];

  float logit = logits[static_cast<size_t>(t) * kNumExperts + e];
  float se = sigmoid_accurate_dev(logit);
  s_sigmoid[e] = se;
  s_bias[e] = se + __bfloat162float(bias[e]);
  __syncthreads();

  if (e < kNGroup) {
    int start = e * kExpertsPerGroup;
    float m1 = -INFINITY;
    float m2 = -INFINITY;
    for (int i = 0; i < kExpertsPerGroup; ++i) {
      float v = s_bias[start + i];
      if (v > m1) {
        m2 = m1;
        m1 = v;
      } else if (v > m2) {
        m2 = v;
      }
    }
    s_group_score[e] = m1 + m2;
  }
  __syncthreads();

  if (e == 0) {
    bool selected_group[kNGroup] = {false};
    for (int k = 0; k < kTopKGroup; ++k) {
      int best_g = -1;
      float best_v = -INFINITY;
      for (int g = 0; g < kNGroup; ++g) {
        if (selected_group[g]) continue;
        float v = s_group_score[g];
        if (v > best_v || (v == best_v && g < best_g)) {
          best_v = v;
          best_g = g;
        }
      }
      s_group_top[k] = best_g;
      selected_group[best_g] = true;
    }

    bool selected_exp[kNumExperts] = {false};
    for (int k = 0; k < kTopK; ++k) {
      int best_e = -1;
      float best_v = -INFINITY;
      for (int gi = 0; gi < kTopKGroup; ++gi) {
        int g = s_group_top[gi];
        int start = g * kExpertsPerGroup;
        for (int i = 0; i < kExpertsPerGroup; ++i) {
          int cand = start + i;
          if (selected_exp[cand]) continue;
          float v = s_bias[cand];
          if (v > best_v || (v == best_v && cand < best_e)) {
            best_v = v;
            best_e = cand;
          }
        }
      }
      s_topk_idx[k] = best_e;
      selected_exp[best_e] = true;
    }

    float sumw = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
      sumw += s_sigmoid[s_topk_idx[k]];
    }
    float inv = routed_scale / (sumw + 1e-20f);
    for (int k = 0; k < kTopK; ++k) {
      s_topk_w[k] = s_sigmoid[s_topk_idx[k]] * inv;
    }
  }
  __syncthreads();

  if (e < kTopK) {
    size_t off = static_cast<size_t>(t) * kTopK + e;
    topk_idx_out[off] = s_topk_idx[e];
    topk_w_out[off] = s_topk_w[e];
  }
}

__global__ void dequant_hidden_kernel(const uint8_t* hidden_fp8, const float* hidden_scale,
                                      float* hidden_f32, int seq_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * kHiddenSize;
  if (idx >= total) return;
  int t = idx / kHiddenSize;
  int c = idx % kHiddenSize;
  int cb = c / kBlockSize;
  float s = hidden_scale[cb * seq_len + t];
  hidden_f32[idx] = fp8e4m3_to_float(hidden_fp8[idx]) * s;
}

__global__ void dequant_weight_kernel(const uint8_t* w_fp8, const float* w_scale, float* w_f32,
                                      int rows, int cols, int num_col_blocks) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (idx >= total) return;
  int r = idx / cols;
  int c = idx % cols;
  int rb = r / kBlockSize;
  int cb = c / kBlockSize;
  float s = w_scale[rb * num_col_blocks + cb];
  w_f32[idx] = fp8e4m3_to_float(w_fp8[idx]) * s;
}

__global__ void gather_rows_kernel(const float* src, float* dst, const int32_t* token_idx, int rows,
                                   int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (idx >= total) return;
  int r = idx / cols;
  int c = idx % cols;
  int src_r = token_idx[r];
  dst[idx] = src[src_r * cols + c];
}

// C = A[M,K] * B[N,K]^T  (B stored row-major as [N,K])
__global__ void gemm_nt_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int TILE = 16;
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  double acc = 0.0;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    int a_col = k0 + threadIdx.x;
    int b_col = k0 + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (col < N && b_col < K) ? B[col * K + b_col] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
      acc += static_cast<double>(As[threadIdx.y][kk]) * static_cast<double>(Bs[kk][threadIdx.x]);
    }
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = static_cast<float>(acc);
}

__global__ void swiglu_kernel(const float* g1, float* c, int rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * kIntermediateSize;
  if (idx >= total) return;
  int r = idx / kIntermediateSize;
  int j = idx % kIntermediateSize;
  float x1 = g1[r * (2 * kIntermediateSize) + j];
  float x2 = g1[r * (2 * kIntermediateSize) + (kIntermediateSize + j)];
  c[idx] = siluf(x2) * x1;
}

__global__ void scatter_add_weighted_kernel(const float* o, float* accum, const int32_t* token_idx,
                                            const float* token_w, int rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * kHiddenSize;
  if (idx >= total) return;
  int r = idx / kHiddenSize;
  int h = idx % kHiddenSize;
  int t = token_idx[r];
  float w = token_w[r];
  // token_idx is unique within each expert pass, so this update is race-free.
  accum[t * kHiddenSize + h] += o[idx] * w;
}

__global__ void float_to_bf16_kernel(const float* in, __nv_bfloat16* out, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) out[idx] = __float2bfloat16(in[idx]);
}

void check_cuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError) << msg << ": " << cudaGetErrorString(err);
  }
}

void kernel(TensorView routing_logits, TensorView routing_bias, TensorView hidden_states,
            TensorView hidden_states_scale, TensorView gemm1_weights, TensorView gemm1_weights_scale,
            TensorView gemm2_weights, TensorView gemm2_weights_scale, int64_t local_expert_offset,
            double routed_scaling_factor, TensorView output) {
  TVM_FFI_ICHECK_EQ(routing_logits.ndim(), 2);
  TVM_FFI_ICHECK_EQ(routing_bias.ndim(), 1);
  TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2);
  TVM_FFI_ICHECK_EQ(hidden_states_scale.ndim(), 2);
  TVM_FFI_ICHECK_EQ(gemm1_weights.ndim(), 3);
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.ndim(), 3);
  TVM_FFI_ICHECK_EQ(gemm2_weights.ndim(), 3);
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale.ndim(), 3);
  TVM_FFI_ICHECK_EQ(output.ndim(), 2);

  int seq_len = static_cast<int>(routing_logits.size(0));
  TVM_FFI_ICHECK_EQ(routing_logits.size(1), kNumExperts);
  TVM_FFI_ICHECK_EQ(routing_bias.size(0), kNumExperts);
  TVM_FFI_ICHECK_EQ(hidden_states.size(0), seq_len);
  TVM_FFI_ICHECK_EQ(hidden_states.size(1), kHiddenSize);
  TVM_FFI_ICHECK_EQ(hidden_states_scale.size(0), kNumHiddenBlocks);
  TVM_FFI_ICHECK_EQ(hidden_states_scale.size(1), seq_len);
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(0), kLocalNumExperts);
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(1), 2 * kIntermediateSize);
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(2), kHiddenSize);
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(0), kLocalNumExperts);
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(1), kNumGemm1OutBlocks);
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(2), kNumHiddenBlocks);
  TVM_FFI_ICHECK_EQ(gemm2_weights.size(0), kLocalNumExperts);
  TVM_FFI_ICHECK_EQ(gemm2_weights.size(1), kHiddenSize);
  TVM_FFI_ICHECK_EQ(gemm2_weights.size(2), kIntermediateSize);
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(0), kLocalNumExperts);
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(1), kNumHiddenBlocks);
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(2), kNumIntermediateBlocks);
  TVM_FFI_ICHECK_EQ(output.size(0), seq_len);
  TVM_FFI_ICHECK_EQ(output.size(1), kHiddenSize);

  auto dev = routing_logits.device();
  cudaStream_t stream =
      reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  const float* d_routing_logits = static_cast<const float*>(routing_logits.data_ptr());
  const __nv_bfloat16* d_routing_bias = static_cast<const __nv_bfloat16*>(routing_bias.data_ptr());
  const uint8_t* d_hidden_states = static_cast<const uint8_t*>(hidden_states.data_ptr());
  const float* d_hidden_states_scale = static_cast<const float*>(hidden_states_scale.data_ptr());
  const uint8_t* d_gemm1_weights = static_cast<const uint8_t*>(gemm1_weights.data_ptr());
  const float* d_gemm1_weights_scale = static_cast<const float*>(gemm1_weights_scale.data_ptr());
  const uint8_t* d_gemm2_weights = static_cast<const uint8_t*>(gemm2_weights.data_ptr());
  const float* d_gemm2_weights_scale = static_cast<const float*>(gemm2_weights_scale.data_ptr());
  __nv_bfloat16* d_output = static_cast<__nv_bfloat16*>(output.data_ptr());

  int32_t* d_topk_idx = nullptr;
  float* d_topk_w = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_topk_idx),
                        static_cast<size_t>(seq_len) * kTopK * sizeof(int32_t)),
             "cudaMalloc topk_idx");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_topk_w),
                        static_cast<size_t>(seq_len) * kTopK * sizeof(float)),
             "cudaMalloc topk_w");
  routing_deepseek_kernel<<<seq_len, kNumExperts, 0, stream>>>(
      d_routing_logits, d_routing_bias, static_cast<float>(routed_scaling_factor), seq_len,
      d_topk_idx, d_topk_w);
  check_cuda(cudaGetLastError(), "launch routing_deepseek_kernel");
  check_cuda(cudaStreamSynchronize(stream), "sync after routing");

  std::vector<int32_t> h_topk_idx(static_cast<size_t>(seq_len) * kTopK);
  std::vector<float> h_topk_w(static_cast<size_t>(seq_len) * kTopK);
  check_cuda(cudaMemcpyAsync(h_topk_idx.data(), d_topk_idx,
                             h_topk_idx.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream),
             "copy topk_idx D2H");
  check_cuda(cudaMemcpyAsync(h_topk_w.data(), d_topk_w, h_topk_w.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, stream),
             "copy topk_w D2H");
  check_cuda(cudaStreamSynchronize(stream), "sync after topk copies");

  float* d_hidden_f32 = nullptr;
  float* d_accum = nullptr;
  float* d_w13 = nullptr;
  float* d_w2 = nullptr;
  float* d_a_expert = nullptr;
  float* d_g1 = nullptr;
  float* d_c = nullptr;
  float* d_o = nullptr;
  int32_t* d_token_idx = nullptr;
  float* d_token_w = nullptr;

  auto alloc_f32 = [&](float** p, size_t n) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(float)), "cudaMalloc float");
  };
  auto alloc_i32 = [&](int32_t** p, size_t n) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(int32_t)), "cudaMalloc int32");
  };

  alloc_f32(&d_hidden_f32, static_cast<size_t>(seq_len) * kHiddenSize);
  alloc_f32(&d_accum, static_cast<size_t>(seq_len) * kHiddenSize);
  alloc_f32(&d_w13, static_cast<size_t>(2 * kIntermediateSize) * kHiddenSize);
  alloc_f32(&d_w2, static_cast<size_t>(kHiddenSize) * kIntermediateSize);
  alloc_f32(&d_a_expert, static_cast<size_t>(seq_len) * kHiddenSize);
  alloc_f32(&d_g1, static_cast<size_t>(seq_len) * (2 * kIntermediateSize));
  alloc_f32(&d_c, static_cast<size_t>(seq_len) * kIntermediateSize);
  alloc_f32(&d_o, static_cast<size_t>(seq_len) * kHiddenSize);
  alloc_i32(&d_token_idx, static_cast<size_t>(seq_len));
  alloc_f32(&d_token_w, static_cast<size_t>(seq_len));

  check_cuda(cudaMemsetAsync(d_accum, 0, static_cast<size_t>(seq_len) * kHiddenSize * sizeof(float),
                             stream),
             "memset accum");

  {
    int total = seq_len * kHiddenSize;
    int blocks = (total + 255) / 256;
    dequant_hidden_kernel<<<blocks, 256, 0, stream>>>(d_hidden_states, d_hidden_states_scale,
                                                       d_hidden_f32, seq_len);
    check_cuda(cudaGetLastError(), "launch dequant_hidden_kernel");
  }

  std::vector<int32_t> h_tok_idx;
  std::vector<float> h_tok_w;
  h_tok_idx.reserve(seq_len);
  h_tok_w.reserve(seq_len);

  int local_start = static_cast<int>(local_expert_offset);
  for (int le = 0; le < kLocalNumExperts; ++le) {
    int ge = local_start + le;
    if (ge < 0 || ge >= kNumExperts) continue;

    h_tok_idx.clear();
    h_tok_w.clear();
    for (int t = 0; t < seq_len; ++t) {
      for (int k = 0; k < kTopK; ++k) {
        size_t off = static_cast<size_t>(t) * kTopK + k;
        if (h_topk_idx[off] == ge) {
          h_tok_idx.push_back(t);
          h_tok_w.push_back(h_topk_w[off]);
          break;
        }
      }
    }
    int m = static_cast<int>(h_tok_idx.size());
    if (m == 0) continue;

    check_cuda(cudaMemcpyAsync(d_token_idx, h_tok_idx.data(), m * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream),
               "copy token_idx H2D");
    check_cuda(cudaMemcpyAsync(d_token_w, h_tok_w.data(), m * sizeof(float), cudaMemcpyHostToDevice,
                               stream),
               "copy token_w H2D");

    {
      int total = m * kHiddenSize;
      int blocks = (total + 255) / 256;
      gather_rows_kernel<<<blocks, 256, 0, stream>>>(d_hidden_f32, d_a_expert, d_token_idx, m,
                                                      kHiddenSize);
      check_cuda(cudaGetLastError(), "launch gather_rows_kernel");
    }

    {
      const uint8_t* w13_fp8 =
          d_gemm1_weights + static_cast<size_t>(le) * (2 * kIntermediateSize) * kHiddenSize;
      const float* w13_scale =
          d_gemm1_weights_scale + static_cast<size_t>(le) * kNumGemm1OutBlocks * kNumHiddenBlocks;
      int total = (2 * kIntermediateSize) * kHiddenSize;
      int blocks = (total + 255) / 256;
      dequant_weight_kernel<<<blocks, 256, 0, stream>>>(w13_fp8, w13_scale, d_w13,
                                                         2 * kIntermediateSize, kHiddenSize,
                                                         kNumHiddenBlocks);
      check_cuda(cudaGetLastError(), "launch dequant_weight_kernel gemm1");
    }

    {
      dim3 block(16, 16);
      dim3 grid((2 * kIntermediateSize + 15) / 16, (m + 15) / 16);
      gemm_nt_kernel<<<grid, block, 0, stream>>>(d_a_expert, d_w13, d_g1, m, 2 * kIntermediateSize,
                                                 kHiddenSize);
      check_cuda(cudaGetLastError(), "launch gemm_nt_kernel gemm1");
    }

    {
      int total = m * kIntermediateSize;
      int blocks = (total + 255) / 256;
      swiglu_kernel<<<blocks, 256, 0, stream>>>(d_g1, d_c, m);
      check_cuda(cudaGetLastError(), "launch swiglu_kernel");
    }

    {
      const uint8_t* w2_fp8 =
          d_gemm2_weights + static_cast<size_t>(le) * kHiddenSize * kIntermediateSize;
      const float* w2_scale =
          d_gemm2_weights_scale + static_cast<size_t>(le) * kNumHiddenBlocks * kNumIntermediateBlocks;
      int total = kHiddenSize * kIntermediateSize;
      int blocks = (total + 255) / 256;
      dequant_weight_kernel<<<blocks, 256, 0, stream>>>(w2_fp8, w2_scale, d_w2, kHiddenSize,
                                                         kIntermediateSize, kNumIntermediateBlocks);
      check_cuda(cudaGetLastError(), "launch dequant_weight_kernel gemm2");
    }

    {
      dim3 block(16, 16);
      dim3 grid((kHiddenSize + 15) / 16, (m + 15) / 16);
      gemm_nt_kernel<<<grid, block, 0, stream>>>(d_c, d_w2, d_o, m, kHiddenSize, kIntermediateSize);
      check_cuda(cudaGetLastError(), "launch gemm_nt_kernel gemm2");
    }

    {
      int total = m * kHiddenSize;
      int blocks = (total + 255) / 256;
      scatter_add_weighted_kernel<<<blocks, 256, 0, stream>>>(d_o, d_accum, d_token_idx, d_token_w,
                                                               m);
      check_cuda(cudaGetLastError(), "launch scatter_add_weighted_kernel");
    }
  }

  {
    int total = seq_len * kHiddenSize;
    int blocks = (total + 255) / 256;
    float_to_bf16_kernel<<<blocks, 256, 0, stream>>>(d_accum, d_output, total);
    check_cuda(cudaGetLastError(), "launch float_to_bf16_kernel");
  }

  check_cuda(cudaStreamSynchronize(stream), "sync at kernel end");

  cudaFree(d_hidden_f32);
  cudaFree(d_accum);
  cudaFree(d_w13);
  cudaFree(d_w2);
  cudaFree(d_a_expert);
  cudaFree(d_g1);
  cudaFree(d_c);
  cudaFree(d_o);
  cudaFree(d_topk_idx);
  cudaFree(d_topk_w);
  cudaFree(d_token_idx);
  cudaFree(d_token_w);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel);
