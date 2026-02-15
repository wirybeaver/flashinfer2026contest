/*
 * Track A CUDA scaffold with TVM-FFI export.
 *
 * This provides a compilable CUDA entrypoint for the FlashInfer-Bench CUDA path.
 * It validates key tensor shapes and writes zeros to output as a safe placeholder.
 * Replace with a fused routing + grouped-GEMM implementation for competitive speed.
 */

#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace flashinfer_contest {

__global__ void FillZeroKernel(__nv_bfloat16* out, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = __float2bfloat16(0.0f);
  }
}

void kernel(tvm::ffi::TensorView routing_logits, tvm::ffi::TensorView routing_bias,
            tvm::ffi::TensorView hidden_states, tvm::ffi::TensorView hidden_states_scale,
            tvm::ffi::TensorView gemm1_weights, tvm::ffi::TensorView gemm1_weights_scale,
            tvm::ffi::TensorView gemm2_weights, tvm::ffi::TensorView gemm2_weights_scale,
            int32_t local_expert_offset, float routed_scaling_factor, tvm::ffi::TensorView output) {
  (void)local_expert_offset;
  (void)routed_scaling_factor;
  (void)routing_bias;
  (void)hidden_states;
  (void)hidden_states_scale;
  (void)gemm1_weights;
  (void)gemm1_weights_scale;
  (void)gemm2_weights;
  (void)gemm2_weights_scale;

  TVM_FFI_ICHECK_EQ(routing_logits.ndim(), 2) << "routing_logits must be 2D";
  TVM_FFI_ICHECK_EQ(routing_logits.size(1), 256) << "num_experts must be 256";
  TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output must be 2D";
  TVM_FFI_ICHECK_EQ(output.size(1), 7168) << "hidden_size must be 7168";
  TVM_FFI_ICHECK_EQ(output.size(0), routing_logits.size(0)) << "seq_len mismatch";
  TVM_FFI_ICHECK_EQ(output.device().device_type, kDLCUDA) << "output must be on CUDA";
  TVM_FFI_ICHECK_EQ(output.dtype().code, kDLBfloat) << "output dtype must be bfloat16";
  TVM_FFI_ICHECK_EQ(output.dtype().bits, 16) << "output dtype must be bfloat16";

  int64_t n = output.numel();
  auto* out_ptr = static_cast<__nv_bfloat16*>(output.data_ptr());
  DLDevice dev = output.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  FillZeroKernel<<<blocks, threads, 0, stream>>>(out_ptr, n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel);

}  // namespace flashinfer_contest
