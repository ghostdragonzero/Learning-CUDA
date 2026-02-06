#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

#include "../tester/utils.h"
  template<typename T> 
  __device__ T warp_reduce(T val){
      for(int offset = 16; offset > 0; offset >>= 1){
          val += __shfl_down_sync(0xFFFFFFFF, val, offset);
      }
      return val;
  }
// 优化版：只处理对角线数据，输入应该是对角线元素的一维数组
template<typename T>
__global__ void traceKernel(T* diagonal, int diag_size, T* result) {
  __shared__ T trace_smem[32];
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  size_t tid = threadIdx.x;

  T sum = T(0);
  for (int i = idx; i < diag_size; i += blockDim.x * gridDim.x) {
    sum += diagonal[i];  // 直接访问对角线元素
  }
  T warp_sum = warp_reduce(sum);
  if (tid % 32 == 0) {
      trace_smem[tid / 32] = warp_sum;
  }
  __syncthreads();
  if (tid < blockDim.x / 32) {
      T block_sum = (tid < (blockDim.x + 31) / 32) ? trace_smem[tid] : T(0);                                                                                                               
      T total = warp_reduce(block_sum);                                                                                                                 
      if (tid == 0) {                                                                                                                                   
          atomicAdd(result, total);
      }
    }    
}

/**
 * @br‘ief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // 优化：在 host 端先提取对角线元素，减少数据传输
  size_t diag_size = std::min(rows, cols);
  std::vector<T> h_diagonal(diag_size);

  for (size_t i = 0; i < diag_size; ++i) {
    h_diagonal[i] = h_input[i * cols + i];  // 提取对角线元素
  }

  // 只传输对角线数据，而非整个矩阵
  T* d_diagonal;
  cudaMalloc(&d_diagonal, diag_size * sizeof(T));
  cudaMemcpy(d_diagonal, h_diagonal.data(), diag_size * sizeof(T), cudaMemcpyHostToDevice);

  T* d_result;
  cudaMalloc(&d_result, sizeof(T));
  cudaMemset(d_result, 0, sizeof(T));

  int blockSize = 256;
  dim3 block(blockSize);
  dim3 grid((diag_size + blockSize - 1) / blockSize);

  traceKernel<<<grid, block>>>(d_diagonal, diag_size, d_result);
  cudaDeviceSynchronize();

  T result;
  cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_diagonal);
  cudaFree(d_result);

  return result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
// ========================================================================
// FlashAttention Kernel: 分块注意力计算的核心实现
// ========================================================================
template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ q,  // [batch_size, target_seq_len, query_heads, head_dim]
    const T* __restrict__ k,  // [batch_size, src_seq_len, kv_heads, head_dim]
    const T* __restrict__ v,  // [batch_size, src_seq_len, kv_heads, head_dim]
    T* __restrict__ o,        // [batch_size, target_seq_len, query_heads, head_dim]
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {

  // ========================================
  // 1. 线程和块的组织
  // ========================================
  // 每个 block 处理一个 query 序列位置
  // blockIdx.x = batch 中的样本索引
  // blockIdx.y = query head 的索引
  // blockIdx.z = 目标序列中的位置 (target_seq_len 中的哪个 token)

  int batch_id = blockIdx.x;
  int q_head_id = blockIdx.y;
  int tgt_pos = blockIdx.z;

  // 处理 GQA (Grouped Query Attention)：多个 query heads 共享一组 kv heads
  int kv_head_id = q_head_id / (query_heads / kv_heads);

  // threadIdx.x 处理 head_dim 的计算 (head_dim=32, blockDim.x=32, 每个线程处理一个元素)
  int tid = threadIdx.x;
  int threads_x = blockDim.x;

  // ========================================
  // 2. 共享内存：存储 query 向量和输出向量
  // ========================================
  // 使用足够大的数组以支持不同的 head_dim 值
  __shared__ float Q_tile[128];   // 当前 query 向量
  __shared__ float O_tile[128];   // 累积输出向量
  __shared__ float max_score_shared;  // 共享的最大分数
  __shared__ float sum_exp_shared;    // 共享的指数和

  // ========================================
  // 3. 加载当前 query 向量到共享内存
  // ========================================
  // Q 的索引计算: [batch_id, tgt_pos, q_head_id, :]
  int q_offset = ((batch_id * target_seq_len + tgt_pos) * query_heads + q_head_id) * head_dim;

  // head_dim = 32 = threads_x, 每个线程加载一个元素
  Q_tile[tid] = float(q[q_offset + tid]);
    // 初始化输出为 0
  O_tile[tid] = 0.0f;
  //把这个Q向量保存起来

  __syncthreads();  // 等待所有线程完成 Q 的加载

  // ========================================
  // 4. FlashAttention 的核心循环：遍历所有 key-value pairs
  // ========================================
  // 使用朴素的两阶段方法：先收集所有注意力分数，再计算 softmax

  // 第一阶段：计算所有注意力分数并找到最大值
  __shared__ float S_tile[4096];  // 最多支持 4096 的序列长度

  if (tid == 0) {
    max_score_shared = -INFINITY;
    for (int k_pos = 0; k_pos < src_seq_len; k_pos++) {
      // Causal masking: 只能注意之前的位置（包括自己）
      if (is_causal && k_pos > tgt_pos) {
        S_tile[k_pos] = -INFINITY;
        continue;
      }

      // 计算 K 的偏移量
      int k_offset = ((batch_id * src_seq_len + k_pos) * kv_heads + kv_head_id) * head_dim;

      // 计算注意力分数: Q · K^T / sqrt(head_dim)
      float s_ij = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        s_ij += Q_tile[d] * float(k[k_offset + d]);
      }
      s_ij /= sqrtf(float(head_dim));

      S_tile[k_pos] = s_ij;
      max_score_shared = fmaxf(max_score_shared, s_ij);
    }
  }

  __syncthreads();

  // 第二阶段：计算 softmax 的指数和（tid=0 计算）
  if (tid == 0) {
    sum_exp_shared = 0.0f;
    for (int k_pos = 0; k_pos < src_seq_len; k_pos++) {
      if (S_tile[k_pos] > -INFINITY / 2) {  // 有效位置
        sum_exp_shared += expf(S_tile[k_pos] - max_score_shared);
      }
    }
  }

  __syncthreads();

  // 第三阶段：计算最终输出
  for (int k_pos = 0; k_pos < src_seq_len; k_pos++) {
    if (S_tile[k_pos] > -INFINITY / 2) {  // 有效位置
      int v_offset = ((batch_id * src_seq_len + k_pos) * kv_heads + kv_head_id) * head_dim;
      float weight = expf(S_tile[k_pos] - max_score_shared) / sum_exp_shared;
      O_tile[tid] += weight * float(v[v_offset + tid]);
    }
  }

  __syncthreads();

  // ========================================
  // 5. 输出已经归一化，直接写回
  // ========================================

  // ========================================
  // 6. 将结果写回全局内存
  // ========================================
  int o_offset = ((batch_id * target_seq_len + tgt_pos) * query_heads + q_head_id) * head_dim;
  o[o_offset + tid] = T(O_tile[tid]);
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {

  // ========================================
  // 主机端函数：内存分配和内核启动
  // ========================================

  // 1. 计算数据大小
  size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
  size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
  size_t o_size = batch_size * target_seq_len * query_heads * head_dim;
 
  // 2. 分配 GPU 内存
  T *d_q, *d_k, *d_v, *d_o;
  cudaMalloc(&d_q, q_size * sizeof(T));
  cudaMalloc(&d_k, kv_size * sizeof(T));
  cudaMalloc(&d_v, kv_size * sizeof(T));
  cudaMalloc(&d_o, o_size * sizeof(T));

  // 3. 拷贝数据到 GPU
  cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);

  // 4. 配置 kernel 启动参数
  // 线程数必须是 2 的幂，用于 warp shuffle
  int threads_x = 32;
  if (head_dim < 32) {
    threads_x = head_dim;
  }
  


  dim3 block(threads_x);
  dim3 grid(batch_size, query_heads, target_seq_len);

  // ========================================
  // 打印线程配置并验证硬件限制
  // ========================================
  /*
  printf("\n========== flashAttentionKernel Launch Configuration ==========\n");
  printf("Problem size: batch=%d, tgt_seq=%d, src_seq=%d, q_heads=%d, kv_heads=%d, head_dim=%d\n",
         batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim);
  printf("==============================================================\n\n");
*/
  // 5. 启动 kernel
  flashAttentionKernel<T><<<grid, block>>>(
      d_q, d_k, d_v, d_o,
      batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim, is_causal);

  // 6. 同步并拷贝结果回主机
  cudaDeviceSynchronize();
  cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);

  // 7. 清理 GPU 内存
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
