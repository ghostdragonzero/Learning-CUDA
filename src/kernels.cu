#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"
template<typename T>
__global__ void traceKernel(T* input, int rows, int cols, T* result) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  size_t tid = threadIdx.x;
  int min_col = min(rows, cols);  // 使用较小的维度
  // 对角线元素数量是 min(rows, cols)
  T sum = T(0);
  for (int i = idx; i < min_col; i += blockDim.x * gridDim.x) {
    sum += input[i * cols + i];  // 访问对角线元素
  }
  for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }
  if (tid % 32 == 0) {                                                                                                                                    
      atomicAdd(result, sum);                                                                                                                             
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
  T* d_input;
  cudaMalloc(&d_input, h_input.size() * sizeof(T));
  
  // ✅ 修复：使用正确的大小参数
  cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice);
  
  T* d_result;
  cudaMalloc(&d_result, sizeof(T));
  cudaMemset(d_result, 0, sizeof(T));
  
  int blockSize = 32;
  dim3 block(blockSize);
  dim3 grid((min(rows, cols) + blockSize - 1) / blockSize);  // ✅ 这行正确
  //使用较小值是因为对角线元素数量是 min(rows, cols)
  traceKernel<<<grid, block>>>(d_input, rows, cols, d_result);
  cudaDeviceSynchronize();  // 添加同步
  
  T result;
  cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
  
  cudaFree(d_input);
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

  // threadIdx.x 处理 head_dim 的计算
  // 每个线程可能需要处理多个元素（如果 head_dim > blockDim.x）
  int tid = threadIdx.x;
  int threads_x = blockDim.x;

  // ========================================
  // 2. 共享内存：用于存储分块数据
  // ========================================
  // FlashAttention 的核心：将大的注意力矩阵分块加载到共享内存
  extern __shared__ float smem[];

  // 共享内存布局：
  // - Q_tile: [head_dim] - 当前 query 向量
  // - K_tile: [BLOCK_SIZE, head_dim] - 分块的 key 矩阵
  // - V_tile: [BLOCK_SIZE, head_dim] - 分块的 value 矩阵
  float* Q_tile = smem;
  float* K_tile = Q_tile + head_dim;
  float* V_tile = K_tile + blockDim.y * head_dim;

  const int BLOCK_SIZE = blockDim.y;  // K/V 分块大小

  // ========================================
  // 3. 加载当前 query 向量到共享内存
  // ========================================
  // Q 的索引计算: [batch_id, tgt_pos, q_head_id, :]
  int q_offset = ((batch_id * target_seq_len + tgt_pos) * query_heads + q_head_id) * head_dim;

  // 每个线程加载多个元素（如果 head_dim > threads_x）
  for (int d = tid; d < head_dim; d += threads_x) {
    Q_tile[d] = float(q[q_offset + d]);
  }

  __syncthreads();  // 等待所有线程完成 Q 的加载

  // ========================================
  // 4. FlashAttention 的核心循环：分块处理 K 和 V
  // ========================================
  // 维护在线 softmax 的统计量：
  // - o_curr: 当前累积的输出向量 [head_dim]
  // - m_curr: 当前最大的注意力分数（用于 softmax 的数值稳定性）
  // - l_curr: 当前累积的 softmax 分母（指数和）

  float* o_curr = Q_tile;  // 复用 Q_tile 的空间来存储输出
  float m_curr = -INFINITY;
  float l_curr = 0.0f;

  // 初始化输出为 0
  for (int d = tid; d < head_dim; d += threads_x) {
    o_curr[d] = 0.0f;
  }

  __syncthreads();  // 等待所有线程完成初始化

  // 遍历所有 K/V 分块
  for (int k_block_start = 0; k_block_start < src_seq_len; k_block_start += BLOCK_SIZE) {

    // ----------------------------------------
    // 4.1 加载 K 和 V 的当前分块到共享内存
    // ----------------------------------------
    int k_block_end = min(k_block_start + BLOCK_SIZE, src_seq_len);
    int actual_block_size = k_block_end - k_block_start;

    // 每个线程加载 K_tile 和 V_tile 的一部分
    // K_tile[k_idx][d] = K[batch_id, k_pos, kv_head_id, d]
    // V_tile[k_idx][d] = V[batch_id, k_pos, kv_head_id, d]

    for (int k_idx = 0; k_idx < actual_block_size; k_idx++) {
      int k_pos = k_block_start + k_idx;

      // 计算 K 和 V 的全局索引
      int kv_offset = ((batch_id * src_seq_len + k_pos) * kv_heads + kv_head_id) * head_dim;

      // 每个线程加载多个元素
      for (int d = tid; d < head_dim; d += threads_x) {
        K_tile[k_idx * head_dim + d] = float(k[kv_offset + d]);
        V_tile[k_idx * head_dim + d] = float(v[kv_offset + d]);
      }
    }

    __syncthreads();  // 等待所有线程完成加载

    // ----------------------------------------
    // 4.2 计算当前分块的注意力分数
    // ----------------------------------------
    // Q @ K^T: query 与当前分块中所有 key 的点积
    // S[j] = Q · K[j]^T / sqrt(head_dim)

    for (int k_idx = 0; k_idx < actual_block_size; k_idx++) {
      int k_pos = k_block_start + k_idx;

      // Causal masking: 只能注意之前的位置（包括自己）
      if (is_causal && k_pos > tgt_pos) {
        continue;
      }

      // 计算 Q · K[j]^T (点积) - 使用归约操作
      float s_ij = 0.0f;
      for (int d = tid; d < head_dim; d += threads_x) {
        s_ij += Q_tile[d] * K_tile[k_idx * head_dim + d];
      }

      // 线程间归约求和（得到完整的点积）
      for (int stride = threads_x / 2; stride > 0; stride /= 2) {
        s_ij += __shfl_down_sync(0xFFFFFFFF, s_ij, stride);
      }

      // 只有 tid=0 的线程持有完整的点积结果，广播给所有线程
      s_ij = __shfl_sync(0xFFFFFFFF, s_ij, 0);
      s_ij *= __frsqrt_rn(float(head_dim));  // 除以 sqrt(head_dim)

      // ----------------------------------------
      // 4.3 在线 softmax 更新 (FlashAttention 核心)
      // ----------------------------------------
      // 使用新的注意力分数 s_ij 更新统计量：
      // - m_new = max(m_curr, s_ij)
      // - l_new = exp(m_curr - m_new) * l_curr + exp(s_ij - m_new)
      // - o_new = exp(m_curr - m_new) * o_curr + exp(s_ij - m_new) * V[j]

      float m_new = max(m_curr, s_ij);

      // 计算新的归一化因子
      float l_new = expf(m_curr - m_new) * l_curr + expf(s_ij - m_new);

      // 更新输出向量（注意：这里不除以 l_new，而是在最后统一除）
      for (int d = tid; d < head_dim; d += threads_x) {
        float o_new = expf(m_curr - m_new) * o_curr[d] + expf(s_ij - m_new) * V_tile[k_idx * head_dim + d];
        o_curr[d] = o_new;
      }

      // 更新统计量
      m_curr = m_new;
      l_curr = l_new;
    }

    __syncthreads();  // 等待所有线程完成当前分块的计算
  }

  // ========================================
  // 5. 最终归一化：除以累积的 softmax 分母
  // ========================================
  // 注意：如果 l_curr 仍然是 0（例如 causal masking 时没有有效位置），
  // 输出应该保持为 0 或设置为一个默认值
  if (l_curr > 0.0f) {
    for (int d = tid; d < head_dim; d += threads_x) {
      o_curr[d] = o_curr[d] / l_curr;
    }
  } else {
    // 没有有效的 attention，输出保持为 0
    for (int d = tid; d < head_dim; d += threads_x) {
      o_curr[d] = 0.0f;
    }
  }

  // ========================================
  // 6. 将结果写回全局内存
  // ========================================
  int o_offset = ((batch_id * target_seq_len + tgt_pos) * query_heads + q_head_id) * head_dim;

  for (int d = tid; d < head_dim; d += threads_x) {
    o[o_offset + d] = T(o_curr[d]);
  }
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
  // 简化版本：使用固定的分块大小
  const int BLOCK_SIZE_KV = 32;  // K/V 分块大小

  // 线程数必须是 2 的幂，用于 warp shuffle
  int threads_x = 32;


  dim3 block(threads_x, BLOCK_SIZE_KV);
  dim3 grid(batch_size, query_heads, target_seq_len);

  // 计算共享内存大小
  size_t smem_size = (head_dim + BLOCK_SIZE_KV * head_dim * 2) * sizeof(float);

  // ========================================
  // 打印线程配置并验证硬件限制
  // ========================================
  printf("\n========== flashAttentionKernel Launch Configuration ==========\n");
  printf("Problem size: batch=%d, tgt_seq=%d, src_seq=%d, q_heads=%d, kv_heads=%d, head_dim=%d\n",
         batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim);
  printf("Block dimensions: (%d, %d, %d)\n", block.x, block.y, block.z);
  printf("Grid dimensions: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
  printf("Threads per block: %d\n", block.x * block.y * block.z);
  printf("Total blocks: %d\n", grid.x * grid.y * grid.z);
  printf("Shared memory per block: %zu bytes\n", smem_size);
  printf("\nHardware Limits Check:\n");
  printf("  ✓ Max threads per block: 1024 [Current: %d %s]\n",
         block.x * block.y * block.z,
         (block.x * block.y * block.z <= 1024) ? "PASS" : "FAIL");
  printf("  ✓ Max block dimensions: 1024 x 1024 x 64 [Current: %d x %d x %d %s]\n",
         block.x, block.y, block.z,
         (block.x <= 1024 && block.y <= 1024 && block.z <= 64) ? "PASS" : "FAIL");
  printf("  ✓ Max grid dimensions: 2147483647 x 65535 x 65535 [Current: %d x %d x %d %s]\n",
         grid.x, grid.y, grid.z,
         (grid.x <= 2147483647 && grid.y <= 65535 && grid.z <= 65535) ? "PASS" : "FAIL");
  printf("  ✓ Max threads per SM: 1536 [Need device query for SM count to verify]\n");
  printf("==============================================================\n\n");

  // 5. 启动 kernel
  flashAttentionKernel<T><<<grid, block, smem_size>>>(
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
