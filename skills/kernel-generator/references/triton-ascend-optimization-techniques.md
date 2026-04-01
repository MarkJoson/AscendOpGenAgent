# Triton Ascend NPU 专家优化技巧

本文档提取自 Ascend NPU 平台上的实际算子优化案例。每条技巧均来自 GPU→NPU 移植过程中专家的真实修改。按照"何时用、怎么改、为什么"的结构组织。

---

## 技巧 1：显式无穷大掩码保护（Explicit Infinity Masking）

### 问题

在 Online Softmax / Flash Attention 等算子中，初始最大值 `m_i` 被设为 `float("-inf")`。后续计算 `tl.exp(m_i - m_ij)` 时，若 `m_i` 和 `m_ij` 同时为 `-inf`，则 `m_i - m_ij = -inf - (-inf) = NaN`。

GPU（Nvidia）硬件按 IEEE 754 规范执行 `exp(-inf) = 0.0`，不会产生 NaN 扩散。
Ascend NPU 的底层浮点指令在极端值（`-inf`）下可能产生 NaN。一旦出现 NaN，会通过累加器 `acc = acc * alpha` 污染整个输出矩阵块。

### 触发条件

同时满足以下条件时必须使用：
- 累加器初始化 `m_i = float("-inf")`
- 存在 `tl.exp(m_i - m_ij)` 或 `tl.exp(qk - m_ij[:, None])` 计算
- 结果参与后续乘法累加（如 `acc = acc * alpha`）

### 修改方法

在每次 `tl.exp` 之后，立即用 `tl.where` 将 `-inf` 对应路径强制归零。

修改前（GPU 原版）：
```python
m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
p = tl.exp(qk - m_ij[:, None])
l_ij = tl.sum(p, axis=1)
alpha = tl.exp(m_i - m_ij)
acc = acc * alpha[:, None]
```

修改后（Ascend 适配）：
```python
m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
p = tl.exp(qk - m_ij[:, None])
p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)      # 新增
l_ij = tl.sum(p, axis=1)
alpha = tl.exp(m_i - m_ij)
alpha = tl.where(m_i == float("-inf"), 0.0, alpha)          # 新增
acc = acc * alpha[:, None]
```

### 规则总结

- 紧跟在 `tl.exp(... - m_ij)` 之后插入 `tl.where(m_ij == float("-inf"), 0.0, result)`
- 紧跟在 `tl.exp(m_i - m_ij)` 之后插入 `tl.where(m_i == float("-inf"), 0.0, alpha)`
- 每个 `tl.exp` 相关的分支都要保护，不要遗漏
- 对性能几乎无影响

---

## 技巧 2：静态内存寻址替代指针累加（Static Memory Addressing）

### 问题

循环内对指针执行 `ptr += offset` 累加操作，形成跨循环的存储依赖（Loop-Carried Dependency）。在 Ascend NPU 上，访存单元需等待前一次 ALU 加法完成后才能发起新的地址 Fetch，阻碍循环展开和流水线化。

### 触发条件

循环体末尾存在类似以下模式：
```python
for j in range(num_blocks):
    data = tl.load(ptr, ...)
    # ... 计算 ...
    ptr += STEP          # 指针累加，形成循环依赖
```

### 修改方法

用基址 + 循环变量乘偏移的方式替代指针累加。

修改前：
```python
for j in range(num_blocks):
    k = tl.load(K_ptr, mask=..., other=0.0)
    v = tl.load(V_ptr, mask=..., other=0.0)
    decay = tl.load(decay_ptr)
    # ... 计算 ...
    K_ptr += CBLOCK * d
    V_ptr += CBLOCK * e
    decay_ptr += CBLOCK
```

修改后：
```python
for j in range(num_blocks):
    k = tl.load(K_ptr + j * CBLOCK * d, mask=..., other=0.0)
    v = tl.load(V_ptr + j * CBLOCK * e, mask=..., other=0.0)
    decay = tl.load(decay_ptr + j * CBLOCK)
    # ... 计算 ...
    # 删除所有 ptr += ... 语句
```

### 规则总结

- 将 `ptr += STEP` 替换为 `ptr + j * STEP`（j 是循环变量）
- 删除循环末尾的所有指针自增语句
- 适用于所有循环内的 `tl.load` / `tl.store` 指针
- 对正确性无影响
- 使得编译器可以完全展开循环并行流水线化多次 `tl.load`

### 注意事项

- 使用此技巧时，偏移量相关参数（如 `d`、`e`、`CBLOCK` 等）最好声明为 `tl.constexpr`，以便编译器在编译期计算地址偏移
- 如果参数不是 `tl.constexpr`，此优化仍然有效，但编译器展开能力受限

---

## 技巧 3：任务聚合 / 线程块批量包裹（Task Aggregation / Block Batching）

### 问题

GPU 可以高效调度大量细粒度小线程块（每个 Block 计算量极小），Ascend NPU 的任务启动和上下文切换开销相对较高。当 Grid 中每个 Block 的工作量很小时，启动开销占比过大。

### 触发条件

同时满足：
- Grid 某个维度很大（如 `NUM_BLOCK * NUM_CBLOCK`）
- 每个 Block 内的计算量很轻（如仅做一次矩阵-向量乘加载/存储）
- 性能瓶颈在 kernel launch 而非计算

### 修改方法

引入一个 `NC_BLOCK`（或类似名字的 `tl.constexpr` 参数），在 kernel 内部增加一层循环，每个 Block 处理原来 `NC_BLOCK` 个 Block 的工作。同时缩小 Grid 为原来的 `1/NC_BLOCK`。

修改前：
```python
@triton.jit
def kernel(...):
    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    # ... 单个子块的计算逻辑 ...

# 启动：grid=(batch*head, NUM_BLOCK * NUM_CBLOCK, num_e_blocks)
```

修改后：
```python
@triton.jit
def kernel(..., NC_BLOCK: tl.constexpr):
    for nci in range(NC_BLOCK):
        off_nc = tl.program_id(1) * NC_BLOCK + nci
        off_n = off_nc // NUM_CBLOCK
        off_c = off_nc % NUM_CBLOCK
        # ... 单个子块的计算逻辑不变，全部包在循环内 ...

# 启动：grid=(batch*head, (NUM_BLOCK * NUM_CBLOCK) // NC_BLOCK, num_e_blocks)
```

### 规则总结

- `NC_BLOCK` 典型取值：2, 4, 8
- 将整个 kernel body 包在 `for nci in range(NC_BLOCK)` 循环内
- `program_id` 的使用改为 `tl.program_id(axis) * NC_BLOCK + nci`
- Grid 对应维度缩小为 `原值 // NC_BLOCK`
- 不改变计算逻辑本身，仅改变调度粒度
- 提升缓存命中率和 L2 访存连续性

---

## 技巧 4：标量到向量的 Tile 化（Tile Vectorization）

### 问题

原始 GPU 代码中，kernel 的每个 Block 仅处理 1 个样本（标量操作），然后通过大 Grid 启动大量标量任务。Ascend NPU 拥有宽 SIMD 向量单元，一次仅处理 1 个元素会导致向量寄存器大面积闲置。

### 触发条件

同时满足：
- `tl.program_id(0)` 直接作为数据索引（如 `batch_idx = tl.program_id(0)`）
- 每个 Block 仅处理一个标量元素
- kernel 内不含矩阵运算（纯标量逻辑）

### 修改方法

引入 `BATCH_BLOCK`（如 8, 16, 32），将标量索引替换为向量索引。

修改前：
```python
@triton.jit
def kernel(ptr_a, ptr_b, ...):
    idx = tl.program_id(0)                    # 标量索引
    a = tl.load(ptr_a + idx)                  # 加载 1 个元素
    b = tl.load(ptr_b + idx)
    result = a + b
    tl.store(out_ptr + idx, result)

# 启动：grid=(N,)
```

修改后：
```python
@triton.jit
def kernel(ptr_a, ptr_b, ..., BATCH_BLOCK: tl.constexpr):
    block_start = tl.program_id(0) * BATCH_BLOCK
    idx = block_start + tl.arange(0, BATCH_BLOCK)   # 向量索引
    a = tl.load(ptr_a + idx)                         # 加载 BATCH_BLOCK 个元素
    b = tl.load(ptr_b + idx)
    result = a + b
    tl.store(out_ptr + idx, result)

# 启动：grid=(N // BATCH_BLOCK,)
```

### 规则总结

- `BATCH_BLOCK` 典型取值：8, 16, 32
- 将 `tl.program_id(0)` 从直接索引改为块起始位置
- 用 `tl.arange(0, BATCH_BLOCK)` 构造向量索引
- 所有 `tl.load` / `tl.store` 自动变为向量操作
- Grid 缩小为 `原值 // BATCH_BLOCK`
- 通常与技巧 5（无分支算术）配合使用

---

## 技巧 5：无分支算术替代 tl.where（Branchless Arithmetic）

### 问题

`tl.where(condition, value, 0)` 在 GPU 上有 predication 指令支持，开销接近零。Ascend NPU 的条件选择可能被编译为较复杂的掩码处理流程，打断向量引擎流水线。

### 触发条件

同时满足：
- `tl.where` 的 "else" 分支为 0（即 `tl.where(cond, x, 0)`）
- condition 是布尔值（0 或 1）
- 用 `tl.where` 的目的是"满足条件保留值，不满足置零"

### 修改方法

将 `tl.where(cond, x, 0)` 替换为 `x * cond`。如果原始条件需要取反，直接构造反向布尔变量。

修改前：
```python
is_prefilling = seq_len < prefill_len
num_sampled = tl.load(ptr + idx)
num_sampled = tl.where(is_prefilling, 0, num_sampled)
num_rejected = num_logits - num_sampled
num_rejected = tl.where(is_prefilling, 0, num_rejected)
```

修改后：
```python
not_prefilling = prefill_len < seq_len + 1             # 反向布尔：0 或 1
num_sampled = tl.load(ptr + idx) * not_prefilling      # 乘法替代 tl.where
num_rejected = not_prefilling * (num_logits - num_sampled)
```

### 规则总结

- `tl.where(cond, value, 0)` → `value * cond`
- `tl.where(cond, 0, value)` → `value * (1 - cond)` 或创建反向布尔 `not_cond`
- 布尔变量在 Triton 中参与乘法时自动转为 0/1 整数
- 如果原始布尔条件为 `a < b`，反向条件可写为 `b < a + 1`（整数场景）或 `b <= a`
- 此技巧仅适用于 "else" 分支为零值的场景
- 若 `tl.where` 的两个分支都是非零值，则不适用此技巧，保留 `tl.where`

---

## 技巧 6：零改造直接编译（Zero-Modification Compatibility）

### 问题判定

并非所有 GPU Triton kernel 都需要修改才能在 Ascend 上运行。很多标准算子可以直接使用。

### 可以零改造的算子特征

满足以下全部条件的 kernel 可以不做任何修改直接编译到 Ascend：
- 仅使用 Triton 标准 API（`tl.load`、`tl.store`、`tl.sum`、`tl.max`、`tl.exp`、`tl.dot`、`tl.maximum` 等）
- 不依赖 CUDA 特有原语（如 warp shuffle、特定 warp size 假设、`tl.inline_asm`）
- 不包含 GPU 特定的 `num_warps`、`num_stages` 等调优参数在 kernel 逻辑层面的硬编码
- Block 级归约操作使用标准 pattern（如 `tl.sum(data, axis=0)`）
- 内存访问是规整的（连续或带标准 mask 的）

### 实际案例

以下类型算子在专家实测中无需修改即可在 Ascend 上正确运行：
- 标准 Block-level Reduce（块内归约 + 原子操作写回）
- 标准 LayerNorm / RMSNorm 前向（均值、方差、归一化、weight/bias 仿射）
- Flash Decoding Stage1（QK 点积 + 局部 softmax）
- Flash Decoding Stage2（全局 reduction + softmax 归一化）
- 纯 IO / Memory-Bound 算子（如 PagedAttention 中的 block table gather）
- 标准 Linear Attention Decode（KV Cache 加载 + 线性衰减聚合）

### 判断规则

遇到将 GPU Triton kernel 移植到 Ascend 时，首先判断：
1. kernel 是否仅使用标准 Triton API？
2. kernel 是否假设了特定的 warp/wavefront 大小？
3. kernel 是否使用了 GPU 特有内联汇编或 libdevice 函数？

如果回答为"是、否、否"，则大概率可以直接编译，不需要改动。先尝试零改造编译运行，仅在出现正确性或性能问题时再应用其他技巧。

---

## 技巧组合使用指南

多个技巧可以在同一个 kernel 中组合使用。常见组合：

### 组合 A：标量 kernel → 向量高性能 kernel
依次应用：技巧 4（Tile Vectorization）+ 技巧 5（Branchless Arithmetic）
- 先将标量操作 Tile 化为向量
- 再将 `tl.where` 替换为乘法
- 适用于 batch 级逐元素处理的轻量 kernel

### 组合 B：循环密集型 kernel 加速
依次应用：技巧 2（Static Addressing）+ 技巧 3（Task Aggregation）
- 先消除循环内指针累加依赖
- 再将多个 Grid Block 合并到单个 Block 内循环
- 适用于多次循环加载矩阵块的 attention / KV 外积类 kernel

### 组合 C：Attention 类 kernel 安全移植
应用：技巧 1（Infinity Masking）+ 技巧 6（Zero-Modification 判断）
- 先判断是否可以零改造
- 如果涉及 Online Softmax，则加上 Infinity Masking
- 其余部分保持不变

---

## 快速查找：按算子类型选择技巧

### Attention / Flash Attention / Online Softmax 算子
- 必须检查：技巧 1（Infinity Masking）
- 可能需要：技巧 2（Static Addressing，如果有循环内指针累加）
- 可能需要：技巧 3（Task Aggregation，如果 Grid 过大且单 Block 计算量小）

### 逐元素 / 标量批处理算子
- 必须检查：技巧 4（Tile Vectorization）
- 配合使用：技巧 5（Branchless Arithmetic）

### 循环密集型矩阵运算算子（KV 外积、Block 级聚合）
- 必须检查：技巧 2（Static Addressing）
- 可能需要：技巧 3（Task Aggregation）

### 标准归约 / LayerNorm / IO 搬运算子
- 先尝试：技巧 6（零改造）
- 通常不需要额外修改
