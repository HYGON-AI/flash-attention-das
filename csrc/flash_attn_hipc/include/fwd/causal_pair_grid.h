#pragma once
#include "hip/hip_runtime.h"

// causal fwd 的 "重块 + 轻块" 配对网格判断 + 尾巴拆分计划 (128x128 / gfx92a 标定版)。
//   1) 只给 MHA (h_q == h_kv); GQA/MQA 走 LPT
//   2) m_eff = round(seqlen_q / kBlockM) >= 2
//   3) 波数 waves = h*b*ceil(m/2) / slots: m%4==2 要求 >= 2.6, 其余 >= 1.8
//   4) 4n / 4n+2 放行; 奇数只放 (m%4 == 1 && m >= 17)
//   5) h_q > 32 且 b < 7 不配
//   6) 尾巴: R = n_pair % slots, 2R <= slots 时拆成 2R 个单块

constexpr int kGfx92aFwdCtasPerCu = 2;   // gfx92a
constexpr int kGfx938FwdCtasPerCu = 3;   // gfx938

struct CausalPairCfg {
    int  waves_x10;        // m%4 != 2 的波数门槛 x10  (18 = 1.8 波)
    int  waves_x10_mod2;   // m%4 == 2 的波数门槛 x10  (26 = 2.6 波)
    int  odd_min_m;        // 奇数 m 的最小 m          (17)
    int  max_q_heads;      // 多头 + 小 batch 的 h 上限 (32)
    int  min_batches;      // 多头 + 小 batch 的 b 下限 (6)
    bool allow_gqa;        // 是否允许 GQA/MQA 配对     (false)
    bool dissolve_tail;    // 尾波拆分                  (true)
};

constexpr CausalPairCfg kCausalPairCfg_gfx92a_128x128 = { 18, 26, 17, 32, 6, false, true };

struct CausalPairPlan {
    bool      use_pair;        // false -> 走 LPT 网格
    int       pair_slots;      // 每个 class 的 pair 槽数 = ceil(m_block/2) (与 kernel 的 pair_m_block 一致)
    int       dissolve_pairs;  // 尾巴里要拆成单块的 pair 数 (0 = 不拆)
    long long n_pair;          // 配对网格 CTA 数 (未拆)
    long long slots;           // ctas_per_cu * cu_count
};

__host__ __device__ inline CausalPairPlan causal_pair_grid_enabled(
        const CausalPairCfg &cfg,
        const int num_q_heads, const int num_heads, const int num_heads_kv,
        const int batch_size, const int num_m_blocks, const int m_eff,
        const int cu_count, const int ctas_per_cu) {
    CausalPairPlan plan{false, 0, 0, 0, 0};

    if ((!cfg.allow_gqa) && num_heads != num_heads_kv) return plan;   // 1)
    if (num_m_blocks < 2) return plan;                               // 2)
    if (ctas_per_cu <= 0 || cu_count <= 0) return plan;

    const long long slots      = (long long)ctas_per_cu * (long long)cu_count;
    const long long pair_slots = (long long)((num_m_blocks + 1) / 2);   // == kernel 的 pair_m_block
    const long long n_pair     = (long long)num_heads * (long long)batch_size * pair_slots;

    plan.pair_slots = (int)pair_slots;
    plan.slots      = slots;
    plan.n_pair     = n_pair;

    const int m = m_eff;
    const long long need_x10 = (m % 4 == 2) ? (long long)cfg.waves_x10_mod2     // 3)
                                           : (long long)cfg.waves_x10;
    if (n_pair * 10 < need_x10 * slots) return plan;

    if ((m & 1) != 0 && !(m % 4 == 1 && m >= cfg.odd_min_m)) return plan;       // 4)
    if (num_q_heads > cfg.max_q_heads) return plan;  // 5)
    if (batch_size < cfg.min_batches) return plan;  // 6)

    plan.use_pair = true;

    const long long r = n_pair % slots;                                        // 7)
    if (cfg.dissolve_tail && r > 0 && r * 5 <= slots) plan.dissolve_pairs = (int)r; // 当尾波块很少时再拆分
    return plan;
}