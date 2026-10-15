#pragma once

// These helpers only issue four matrix DS reads with one shared LDS base setup.
// Callers must issue the required s_waitcnt for DS-read completion and any
// s_barrier needed to synchronize waves in the workgroup.
template<typename Vec>
__forceinline__ __device__ void inline_ds_read_matrix_trans_64x16_b8x4_no_wait(
    int8_t* base, Vec& r0, Vec& r1, Vec& r2, Vec& r3
) {
    const uint32_t address = uint32_t(reinterpret_cast<uintptr_t>(base));
    asm volatile("s_mov_b32 m0, %4\n\ts_nop 0\n\t"
                 "ds_read_matrix_trans_format %0, m0 offset:0 element:1 row:3 col:1 alt:0\n\t"
                 "ds_read_matrix_trans_format %1, m0 offset:1024 element:1 row:3 col:1 alt:0\n\t"
                 "ds_read_matrix_trans_format %2, m0 offset:2048 element:1 row:3 col:1 alt:0\n\t"
                 "ds_read_matrix_trans_format %3, m0 offset:3072 element:1 row:3 col:1 alt:0\n"
                 : "=v"(r0), "=v"(r1), "=v"(r2), "=v"(r3)
                 : "s"(address)
                 : "m0", "memory");
}

template<typename Vec>
__forceinline__ __device__ void inline_ds_read_matrix_32x32_b8x4_no_wait(
    int8_t* base, Vec& r0, Vec& r1, Vec& r2, Vec& r3
) {
    const uint32_t address = uint32_t(reinterpret_cast<uintptr_t>(base));
    asm volatile("s_mov_b32 m0, %4\n\ts_nop 0\n\t"
                 "ds_read_matrix_format %0, m0 offset:0 element:1 row:2 col:2 alt:0\n\t"
                 "ds_read_matrix_format %1, m0 offset:32 element:1 row:2 col:2 alt:0\n\t"
                 "ds_read_matrix_format %2, m0 offset:2048 element:1 row:2 col:2 alt:0\n\t"
                 "ds_read_matrix_format %3, m0 offset:2080 element:1 row:2 col:2 alt:0\n"
                 : "=v"(r0), "=v"(r1), "=v"(r2), "=v"(r3)
                 : "s"(address)
                 : "m0", "memory");
}
