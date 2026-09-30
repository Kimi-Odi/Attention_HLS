// =============================================================
//  attention.h  --  共用標頭檔 (所有 solution 共用)
//  單頭 scaled dot-product attention, FP16
//
//  所有架構都實作同一個介面:
//      void attention_head(Q, K, V, O)
//  因此同一份 testbench 可以驗證任何 solution。
// =============================================================
#ifndef ATTENTION_H
#define ATTENTION_H

#include <ap_fixed.h>
#include <hls_math.h>

#define N   1024     // token 數
#define DK  128      // d_k = d_model / h = 4096 / 32

// FP16 半精度資料型別 (Vitis HLS 的 half)
typedef half data_t;

// 1/sqrt(128) 預先算好
static const data_t INV_SQRT_DK = (data_t)0.08838834764831845f;

// 所有 solution 都要實作這個函式
void attention_head(
    data_t Q[N][DK],
    data_t K[N][DK],
    data_t V[N][DK],
    data_t O[N][DK]);

#endif // ATTENTION_H