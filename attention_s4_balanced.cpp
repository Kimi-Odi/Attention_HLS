// =============================================================
//  Solution #4 : BALANCED  (平衡 / 低 ATP 甜蜜點)
//  策略:
//    (a) 內層 dot-product 部分展開 (factor = UF), 非全展開
//    (b) 對應做 ARRAY_PARTITION cyclic factor=UF
//        -> 每 cycle 讀出 UF 個元素, 用 UF 個乘法器
//    (c) 外層迴圈 PIPELINE II=1
//  特性: DSP 用量適中 (~2*UF 個乘法器), 延遲比 baseline 降約 UF 倍
//  定位: 通常是 ATP 最低點附近 -> 面積與延遲的良好平衡
//
//  可調參數: UF (unroll factor)。先設 16, 之後可掃 8/16/32 找最佳 ATP。
// =============================================================
#include "attention.h"

#define UF 16   // 展開因子 (DK 與 N 都能被 16 整除)

void attention_head(
    data_t Q[N][DK],
    data_t K[N][DK],
    data_t V[N][DK],
    data_t O[N][DK])
{
    // ---- 對展開維度做 cyclic 切割, 提供 UF 個並行讀取埠 ----
    #pragma HLS ARRAY_PARTITION variable=Q cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=K cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=V cyclic factor=16 dim=2

    data_t S_row[N];

    Row_Loop: for (int i = 0; i < N; i++) {

        // ---- 1) S = Q.K^T (內層部分展開 factor=UF) ----
        QKt_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t sum = (data_t)0;
            Dot_Loop: for (int k = 0; k < DK; k++) {
                #pragma HLS UNROLL factor=16
                sum += Q[i][k] * K[j][k];
            }
            S_row[j] = (data_t)(sum * INV_SQRT_DK);
        }

        // ---- 2) softmax ----
        data_t max_v = S_row[0];
        Max_Loop: for (int j = 1; j < N; j++) {
            #pragma HLS PIPELINE II=1
            if (S_row[j] > max_v) max_v = S_row[j];
        }
        data_t denom = (data_t)0;
        Exp_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t e = hls::exp((data_t)(S_row[j] - max_v));
            S_row[j] = e;
            denom += e;
        }
        data_t inv_denom = (data_t)((data_t)1 / denom);
        Norm_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            S_row[j] = S_row[j] * inv_denom;
        }

        // ---- 3) O = P.V (內層部分展開 factor=UF) ----
        PV_Loop: for (int d = 0; d < DK; d++) {
            #pragma HLS PIPELINE II=1
            data_t acc = (data_t)0;
            PV_Inner: for (int j = 0; j < N; j++) {
                #pragma HLS UNROLL factor=16
                acc += S_row[j] * V[j][d];
            }
            O[i][d] = (data_t)acc;
        }
    }
}