// =============================================================
//  Solution #3 : LOW-LATENCY  (低延遲方向)
//  策略:
//    (a) 內層 dot-product 完全 UNROLL -> 形成加法樹 (log 深度)
//    (b) 對 Q/K/V 在 DK 維度做 ARRAY_PARTITION complete
//        -> 一個 cycle 可同時讀出 128 個元素, 餵給 128 個乘法器
//    (c) 外層 QKt_Loop / PV_Loop 做 PIPELINE II=1
//        -> 每個 cycle 發射一個完整的 dot-product
//  特性: 大量 DSP (約 128+ 個乘法器), 延遲大幅下降
//  定位: Pareto 上的低延遲 / 高面積角落
//  注意: 在 Artix-7 上 DSP 可能不夠 (xc7a200t 有 740 個 DSP),
//        若資源爆掉請改用 Virtex-7 (xc7vx485t 有 2800 個 DSP)。
// =============================================================
#include "attention.h"

void attention_head(
    data_t Q[N][DK],
    data_t K[N][DK],
    data_t V[N][DK],
    data_t O[N][DK])
{
    // ---- 對 DK 維度 (第 2 維) 做完整切割, 支援同時讀取 ----
    #pragma HLS ARRAY_PARTITION variable=Q complete dim=2
    #pragma HLS ARRAY_PARTITION variable=K complete dim=2
    // V 在 PV 階段是沿 j(=N) 方向累加, 對 DK 維切割可平行輸出
    #pragma HLS ARRAY_PARTITION variable=V complete dim=2

    data_t S_row[N];

    Row_Loop: for (int i = 0; i < N; i++) {

        // ---- 1) S = Q.K^T : 每個 j 一個 cycle (內層全展開成加法樹) ----
        QKt_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t prod[DK];
            #pragma HLS ARRAY_PARTITION variable=prod complete dim=1
            Mul_K: for (int k = 0; k < DK; k++) {
                #pragma HLS UNROLL
                prod[k] = Q[i][k] * K[j][k];
            }
            // 加法樹 reduction
            data_t sum = (data_t)0;
            Add_K: for (int k = 0; k < DK; k++) {
                #pragma HLS UNROLL
                sum += prod[k];
            }
            S_row[j] = (data_t)(sum * INV_SQRT_DK);
        }

        // ---- 2) softmax (此處仍逐元素, 非瓶頸) ----
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

        // ---- 3) O = P.V : DK 個輸出同時算 (對 d 展開) ----
        // 每個 j 把 S_row[j] 廣播到 DK 個累加器
        data_t acc[DK];
        #pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        Init_Acc: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            acc[d] = (data_t)0;
        }
        PV_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t p = S_row[j];
            MAC_D: for (int d = 0; d < DK; d++) {
                #pragma HLS UNROLL
                acc[d] += p * V[j][d];
            }
        }
        Write_O: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            O[i][d] = (data_t)acc[d];
        }
    }
}