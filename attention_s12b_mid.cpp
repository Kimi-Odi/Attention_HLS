// =============================================================
//  Solution #12a : 中等並行度  (填 Pareto 右下角空檔)
//
//  動機: 前緣在 delay 0.22s (S5c_f8) 到 12.8s (S2) 之間有 ~59x 的
//    巨大空檔。原因: 所有 S3-S11 都拼低延遲 (PV 完全展開 128 路,
//    DSP 100+), 而 S1/S2 是純循序 (DSP <10)。中等並行度沒人探索。
//
//  S12b 策略: 大幅降低並行度, 換中等延遲、低面積。
//    關鍵改動: PV 不再完全展開 d, 只展開 PV_UF=8 路。
//      -> 每個 j 要 DK/PV_UF = 16 拍, PV 慢很多, 但只用 8 個乘法器。
//    QKt 也用小 factor=8 (8 路乘法器)。
//    softmax 維持 u4 (它便宜又有效, 沒理由退化)。
//
//  預期落點: DSP ~10-30, delay ~0.3s, ENS ~10000-15000。
//    填補右下角, 讓前緣從低延遲區連到低面積區。
//
//  !! K cyclic factor 數字必須是字面常數 !!
// =============================================================
#include "attention.h"

#define QK_BANKS 4    // QKt: 小 factor (8 路乘法器)
#define PV_UF    4    // PV: 只展開 8 路 d (不是 128)

void attention_head(
    data_t Q[N][DK],
    data_t K[N][DK],
    data_t V[N][DK],
    data_t O[N][DK])
{
    // Q/K 仍對 DK partition 以支援 QKt, 但 factor 較小。
    #pragma HLS ARRAY_PARTITION variable=Q cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=K cyclic factor=4 dim=2
    // V: 只需 PV_UF 路平行讀 -> cyclic factor=8 (不是 complete)
    #pragma HLS ARRAY_PARTITION variable=V cyclic factor=4 dim=2

    data_t S_row[N];

    Row_Loop: for (int i = 0; i < N; i++) {

        // ===== 1) S = Q.K^T : QKt factor=8 =====
        QKt_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t group_sum = (data_t)0;
            Outer_K: for (int kb = 0; kb < DK; kb += QK_BANKS) {
                data_t psum = (data_t)0;
                Inner_K: for (int kk = 0; kk < QK_BANKS; kk++) {
                    #pragma HLS UNROLL
                    psum += Q[i][kb+kk] * K[j][kb+kk];
                }
                group_sum += psum;
            }
            S_row[j] = (data_t)(group_sum * INV_SQRT_DK);
        }

        // ===== 2a) Max =====
        data_t max_v = S_row[0];
        Max_Loop: for (int j = 1; j < N; j++) {
            #pragma HLS PIPELINE II=1
            if (S_row[j] > max_v) max_v = S_row[j];
        }

        // ===== 2b) Exp + Sum : softmax unroll-by-4 (維持, 便宜有效) =====
        data_t denom0 = (data_t)0, denom1 = (data_t)0;
        data_t denom2 = (data_t)0, denom3 = (data_t)0;
        Exp_Loop: for (int j = 0; j < N; j += 4) {
            #pragma HLS PIPELINE II=1
            data_t e0 = hls::exp((data_t)(S_row[j]   - max_v));
            data_t e1 = hls::exp((data_t)(S_row[j+1] - max_v));
            data_t e2 = hls::exp((data_t)(S_row[j+2] - max_v));
            data_t e3 = hls::exp((data_t)(S_row[j+3] - max_v));
            S_row[j]   = e0;  S_row[j+1] = e1;
            S_row[j+2] = e2;  S_row[j+3] = e3;
            denom0 += e0;  denom1 += e1;
            denom2 += e2;  denom3 += e3;
        }
        data_t denom = (denom0 + denom1) + (denom2 + denom3);

        // ===== 2c) Norm =====
        data_t inv_denom = (data_t)((data_t)1 / denom);
        Norm_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            S_row[j] = S_row[j] * inv_denom;
        }

        // ===== 3) O = P.V : PV 只展開 PV_UF 路 d (中等並行) =====
        // acc 仍是 128 路暫存, 但每拍只更新 PV_UF 個 (外層 j, 內層 d 分塊)
        data_t acc[DK];
        #pragma HLS ARRAY_PARTITION variable=acc cyclic factor=4 dim=1
        Init_Acc: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL factor=4
            acc[d] = (data_t)0;
        }
        // 外層 j (1024), 內層 d 分成 DK/PV_UF=16 塊, 每塊 PV_UF 路並行
        PV_Loop: for (int j = 0; j < N; j++) {
            data_t p = S_row[j];
            PV_D_Outer: for (int db = 0; db < DK; db += PV_UF) {
                #pragma HLS PIPELINE II=1
                PV_D_Inner: for (int dd = 0; dd < PV_UF; dd++) {
                    #pragma HLS UNROLL
                    acc[db+dd] += p * V[j][db+dd];
                }
            }
        }
        Write_O: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL factor=4
            O[i][d] = (data_t)acc[d];
        }
    }
}