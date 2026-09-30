// =============================================================
//  Solution #13 : factor=16 + PV 2組部分和  (挑戰 ATP 冠軍)
//
//  動機: ATP 冠軍區全是 QKt factor=16 (S7/S6c, ATP 5240~5290)。
//    它們的 PV 是完全展開但「1 組」(II=4, 4104 cyc, 佔每列 36%)。
//    S10 證明 PV 2 組部分和能讓 PV 減半 -- 但 S10 用在 factor=64
//    的高面積基底。把「PV 2 組」搬到 factor=16 的低 ENS 基底上,
//    PV 4104->2056 (delay -15%), ENS 只小增 (多 128 路 acc 的 FF)。
//
//  預期: delay 降 ~15% 勝過 ENS 增 ~14% -> ATP 可能跌破 S7 的 5240。
//    (臨界, 取決於 FF 增量與 clock, 需實測。)
//
//  基底 = S7 (factor=16 + softmax u4 + QKt 加法樹)。只把 PV 改 2 組。
//
//  關鍵風險:
//    A. clock 守住 + PV 減半 -> ATP 新低。
//    B. ENS 增太多 (FF) 或 clock 惡化 -> 持平/略差, 確認 S7 是 f16 的甜蜜點。
// =============================================================
#include "attention.h"

#define QK_BANKS 16

void attention_head(
    data_t Q[N][DK],
    data_t K[N][DK],
    data_t V[N][DK],
    data_t O[N][DK])
{
    #pragma HLS ARRAY_PARTITION variable=Q complete dim=2
    #pragma HLS ARRAY_PARTITION variable=K cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=V complete dim=2

    data_t S_row[N];

    Row_Loop: for (int i = 0; i < N; i++) {

        // ===== 1) S = Q.K^T : factor=16 + 加法樹 (同 S7) =====
        QKt_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t ps[8];
            #pragma HLS ARRAY_PARTITION variable=ps complete dim=1
            Group_K: for (int g = 0; g < 8; g++) {
                #pragma HLS UNROLL
                data_t s = (data_t)0;
                Inner_K: for (int kk = 0; kk < QK_BANKS; kk++) {
                    #pragma HLS UNROLL
                    s += Q[i][g*QK_BANKS+kk] * K[j][g*QK_BANKS+kk];
                }
                ps[g] = s;
            }
            data_t group_sum = ((ps[0] + ps[1]) + (ps[2] + ps[3]))
                             + ((ps[4] + ps[5]) + (ps[6] + ps[7]));
            S_row[j] = (data_t)(group_sum * INV_SQRT_DK);
        }

        // ===== 2a) Max =====
        data_t max_v = S_row[0];
        Max_Loop: for (int j = 1; j < N; j++) {
            #pragma HLS PIPELINE II=1
            if (S_row[j] > max_v) max_v = S_row[j];
        }

        // ===== 2b) Exp + Sum : softmax unroll-by-4 (同 S7) =====
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

        // ===== 3) O = P.V : 2 組部分和 (新, 取代 S7 的 1 組) =====
        data_t acc0[DK], acc1[DK];
        #pragma HLS ARRAY_PARTITION variable=acc0 complete dim=1
        #pragma HLS ARRAY_PARTITION variable=acc1 complete dim=1
        Init_Acc: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            acc0[d] = (data_t)0;
            acc1[d] = (data_t)0;
        }
        PV_Loop: for (int j = 0; j < N; j += 2) {
            #pragma HLS PIPELINE II=1
            data_t p0 = S_row[j];
            data_t p1 = S_row[j+1];
            MAC_D: for (int d = 0; d < DK; d++) {
                #pragma HLS UNROLL
                acc0[d] += p0 * V[j][d];
                acc1[d] += p1 * V[j+1][d];
            }
        }
        Write_O: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            O[i][d] = (data_t)(acc0[d] + acc1[d]);
        }
    }
}