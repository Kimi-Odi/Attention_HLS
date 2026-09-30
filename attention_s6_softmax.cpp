// =============================================================
//  Solution #6 : SOFTMAX 優化  (打破 Exp_Loop 累加相依性)
//
//  診斷 (來自 S5c f16 實測 Loop 表):
//    Exp_Loop II=4, 佔每列 28.4% (4109 cyc)。
//    根因: denom += e 的 loop-carried dependency。FP16 fadd 多週期,
//          下一次累加要等上一次完成 -> II 卡在 4。
//    (這和當初 QKt baseline 的累加瓶頸完全相同。)
//
//  策略 (沿用 S5b/S5c 驗證過的技巧):
//    用 SUM_BANKS 個獨立部分和累加器輪流接收 exp 結果,
//    相鄰加法落在不同累加器 -> 不再互相依賴 -> II 可望降到 1。
//    最後把 SUM_BANKS 個部分和樹狀合併成 denom。
//
//  基底架構 = S5c f16 (目前 ATP 最佳: QKt cyclic factor=16, PV 完全展開)。
//  唯一改動: Exp_Loop 的累加結構。其餘完全不變。
//
//  預期: Exp_Loop II 4->1, 每列省 ~3072 cyc (-21%),
//        全機延遲同比例下降 -> 整條 Pareto 前緣左移。
//        資源幾乎不變 (只多幾個 FP16 加法器做部分和合併)。
// =============================================================
#include "attention.h"

#define QK_BANKS  16   // QKt: 同 S5c f16 (已知 ATP 最佳)
#define SUM_BANKS 8    // softmax denom 的部分和數量 (打破相依性)

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

        // ===== 1) S = Q.K^T (同 S5c f16) =====
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

        // ===== 2a) Max (同前, 已 II=1) =====
        data_t max_v = S_row[0];
        Max_Loop: for (int j = 1; j < N; j++) {
            #pragma HLS PIPELINE II=1
            if (S_row[j] > max_v) max_v = S_row[j];
        }

        // ===== 2b) Exp + Sum : 用 SUM_BANKS 個部分和打破相依性 =====
        data_t denom_part[SUM_BANKS];
        #pragma HLS ARRAY_PARTITION variable=denom_part complete dim=1
        Init_Denom: for (int b = 0; b < SUM_BANKS; b++) {
            #pragma HLS UNROLL
            denom_part[b] = (data_t)0;
        }
        // 每個 j 把 exp 結果加到 denom_part[j % SUM_BANKS]。
        // 相鄰的 j 落在不同累加器 -> 連續 SUM_BANKS 次加法無相依 -> II=1。
        Exp_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t e = hls::exp((data_t)(S_row[j] - max_v));
            S_row[j] = e;
            denom_part[j % SUM_BANKS] += e;
        }
        // 樹狀合併 SUM_BANKS 個部分和
        data_t denom = (data_t)0;
        Reduce_Denom: for (int b = 0; b < SUM_BANKS; b++) {
            #pragma HLS UNROLL
            denom += denom_part[b];
        }

        // ===== 2c) Norm (同前, 已 II=1) =====
        data_t inv_denom = (data_t)((data_t)1 / denom);
        Norm_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            S_row[j] = S_row[j] * inv_denom;
        }

        // ===== 3) O = P.V (同 S5c f16, 完全展開) =====
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