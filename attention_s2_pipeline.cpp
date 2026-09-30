// =============================================================
//  Solution #2 : PIPELINE ONLY  (只加管線化)
//  策略: 在內層 dot-product 迴圈加 #pragma HLS PIPELINE II=1
//        讓相鄰的乘加運算重疊執行 (前一筆還沒算完, 下一筆就進來)
//  特性: DSP 幾乎不增加, 但延遲大幅下降 -> Pareto 上的高 CP 值起步點
//  注意: 內層做 floating-point 累加會有 loop-carried dependency,
//        Vitis 仍可達到不錯的 II, 但 II 不一定能到 1 (受 fadd 延遲影響)。
//        若要強制 II=1, 後續 S4 會用「部分展開 + 加法樹」打破相依性。
// =============================================================
#include "attention.h"

void attention_head(
    data_t Q[N][DK],
    data_t K[N][DK],
    data_t V[N][DK],
    data_t O[N][DK])
{
    data_t S_row[N];

    Row_Loop: for (int i = 0; i < N; i++) {

        // ---- 1) S = Q.K^T ----
        QKt_Loop: for (int j = 0; j < N; j++) {
            data_t sum = (data_t)0;
            Dot_Loop: for (int k = 0; k < DK; k++) {
                #pragma HLS PIPELINE II=1
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

        // ---- 3) O = P.V ----
        PV_Loop: for (int d = 0; d < DK; d++) {
            data_t acc = (data_t)0;
            PV_Inner: for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                acc += S_row[j] * V[j][d];
            }
            O[i][d] = (data_t)acc;
        }
    }
}