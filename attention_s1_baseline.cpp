// =============================================================
//  Solution #1 : BASELINE  (完全不優化)
//  策略: 純循序三層迴圈, 單埠 BRAM, 無任何 pragma
//  定位: 高 area-time / 低資源的參考點 (Pareto 探索起點)
// =============================================================
#include "attention.h"

void attention_head(
    data_t Q[N][DK],
    data_t K[N][DK],
    data_t V[N][DK],
    data_t O[N][DK])
{
    // 所有陣列預設映射到單埠 BRAM, 不做 partition, 不做 pipeline/unroll

    data_t S_row[N];   // 一次處理一列的分數暫存

    Row_Loop: for (int i = 0; i < N; i++) {

        // ---- 1) S_row[j] = (Q[i] . K[j]) * scale ----
        QKt_Loop: for (int j = 0; j < N; j++) {
            data_t sum = (data_t)0;
            Dot_Loop: for (int k = 0; k < DK; k++) {
                sum += Q[i][k] * K[j][k];
            }
            S_row[j] = (data_t)(sum * INV_SQRT_DK);
        }

        // ---- 2) softmax (含 max 相減做數值穩定) ----
        data_t max_v = S_row[0];
        Max_Loop: for (int j = 1; j < N; j++) {
            if (S_row[j] > max_v) max_v = S_row[j];
        }

        data_t denom = (data_t)0;
        Exp_Loop: for (int j = 0; j < N; j++) {
            data_t e = hls::exp((data_t)(S_row[j] - max_v));
            S_row[j] = e;
            denom += e;
        }

        data_t inv_denom = (data_t)((data_t)1 / denom);
        Norm_Loop: for (int j = 0; j < N; j++) {
            S_row[j] = S_row[j] * inv_denom;   // 即 P[i][j]
        }

        // ---- 3) O[i] = P_row . V ----
        PV_Loop: for (int d = 0; d < DK; d++) {
            data_t acc = (data_t)0;
            PV_Inner: for (int j = 0; j < N; j++) {
                acc += S_row[j] * V[j][d];
            }
            O[i][d] = (data_t)acc;
        }
    }
}