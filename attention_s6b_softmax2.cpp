// =============================================================
//  Solution #6b : SOFTMAX 優化 (修正 S6 的 clock 惡化)
//
//  S6 教訓: 8 個部分和 + mod 索引的 read-modify-write 鏈太長,
//           clock 從 11.411 爆到 18.473 ns (+62%), 淨延遲反而升 51%。
//
//  S6b 策略: 只用 2 個部分和 (奇/偶), 避免動態 mod 索引,
//           用兩個明確命名的累加器 denom0/denom1 交替。
//           這樣 read-modify-write 鏈最短, 合併只需 1 個 fadd,
//           試圖在「打破相依性」與「不拉長關鍵路徑」之間取得平衡。
//
//  基底 = S5c f16。唯一改動: Exp_Loop 用 2 個交替累加器。
//
//  關鍵問題 (待實測): 2 個累加器是否足以讓 II<4, 同時 clock 維持 ~11.4 ns?
//    若 clock 守住且 II 降 -> 贏。
//    若 clock 仍惡化或 II 沒降 -> 確認 S5c f16 的 II=4 是 timing 最佳點。
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

        // ===== 2a) Max =====
        data_t max_v = S_row[0];
        Max_Loop: for (int j = 1; j < N; j++) {
            #pragma HLS PIPELINE II=1
            if (S_row[j] > max_v) max_v = S_row[j];
        }

        // ===== 2b) Exp + Sum : 2 個交替累加器 (奇/偶), 無動態索引 =====
        data_t denom0 = (data_t)0;
        data_t denom1 = (data_t)0;
        Exp_Loop: for (int j = 0; j < N; j += 2) {
            #pragma HLS PIPELINE II=1
            // 一次處理兩個元素, 分別進兩個累加器 -> 相鄰加法無相依
            data_t e0 = hls::exp((data_t)(S_row[j]   - max_v));
            data_t e1 = hls::exp((data_t)(S_row[j+1] - max_v));
            S_row[j]   = e0;
            S_row[j+1] = e1;
            denom0 += e0;
            denom1 += e1;
        }
        data_t denom = denom0 + denom1;   // 合併只需 1 個 fadd

        // ===== 2c) Norm =====
        data_t inv_denom = (data_t)((data_t)1 / denom);
        Norm_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            S_row[j] = S_row[j] * inv_denom;
        }

        // ===== 3) O = P.V (同 S5c f16) =====
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