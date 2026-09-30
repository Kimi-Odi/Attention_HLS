// =============================================================
//  Solution #6c : SOFTMAX unroll-by-4  (推進 S6b 的成功)
//
//  S6b 成功經驗: unroll-by-2 (2 個固定累加器, 無動態索引),
//    Exp_Loop trip 1024->512, clock 守住 11.411 ns, ATP 改善 13.8%。
//  S6 失敗教訓: 8 個累加器 + denom[j%8] 動態索引 -> clock 爆 18.47 ns。
//
//  S6c 策略: 延續 S6b 模式, 擴展到 4 個固定累加器 (denom0~3),
//    一次處理連續 4 個元素 (trip 256)。
//    刻意維持「固定命名、無動態索引」以避免 S6 的關鍵路徑陷阱。
//
//  基底 = S6b (QKt cyclic factor=16, PV 完全展開, softmax unroll)。
//
//  關鍵風險 (待實測):
//    4 個 hls::exp 實例並排, 組合邏輯比 S6b 寬一倍。
//    若 clock 守住 ~11.4 ns -> 每列再省 ~8%, ATP 再降。
//    若 clock 惡化 (像 S6) -> 確認 unroll-by-2 已是 softmax 的甜蜜點。
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

        // ===== 1) S = Q.K^T (同 S5c f16 / S6b) =====
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

        // ===== 2b) Exp + Sum : 4 個固定累加器, 一次處理 4 元素 =====
        data_t denom0 = (data_t)0;
        data_t denom1 = (data_t)0;
        data_t denom2 = (data_t)0;
        data_t denom3 = (data_t)0;
        Exp_Loop: for (int j = 0; j < N; j += 4) {
            #pragma HLS PIPELINE II=1
            data_t e0 = hls::exp((data_t)(S_row[j]   - max_v));
            data_t e1 = hls::exp((data_t)(S_row[j+1] - max_v));
            data_t e2 = hls::exp((data_t)(S_row[j+2] - max_v));
            data_t e3 = hls::exp((data_t)(S_row[j+3] - max_v));
            S_row[j]   = e0;
            S_row[j+1] = e1;
            S_row[j+2] = e2;
            S_row[j+3] = e3;
            denom0 += e0;
            denom1 += e1;
            denom2 += e2;
            denom3 += e3;
        }
        // 樹狀合併 4 個部分和 (僅 3 個 fadd, 在 pipeline 外)
        data_t denom = (denom0 + denom1) + (denom2 + denom3);

        // ===== 2c) Norm =====
        data_t inv_denom = (data_t)((data_t)1 / denom);
        Norm_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            S_row[j] = S_row[j] * inv_denom;
        }

        // ===== 3) O = P.V (同 S5c f16 / S6b) =====
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