// =============================================================
//  Solution #6d : SOFTMAX unroll-by-8  (確認 softmax 絕對極限)
//
//  目的: 找 softmax 優化的天花板。前面 u1->u2->u4 ATP 持續降
//        但報酬遞減 (-13.8% -> -7.6%), 預期 u8 < 5%。
//
//  關鍵風險: 8 個 hls::exp 並排, 組合邏輯寬度是 S6c 的 2 倍。
//    S6 (8累加器 + 動態索引 denom[j%8]) 已證明會讓 clock 爆到 18.47ns。
//    S6d 嚴格延續 S6b/S6c 的「8 個固定命名累加器, 無動態索引」公式,
//    測試這樣能否守住 clock。這是 S6 失敗變因的乾淨對照實驗:
//      S6  = 8 累加器 + 動態索引  -> clock 爆 (已知)
//      S6d = 8 累加器 + 固定命名  -> clock ? (待測)
//    若 S6d 守住 -> 證明元兇是「動態索引」而非「8 個累加器」。
//    若 S6d 也爆 -> 證明 8-way exp 本身就是 timing 上限。
//
//  基底 = S6c (QKt cyclic factor=16, PV 完全展開, softmax unroll-by-4)。
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

        // ===== 1) S = Q.K^T (同 S5c f16 / S6b / S6c) =====
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

        // ===== 2b) Exp + Sum : 8 個固定命名累加器, 一次處理 8 元素 =====
        data_t denom0 = (data_t)0, denom1 = (data_t)0;
        data_t denom2 = (data_t)0, denom3 = (data_t)0;
        data_t denom4 = (data_t)0, denom5 = (data_t)0;
        data_t denom6 = (data_t)0, denom7 = (data_t)0;
        Exp_Loop: for (int j = 0; j < N; j += 8) {
            #pragma HLS PIPELINE II=1
            data_t e0 = hls::exp((data_t)(S_row[j]   - max_v));
            data_t e1 = hls::exp((data_t)(S_row[j+1] - max_v));
            data_t e2 = hls::exp((data_t)(S_row[j+2] - max_v));
            data_t e3 = hls::exp((data_t)(S_row[j+3] - max_v));
            data_t e4 = hls::exp((data_t)(S_row[j+4] - max_v));
            data_t e5 = hls::exp((data_t)(S_row[j+5] - max_v));
            data_t e6 = hls::exp((data_t)(S_row[j+6] - max_v));
            data_t e7 = hls::exp((data_t)(S_row[j+7] - max_v));
            S_row[j]   = e0;  S_row[j+1] = e1;
            S_row[j+2] = e2;  S_row[j+3] = e3;
            S_row[j+4] = e4;  S_row[j+5] = e5;
            S_row[j+6] = e6;  S_row[j+7] = e7;
            denom0 += e0;  denom1 += e1;
            denom2 += e2;  denom3 += e3;
            denom4 += e4;  denom5 += e5;
            denom6 += e6;  denom7 += e7;
        }
        // 樹狀合併 8 個部分和 (在 pipeline 外)
        data_t denom = ((denom0 + denom1) + (denom2 + denom3))
                     + ((denom4 + denom5) + (denom6 + denom7));

        // ===== 2c) Norm =====
        data_t inv_denom = (data_t)((data_t)1 / denom);
        Norm_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            S_row[j] = S_row[j] * inv_denom;
        }

        // ===== 3) O = P.V (同 S5c f16 / S6b / S6c) =====
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