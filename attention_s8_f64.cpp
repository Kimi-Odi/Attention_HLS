// =============================================================
//  Solution #8 : 寬 QKt partition + softmax 優化  (往低延遲角落)
//
//  洞察 (來自 S7): QKt 的 II 嚴格由 K 的 partition 寬度決定:
//    factor 128->32->16->8  對應  II 1->2->4->8。
//    這是記憶體頻寬限制 (每拍能讀幾個 K 元素), 非計算相依性。
//
//  S8 策略: 結合「寬 QKt partition」+「已優化的 softmax (u4)」+
//    「PV 完全展開」, 把三個優化疊起來往低延遲角落推。
//    這次用 K cyclic factor=64 -> QKt II 應降到 ~1-2 (接近 S3 的速度),
//    但 DSP 會上升 (約 64 路 QKt 乘法器)。
//
//  與 S7 (factor=16) 的差異: 純粹是 K partition 從 16 加寬到 64。
//    QKt 變快 (II 降), DSP 變多。softmax/PV 部分完全相同。
//
//  目的: 不是打敗 S7 的 ATP, 而是在 ATP-delay 空間開出「低延遲 + 中面積」
//    的新 Pareto 點, 填補 S7(中) 和 S3(高面積低延遲) 之間。
//
//  !! K 的 cyclic factor 數字必須是字面常數 (HLS 不吃巨集) !!
// =============================================================
#include "attention.h"

#define QK_BANKS 64   // K partition 寬度 (這次加寬到 64)

void attention_head(
    data_t Q[N][DK],
    data_t K[N][DK],
    data_t V[N][DK],
    data_t O[N][DK])
{
    #pragma HLS ARRAY_PARTITION variable=Q complete dim=2
    #pragma HLS ARRAY_PARTITION variable=K cyclic factor=64 dim=2
    #pragma HLS ARRAY_PARTITION variable=V complete dim=2

    data_t S_row[N];

    Row_Loop: for (int i = 0; i < N; i++) {

        // ===== 1) S = Q.K^T : 內層展開 QK_BANKS, 加法樹合併 =====
        // DK/QK_BANKS = 128/64 = 2 組
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

        // ===== 2b) Exp + Sum : softmax unroll-by-4 (同 S6c/S7) =====
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

        // ===== 3) O = P.V (同 S6c/S7, 完全展開) =====
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