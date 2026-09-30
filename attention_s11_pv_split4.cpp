// =============================================================
//  Solution #11 : PV 4 組部分和  (推進 S10 的成功)
//
//  S10 成功: PV 2 組部分和 -> PV cycles 4104->2056 (減半), 新延遲冠軍。
//  S11 策略: 推到 4 組 (acc0~3), 一次處理 4 個 j, trip 1024->256。
//    完全對照 softmax u2->u4 的成功路徑 (那也是 2 組->4 組)。
//
//  基底 = S10 (PV 2組 + factor=64 + softmax u4)。只把 PV 改成 4 組。
//
//  延續所有教訓:
//    - 固定命名陣列 acc0~3, 無動態索引 (避開 S6 的 clock 陷阱)。
//    - 樹狀合併 O[d] = (acc0+acc1)+(acc2+acc3), 在迴圈外。
//
//  風險:
//    A. clock 守住 + PV 再減半 -> 延遲冠軍再進化 (~0.065s 預期)。
//    B. FF 大漲 (4 組 acc = 512 路 FP16 暫存), 但預期 ~70k (26% of 269k)。
//    C. clock 惡化 -> 確認 PV 2 組是甜蜜點 (像 softmax u4 是極限)。
// =============================================================
#include "attention.h"

#define QK_BANKS 64

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

        // ===== 1) S = Q.K^T (同 S8/S10) =====
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

        // ===== 2b) Exp + Sum : softmax unroll-by-4 (同 S6c/S8/S10) =====
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

        // ===== 3) O = P.V : 4 組部分和 (推進 S10 的 2 組) =====
        data_t acc0[DK], acc1[DK], acc2[DK], acc3[DK];
        #pragma HLS ARRAY_PARTITION variable=acc0 complete dim=1
        #pragma HLS ARRAY_PARTITION variable=acc1 complete dim=1
        #pragma HLS ARRAY_PARTITION variable=acc2 complete dim=1
        #pragma HLS ARRAY_PARTITION variable=acc3 complete dim=1
        Init_Acc: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            acc0[d] = (data_t)0;  acc1[d] = (data_t)0;
            acc2[d] = (data_t)0;  acc3[d] = (data_t)0;
        }
        // 一次處理 4 個 j: j->acc0, j+1->acc1, j+2->acc2, j+3->acc3。
        // 連續 4 次累加落在不同陣列 -> 打破相依性 -> II 維持但 trip 減半。
        PV_Loop: for (int j = 0; j < N; j += 4) {
            #pragma HLS PIPELINE II=1
            data_t p0 = S_row[j];
            data_t p1 = S_row[j+1];
            data_t p2 = S_row[j+2];
            data_t p3 = S_row[j+3];
            MAC_D: for (int d = 0; d < DK; d++) {
                #pragma HLS UNROLL
                acc0[d] += p0 * V[j][d];
                acc1[d] += p1 * V[j+1][d];
                acc2[d] += p2 * V[j+2][d];
                acc3[d] += p3 * V[j+3][d];
            }
        }
        // 樹狀合併 4 組部分和
        Write_O: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            O[i][d] = (data_t)((acc0[d] + acc1[d]) + (acc2[d] + acc3[d]));
        }
    }
}