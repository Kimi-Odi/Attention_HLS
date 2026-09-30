// =============================================================
//  Solution #10 : PV 部分和  (打破 PV 的 II=4 計算相依性)
//
//  診斷 (已驗證, 與 QKt 不同):
//    PV_Loop II=4 的根因是 acc[d] += p*V[j][d] 跨 j 的累加相依性
//    (loop-carried dependency), 不是記憶體頻寬 (V 已 complete partition,
//    128 路全平行讀)。證據: 所有版本 V 都 complete 但 PV II 都=4。
//    => 這和 softmax denom 累加是同一個病, 應該能用「部分和」打破。
//
//  S10 策略 (套用 S6b 對 softmax 成功的技巧):
//    把 acc[d] 拆成 2 組: acc0[d] (處理偶數批 j) + acc1[d] (奇數批 j)。
//    一次處理 2 個 j (j 和 j+1), 各自累加到不同的 acc 組,
//    相鄰 j 的累加落在不同陣列 -> 打破相依性 -> PV II 可望降。
//    最後把 acc0[d] + acc1[d] 合併成 O[i][d]。
//
//  基底 = S8 (factor=64, 延遲冠軍)。只改 PV 的累加結構。
//
//  關鍵風險:
//    A. clock 守住 + PV II 降 -> S8 延遲再大降 (PV 佔 47%!), 延遲冠軍進化。
//    B. clock 惡化 (2 組 acc 各 128 路, register/邏輯加倍) -> 可能淨輸。
//       (但用固定命名陣列, 無動態索引, 應比 S6 安全。)
//
//  資源預期: acc register 加倍 (256 路 FP16 暫存), FF 會明顯上升。
// =============================================================
#include "attention.h"

#define QK_BANKS 64   // 同 S8 (延遲冠軍基底)

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

        // ===== 1) S = Q.K^T (同 S8) =====
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

        // ===== 2b) Exp + Sum : softmax unroll-by-4 (同 S6c/S8) =====
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

        // ===== 3) O = P.V : 2 組部分和打破累加相依性 =====
        data_t acc0[DK];   // 處理偶數批 j
        data_t acc1[DK];   // 處理奇數批 j
        #pragma HLS ARRAY_PARTITION variable=acc0 complete dim=1
        #pragma HLS ARRAY_PARTITION variable=acc1 complete dim=1
        Init_Acc: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            acc0[d] = (data_t)0;
            acc1[d] = (data_t)0;
        }
        // 一次處理 2 個 j: j -> acc0, j+1 -> acc1。
        // 相鄰 j 累加到不同陣列 -> 連續兩次累加無相依 -> II 可望降。
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
        // 合併 2 組部分和
        Write_O: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            O[i][d] = (data_t)(acc0[d] + acc1[d]);
        }
    }
}