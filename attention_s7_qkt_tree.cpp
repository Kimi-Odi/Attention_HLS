// =============================================================
//  Solution #7 : QKt 加法樹  (打破 QKt 的 II=4 累加相依性)
//
//  診斷 (來自 S6c 實測): QKt_Loop II=4, 佔每列 37%。
//    根因: group_sum += psum 的外層串行累加 (DK/16 = 8 次串行)。
//    這和 softmax Exp_Loop 的 denom 累加是同一個病 (loop-carried dep)。
//
//  S7 策略: 把 QKt 內層的 8 次串行累加改成「完全展開 + 加法樹」。
//    8 個 psum 各自獨立算 (16 路並行乘法), 再用 log2(8)=3 深度的
//    明確樹狀表達式合併, 取代深度 8 的串行鏈 -> II 可望降到 1。
//
//  延續前面的教訓:
//    - 用「完全展開 + 明確命名」避免 S6 的動態索引陷阱 (護住 clock)。
//    - 整個 QKt 內積 (128 維) 變成: 128 路乘法 + 加法樹。
//      這其實接近 S3 的完全展開, 但乘法器數量由 cyclic factor=16 控制
//      (每拍 16 路, 8 拍完成所有乘法, 但累加用樹)。
//
//  基底 = S6c (softmax unroll-by-4, 目前最佳)。只改 QKt 內層累加結構。
//
//  關鍵風險:
//    A. clock 守住 + II 降 -> 每列降 ~28%, ATP 大幅改善。
//    B. clock 惡化 (加法樹拉長關鍵路徑) -> 可能淨輸 (S6 的教訓)。
// =============================================================
#include "attention.h"

#define QK_BANKS 16   // K cyclic factor=16 (同 S5c/S6 系列)

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

        // ===== 1) S = Q.K^T : 8 個 psum 完全展開 + 加法樹 =====
        QKt_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            // 8 個獨立部分和 (DK/QK_BANKS = 128/16 = 8 組)
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
            // 明確的 log 深度加法樹 (取代串行累加)
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

        // ===== 2b) Exp + Sum : softmax unroll-by-4 (同 S6c) =====
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

        // ===== 3) O = P.V (同 S6c, 完全展開) =====
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