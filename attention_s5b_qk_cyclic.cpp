// =============================================================
//  Solution #5b : PV-full / QKt 真正削減  (修正 S5 的無效削減)
//
//  S5 的教訓: 對 Q/K 用 complete partition + II=1, Vitis 會無視
//  UNROLL factor=32, 仍完全展開成 128 乘法器 -> S5 == S3。
//  => 控制乘法器數量的是「partition 寬度」, 不是 UNROLL factor。
//
//  S5b 的修正:
//    (a) PV 階段: 維持 complete partition + 完全展開 (同 S3, 已驗證)。
//    (b) QKt 階段: K 改用 cyclic factor=32 (而非 complete)。
//        每 cycle 只能讀出 32 個 K 元素 -> Vitis 最多用 32 乘法器。
//        Q 仍 complete (Q[i][k] 中 i 固定, 整列可平行讀)。
//
//  避開 S4 mux 陷阱的理由:
//    cyclic 是否產生選路 mux, 取決於存取索引是否與 bank 對齊。
//    QKt 內層 k 連續存取 (0,1,...,127 輪流落在 32 bank) -> 對齊 -> 無 mux。
//    S4 的錯誤是把 cyclic 用在 PV (V[j][d], j 外層變動) -> 不對齊 -> mux 爆。
//    S5b 的 PV 用 complete, 只有對齊的 QKt 用 cyclic。
//
//  QKt 內層用「部分積 + 加法樹」避免單一 sum 的 loop-carried dependency:
//    32 個乘法器各自算部分積 -> 分組累加 -> 降低 II。
// =============================================================
#include "attention.h"

#define QK_BANKS 32   // QKt 的 K 切成 32 個 bank (128/32 = 4 拍讀完)

void attention_head(
    data_t Q[N][DK],
    data_t K[N][DK],
    data_t V[N][DK],
    data_t O[N][DK])
{
    // Q: complete (整列同時讀, i 固定)
    #pragma HLS ARRAY_PARTITION variable=Q complete dim=2
    // K: cyclic factor=32 -> 限制 QKt 每拍只讀 32 個 -> 最多 32 乘法器
    #pragma HLS ARRAY_PARTITION variable=K cyclic factor=32 dim=2
    // V: complete (PV 階段完全展開, 同 S3, 無 mux)
    #pragma HLS ARRAY_PARTITION variable=V complete dim=2

    data_t S_row[N];

    Row_Loop: for (int i = 0; i < N; i++) {

        // ===== 1) S = Q.K^T : 每拍處理 32 個 k, 共 4 拍 =====
        // 外層對 j pipeline; 內層分 4 組, 每組 32 個乘法器
        QKt_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t group_sum = (data_t)0;
            // 分 DK/QK_BANKS = 4 拍, 每拍 32 個 MAC (這 32 個並行)
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

        // ===== 2) softmax (固定開銷) =====
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

        // ===== 3) O = P.V : 完全展開 d_k (同 S3) =====
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