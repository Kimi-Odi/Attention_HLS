// =============================================================
//  Solution #19 : 三階段融合 (loop fusion)  -- 演算法級優化
//
//  洞察: S16 有三個獨立的 j 迴圈 (QKt, Exp, PV), 每個掃 1024 個 j。
//    但延遲正規化讓我們可以把它們「融合成一個 j 迴圈」:
//    每個 j 一口氣算 S[j] -> exp -> 累加到 acc[all d]。
//    每列只掃一次 j (而非三次)!
//
//  融合後每列: 1 個 j 迴圈, 每個 j 做:
//    1. QKt 單點: S = Q_row . K[j]  (128 MAC + 加法樹)
//    2. exp: E = exp(S), denom += E
//    3. PV 單點: acc[d] += E * V[j][d]  (128 MAC 累加)
//
//  為什麼這不會像 S18 (dataflow) 失敗:
//    S18 用 FIFO stream 串接 stage, PV 從 stream 讀 -> 序列化破壞 II。
//    融合是「同一個 loop body」, S[j]->E->PV 用暫存器/wire 傳遞,
//    沒有 stream, 沒有跨 task 邊界 -> II 不受 stream 影響。
//
//  潛力: 每列 9339 -> ~4200 (三次掃 j 變一次), 延遲降 ~55%。
//  代價: 需要 256 乘法器 (128 QKt + 128 PV 同時), DSP 增加。
//
//  基底邏輯 = S16 (factor=16, 延遲正規化, 省 max)。
//  風險: loop body 大, 關鍵路徑可能變長 (clock); II 可能 >4。
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

    Row_Loop: for (int i = 0; i < N; i++) {

        // 載入當前列的 Q
        data_t Q_row[DK];
        #pragma HLS ARRAY_PARTITION variable=Q_row complete dim=1
        Load_Q: for (int k = 0; k < DK; k++) {
            #pragma HLS UNROLL
            Q_row[k] = Q[i][k];
        }

        // PV 累加器 (未正規化), 在 j 迴圈外初始化
        data_t acc[DK];
        #pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        Init_Acc: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            acc[d] = (data_t)0;
        }

        data_t denom = (data_t)0;

        // ===== 融合 j 迴圈: QKt + exp + PV 一次做完 =====
        Fused_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1

            // --- (1) QKt 單點: S = Q_row . K[j] ---
            data_t ps[8];
            #pragma HLS ARRAY_PARTITION variable=ps complete dim=1
            Group_K: for (int g = 0; g < 8; g++) {
                #pragma HLS UNROLL
                data_t s = (data_t)0;
                Inner_K: for (int kk = 0; kk < QK_BANKS; kk++) {
                    #pragma HLS UNROLL
                    s += Q_row[g*QK_BANKS+kk] * K[j][g*QK_BANKS+kk];
                }
                ps[g] = s;
            }
            data_t group_sum = ((ps[0] + ps[1]) + (ps[2] + ps[3]))
                             + ((ps[4] + ps[5]) + (ps[6] + ps[7]));
            data_t S = (data_t)(group_sum * INV_SQRT_DK);

            // --- (2) exp (無 max 相減), 累加 denom ---
            data_t E = hls::exp(S);
            denom += E;

            // --- (3) PV 單點: 累加到所有 acc[d] ---
            PV_MAC: for (int d = 0; d < DK; d++) {
                #pragma HLS UNROLL
                acc[d] += E * V[j][d];
            }
        }

        // ===== 正規化輸出 (延遲正規化) =====
        data_t inv_denom = (data_t)((data_t)1 / denom);
        Write_O: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            O[i][d] = (data_t)(acc[d] * inv_denom);
        }
    }
}