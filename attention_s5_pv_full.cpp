// =============================================================
//  Solution #5 : PV-FULL / QKt-REDUCED  (修正 PV 瓶頸, 削減 QKt)
//
//  設計理念 (根據 S3 latency 分解):
//    softmax 佔 52% 延遲 (固定開銷), QKt 只佔 14%, PV 佔 34%。
//    => PV 是關鍵, 必須維持 S3 的完全展開 (已驗證可 pipeline)。
//    => QKt 佔比小, 可削減並行度來大幅省 DSP, 延遲只小幅增加。
//
//  策略:
//    (a) PV 階段: 完全展開 d_k (128 累加器), 對 V 做 complete partition
//        — 與 S3 完全相同, 已知 II 可達成, 無 mux 負擔。
//    (b) QKt 階段: 內層 dot-product 只展開 factor=QK_UF (預設 32),
//        即用 ~32 乘法器而非 128。對 Q/K 做 complete partition
//        (注意: 不用 cyclic, 避免 S4 的 mux 爆炸陷阱)。
//
//  避開 S4 陷阱的關鍵:
//    S4 用 cyclic factor=16 -> 產生大量位址選路 mux (LUT 爆 78%)。
//    S5 對 Q/K/V 全部用 complete partition (拆成獨立暫存器),
//    QKt 內層用 UNROLL factor 讓 Vitis 自行打包乘法器, 不產生 bank mux。
//
//  預期: DSP 從 S3 的 519 大幅下降 (QKt 少用 ~96 乘法器),
//        延遲只增加 QKt 那 14% 的一部分 -> 往 Pareto 左下推。
// =============================================================
#include "attention.h"

#define QK_UF 32   // QKt 內層展開因子 (128 能被 32 整除)

void attention_head(
    data_t Q[N][DK],
    data_t K[N][DK],
    data_t V[N][DK],
    data_t O[N][DK])
{
    // Q/K/V 全部對 DK 維度做 complete partition (拆成獨立暫存器, 無 mux)
    #pragma HLS ARRAY_PARTITION variable=Q complete dim=2
    #pragma HLS ARRAY_PARTITION variable=K complete dim=2
    #pragma HLS ARRAY_PARTITION variable=V complete dim=2

    data_t S_row[N];

    Row_Loop: for (int i = 0; i < N; i++) {

        // ===== 1) S = Q.K^T : QKt 內層只展開 factor=QK_UF =====
        // 外層 pipeline, 內層部分展開 -> 用 ~QK_UF 個乘法器
        QKt_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t sum = (data_t)0;
            Dot_Loop: for (int k = 0; k < DK; k++) {
                #pragma HLS UNROLL factor=32
                sum += Q[i][k] * K[j][k];
            }
            S_row[j] = (data_t)(sum * INV_SQRT_DK);
        }

        // ===== 2) softmax (固定開銷, 與並行度無關) =====
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

        // ===== 3) O = P.V : 完全展開 d_k (與 S3 相同, 128 累加器) =====
        // 這是 S3 已驗證可 pipeline 的關鍵結構, 維持不變
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
