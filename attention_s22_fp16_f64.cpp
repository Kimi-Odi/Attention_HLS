// =============================================================
//  Solution #22 : S16 + factor=64 (FP16 合規, 降 QKt II)
//
//  目標: 在 FP16 規格內 (題目要求) 繼續壓 ATP。
//    S16 (FP16, 數學簡化) ATP 3808 是目前 FP16 合規最佳。
//    S16 每列: QKt(4200,II=4) + Exp(1035) + PV(4104,II=4) = 9339
//
//  S16 的 QKt II=4 來源 = 記憶體頻寬 (factor=16, 每拍讀 16 個 K)。
//    這和定點無關 -> FP16 加寬 factor 也能降 QKt II。
//    factor=64 -> QKt II 4->1 (記憶體頻寬解放, 全 FP16)。
//
//  注意: PV 還是 II=4 (FP16 fadd 相依, 無法像定點降到 1)。
//    所以 PV (4104) 仍是大頭, QKt 變快但上限受 PV 限制。
//
//  關鍵不確定: factor=64 的 QKt 乘法器是否增 DSP?
//    S21 (定點) 加寬 QKt 時 DSP 沒增 (乘法器共用)。
//    FP16 是否也共用? 若共用 -> ATP 可能突破 3808。
//
//  基底 = S16 (數學簡化: 省 max + 延遲正規化), 全 FP16。
//    只把 QKt 從 factor=16 改成 factor=64。
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

    data_t E_row[N];   // 未正規化的 exp 值

    Row_Loop: for (int i = 0; i < N; i++) {

        // ===== 1) S = Q.K^T : factor=64 (QKt II=1) =====
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
            E_row[j] = (data_t)(group_sum * INV_SQRT_DK);   // 先存 S
        }

        // ===== 2) Exp (無 max) + Sum : softmax u4, 未正規化 E =====
        data_t denom0 = (data_t)0, denom1 = (data_t)0;
        data_t denom2 = (data_t)0, denom3 = (data_t)0;
        Exp_Loop: for (int j = 0; j < N; j += 4) {
            #pragma HLS PIPELINE II=1
            data_t e0 = hls::exp((data_t)E_row[j]);
            data_t e1 = hls::exp((data_t)E_row[j+1]);
            data_t e2 = hls::exp((data_t)E_row[j+2]);
            data_t e3 = hls::exp((data_t)E_row[j+3]);
            E_row[j]   = e0;  E_row[j+1] = e1;
            E_row[j+2] = e2;  E_row[j+3] = e3;
            denom0 += e0;  denom1 += e1;
            denom2 += e2;  denom3 += e3;
        }
        data_t denom = (denom0 + denom1) + (denom2 + denom3);
        data_t inv_denom = (data_t)((data_t)1 / denom);

        // ===== 3) O = (1/denom) * (E . V) : 完全展開, 未正規化 E =====
        data_t acc[DK];
        #pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        Init_Acc: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            acc[d] = (data_t)0;
        }
        PV_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t e = E_row[j];
            MAC_D: for (int d = 0; d < DK; d++) {
                #pragma HLS UNROLL
                acc[d] += e * V[j][d];
            }
        }
        Write_O: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            O[i][d] = (data_t)(acc[d] * inv_denom);
        }
    }
}