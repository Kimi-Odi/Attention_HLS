// =============================================================
//  Solution #5c : QKt cyclic factor 掃描  (砍更兇: 16 / 8)
//
//  與 S5b 完全相同的架構, 唯一差別是 QK_BANKS 參數。
//  S5b 已驗證 factor=32 有效 (DSP 519->267, ATP 改善 29%)。
//  S5c 往更兇砍, 看 ATP 能否再降:
//     QK_BANKS=16 -> QKt 用 16 乘法器 (內積分 8 拍)
//     QK_BANKS=8  -> QKt 用 8  乘法器 (內積分 16 拍)
//
//  使用方式: 改下面這一行的數字, 重新合成。
//     16 -> 跑一次 (S5c_f16)
//     8  -> 改成 8 再跑一次 (S5c_f8)
//
//  注意: QK_BANKS 必須能整除 DK(=128)。合法值: 128,64,32,16,8,4,2,1
//        ARRAY_PARTITION 的 factor 數字也要跟著一起改 (HLS pragma
//        不吃巨集展開, 必須是字面常數)。
//        => 改 QK_BANKS 時, 同步改 K 那行 cyclic factor=XX 的數字!
// =============================================================
#include "attention.h"

#define QK_BANKS 8   // <<< 掃描點: 改成 16 或 8

void attention_head(
    data_t Q[N][DK],
    data_t K[N][DK],
    data_t V[N][DK],
    data_t O[N][DK])
{
    #pragma HLS ARRAY_PARTITION variable=Q complete dim=2
    // !! 下面這個 factor 數字必須與 QK_BANKS 一致 (HLS 不吃巨集) !!
    #pragma HLS ARRAY_PARTITION variable=K cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=V complete dim=2

    data_t S_row[N];

    Row_Loop: for (int i = 0; i < N; i++) {

        // ===== 1) S = Q.K^T : 每拍 QK_BANKS 個 MAC, 共 DK/QK_BANKS 拍 =====
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

        // ===== 3) O = P.V : 完全展開 d_k (同 S3/S5b) =====
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