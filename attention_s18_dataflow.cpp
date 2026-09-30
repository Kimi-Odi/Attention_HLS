// =============================================================
//  Solution #18 : DATAFLOW (函式級三階段串流, 修正版)
//
//  第一版錯誤: DATAFLOW 在 Row_Loop 內 + Write_O 在 region 內寫外部 O
//    -> 數百個 process 競爭寫 O -> Pre-synthesis failed。
//
//  修正: 把 DATAFLOW 提到「函式級」, 三個 task 各自處理「全部 N 列」,
//    用 hls::stream 串接 (跨所有列連續串流)。只有最後的 PV task 寫 O,
//    且正規化併入 PV (用從 Exp 傳來的 per-row denom)。
//
//  結構:
//    stage_qkt: 對所有列算 S, 串流輸出 (N*N 個值)
//    stage_exp: 算 exp + 每列的 denom, 串流輸出 E 和 denom
//    stage_pv:  用 E 算 PV, 拿 denom 正規化, 寫 O (唯一寫 O 的 task)
//
//  三個 task 在函式級 DATAFLOW 下「同時跑」, 跨列連續串流。
//  這樣不需 per-row fill/drain, 列與列之間也重疊。
//
//  基底邏輯 = S16 (factor=16, 延遲正規化, 省 max)。
// =============================================================
#include "attention.h"
#include <hls_stream.h>

#define QK_BANKS 16

// ---- 階段 1: QKt, 對所有列串流輸出 S ----
static void stage_qkt(
    data_t Q[N][DK],
    data_t K[N][DK],
    hls::stream<data_t> &s_out)
{
    Row_QKt: for (int i = 0; i < N; i++) {
        data_t Q_row[DK];
        #pragma HLS ARRAY_PARTITION variable=Q_row complete dim=1
        Load_Q: for (int k = 0; k < DK; k++) {
            #pragma HLS UNROLL
            Q_row[k] = Q[i][k];
        }
        QKt_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
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
            s_out.write((data_t)(group_sum * INV_SQRT_DK));
        }
    }
}

// ---- 階段 2: Exp (無 max) + 每列 denom, 串流輸出 E 和 denom ----
static void stage_exp(
    hls::stream<data_t> &s_in,
    hls::stream<data_t> &e_out,
    hls::stream<data_t> &denom_out)
{
    Row_Exp: for (int i = 0; i < N; i++) {
        data_t denom = (data_t)0;
        Exp_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t s = s_in.read();
            data_t e = hls::exp(s);
            e_out.write(e);
            denom += e;
        }
        denom_out.write(denom);   // 每列一個 denom
    }
}

// ---- 階段 3: PV (用 E) + 正規化, 寫 O (唯一寫 O 的 task) ----
static void stage_pv(
    hls::stream<data_t> &e_in,
    hls::stream<data_t> &denom_in,
    data_t V[N][DK],
    data_t O[N][DK])
{
    Row_PV: for (int i = 0; i < N; i++) {
        data_t acc[DK];
        #pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        Init_Acc: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            acc[d] = (data_t)0;
        }
        PV_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t e = e_in.read();
            MAC_D: for (int d = 0; d < DK; d++) {
                #pragma HLS UNROLL
                acc[d] += e * V[j][d];
            }
        }
        // 拿這列的 denom, 正規化寫 O
        data_t denom = denom_in.read();
        data_t inv_denom = (data_t)((data_t)1 / denom);
        Write_O: for (int d = 0; d < DK; d++) {
            #pragma HLS UNROLL
            O[i][d] = (data_t)(acc[d] * inv_denom);
        }
    }
}

void attention_head(
    data_t Q[N][DK],
    data_t K[N][DK],
    data_t V[N][DK],
    data_t O[N][DK])
{
    #pragma HLS ARRAY_PARTITION variable=Q complete dim=2
    #pragma HLS ARRAY_PARTITION variable=K cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=V complete dim=2

    // ---- 函式級 DATAFLOW: 三階段同時跑, 跨列連續串流 ----
    #pragma HLS DATAFLOW
    hls::stream<data_t> s_stream("s_stream");
    hls::stream<data_t> e_stream("e_stream");
    hls::stream<data_t> denom_stream("denom_stream");
    #pragma HLS STREAM variable=s_stream depth=64
    #pragma HLS STREAM variable=e_stream depth=64
    #pragma HLS STREAM variable=denom_stream depth=4

    stage_qkt(Q, K, s_stream);
    stage_exp(s_stream, e_stream, denom_stream);
    stage_pv(e_stream, denom_stream, V, O);
}