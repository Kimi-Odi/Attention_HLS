// =============================================================
//  Solution #24 : S23 + 32-entry 表  (榨 ROM 成本)
//
//  想法: softmax 用 exp(x), 但 exp(x) = 2^(x * log2(e))。
//    softmax 對底數不敏感: 把 scale log2(e) 併入 1/sqrt(dk) 後,
//    用 2^S 算 (= exp), 數學完全等價。
//    2^y 在硬體便宜: 整數部分 = bit shift (ldexp), 小數部分查小表。
//
//  實作:
//    S = QK^T * (1/sqrt(dk) * log2(e))   <- scale 併入 log2e
//    E = 2^S = ldexp(2^frac, int)
//      int 部分: hls::ldexpf (指數運算, 便宜)
//      frac 部分: 查 256-entry 2^f 表 (f in [0,1))
//    比 hls::exp (CORDIC/多項式) 的運算便宜, 表也比 e^x 全範圍小。
//
//  S24: 表 256->32 (C 模擬: 32-entry max_abs 0.0019, 和 256 幾乎同)。
//  目的: ROM 成本降 -> ENS 降 -> ATP 再微幅改善。
//
//  基底 = S16 (數學簡化: 省 max + 延遲正規化), 全 FP16。
//    只把 exp 換成 base-2 (2^S)。
//  預期: exp 計算便宜 -> LUT/FF 可能省, clock 可能改善。
//    但 exp 佔每列僅 ~11%, 預期小改善。
// =============================================================
#include "attention.h"

#define QK_BANKS 16
#define POW2_TSIZE 32

static const float POW2_ROM[32] = {
    1.00000000f, 1.02189715f, 1.04427378f, 1.06714040f, 1.09050773f, 1.11438674f, 1.13878863f, 1.16372486f,
    1.18920712f, 1.21524736f, 1.24185781f, 1.26905096f, 1.29683955f, 1.32523664f, 1.35425555f, 1.38390988f,
    1.41421356f, 1.44518081f, 1.47682615f, 1.50916443f, 1.54221083f, 1.57598085f, 1.61049033f, 1.64575548f,
    1.68179283f, 1.71861930f, 1.75625216f, 1.79470908f, 1.83400809f, 1.87416763f, 1.91520656f, 1.95714412f,
};

// scale 已含 log2(e): 1/sqrt(128) * 1.44269504 = 0.08838835 * 1.44269504
static const data_t INV_SQRT_DK_LOG2E = (data_t)0.127500604f;  // = (1/sqrt128)*log2(e)

// 2^y 查表: y in (-inf, ~6], 回傳 data_t
//   2^y = 2^floor(y) * 2^frac, floor 用 ldexp (shift), frac 查表
static inline data_t pow2_lut(const data_t tab[POW2_TSIZE], data_t y) {
    #pragma HLS INLINE
    float yf = (float)y;
    if (yf < -24.0f) return (data_t)0;   // 太小, 趨近 0
    int i = (int)hls::floorf(yf);
    float f = yf - (float)i;              // [0,1)
    int idx = (int)(f * POW2_TSIZE);
    if (idx < 0) idx = 0;
    if (idx >= POW2_TSIZE) idx = POW2_TSIZE - 1;
    // 2^f (查表) * 2^i (ldexp = shift)
    float val = hls::ldexpf((float)tab[idx], i);
    return (data_t)val;
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

    // 2^f 常數表 ROM (切 4 bank 支援 u4)
    data_t pow2_tab[POW2_TSIZE];
    #pragma HLS ARRAY_PARTITION variable=pow2_tab cyclic factor=4 dim=1
    Init_ROM: for (int t = 0; t < POW2_TSIZE; t++) {
        #pragma HLS UNROLL factor=4
        pow2_tab[t] = (data_t)POW2_ROM[t];
    }

    data_t E_row[N];

    Row_Loop: for (int i = 0; i < N; i++) {

        // ===== 1) S = Q.K^T : factor=16 + 加法樹 (scale 含 log2e) =====
        QKt_Loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
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
            data_t group_sum = ((ps[0] + ps[1]) + (ps[2] + ps[3]))
                             + ((ps[4] + ps[5]) + (ps[6] + ps[7]));
            // scale 含 log2(e): 之後 2^S = exp(原始 S)
            E_row[j] = (data_t)(group_sum * INV_SQRT_DK_LOG2E);
        }

        // ===== 2) 2^S (base-2 exp, 無 max) + Sum : softmax u4 =====
        data_t denom0 = (data_t)0, denom1 = (data_t)0;
        data_t denom2 = (data_t)0, denom3 = (data_t)0;
        Exp_Loop: for (int j = 0; j < N; j += 4) {
            #pragma HLS PIPELINE II=1
            data_t e0 = pow2_lut(pow2_tab, E_row[j]);     // 2^S = exp!
            data_t e1 = pow2_lut(pow2_tab, E_row[j+1]);
            data_t e2 = pow2_lut(pow2_tab, E_row[j+2]);
            data_t e3 = pow2_lut(pow2_tab, E_row[j+3]);
            E_row[j]   = e0;  E_row[j+1] = e1;
            E_row[j+2] = e2;  E_row[j+3] = e3;
            denom0 += e0;  denom1 += e1;
            denom2 += e2;  denom3 += e3;
        }
        data_t denom = (denom0 + denom1) + (denom2 + denom3);
        data_t inv_denom = (data_t)((data_t)1 / denom);

        // ===== 3) O = (1/denom) * (E . V) : 完全展開 =====
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