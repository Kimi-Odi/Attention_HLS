// =============================================================
//  Solution #23 : S16 + base-2 softmax  (FP16 ??àË??, exp Á∞°Â??)
//
//  ?É≥Ê≥?: softmax ?î® exp(x), ‰Ω? exp(x) = 2^(x * log2(e))??
//    softmax Â∞çÂ?ïÊï∏‰∏çÊ?èÊ??: ??? scale log2(e) ‰ΩµÂÖ• 1/sqrt(dk) Âæ?,
//    ?î® 2^S ÁÆ? (= exp), ?ï∏Â≠∏Â?åÂÖ®Á≠âÂÉπ??
//    2^y ?ú®Á°¨È?î‰æøÂÆ?: ?ï¥?ï∏?É®??? = bit shift (ldexp), Â∞èÊï∏?É®??ÜÊü•Â∞èË°®??
//
//  ÂØ¶‰??:
//    S = QK^T * (1/sqrt(dk) * log2(e))   <- scale ‰ΩµÂÖ• log2e
//    E = 2^S = ldexp(2^frac, int)
//      int ?É®???: hls::ldexpf (??áÊï∏??ãÁ??, ‰æøÂ??)
//      frac ?É®???: ?ü• 256-entry 2^f Ë°? (f in [0,1))
//    ÊØ? hls::exp (CORDIC/Â§öÈ?ÖÂ??) ??ÑÈ?ãÁ?ó‰æøÂÆ?, Ë°®‰?üÊ?? e^x ?Ö®ÁØÑÂ?çÂ?è„??
//
//  C Ê®°Êì¨È©óË?? (4 seed): base-2 + shift + 256 Ë°?, max_abs 0.0022, PASS??
//
//  ?ü∫Â∫? = S16 (?ï∏Â≠∏Á∞°???: ??? max + Âª∂ÈÅ≤Ê≠?Ë¶èÂ??), ?Ö® FP16??
//    ?è™??? exp ??õÊ?? base-2 (2^S)??
//  ??êÊ??: exp Ë®àÁ?ó‰æøÂÆ? -> LUT/FF ?èØ?ÉΩ???, clock ?èØ?ÉΩ?îπ??Ñ„??
//    ‰Ω? exp ‰ΩîÊ?èÂ?óÂ?? ~11%, ??êÊ?üÂ?èÊîπ??Ñ„??
// =============================================================
#include "attention.h"

#define QK_BANKS 16
#define POW2_TSIZE 256

static const float POW2_ROM[256] = {
    1.00000000f, 1.00271128f, 1.00542990f, 1.00815590f, 1.01088929f, 1.01363008f, 1.01637831f, 1.01913400f,
    1.02189715f, 1.02466779f, 1.02744595f, 1.03023164f, 1.03302488f, 1.03582569f, 1.03863410f, 1.04145012f,
    1.04427378f, 1.04710510f, 1.04994409f, 1.05279077f, 1.05564518f, 1.05850732f, 1.06137723f, 1.06425491f,
    1.06714040f, 1.07003371f, 1.07293487f, 1.07584389f, 1.07876080f, 1.08168561f, 1.08461836f, 1.08755906f,
    1.09050773f, 1.09346440f, 1.09642908f, 1.09940180f, 1.10238258f, 1.10537145f, 1.10836841f, 1.11137350f,
    1.11438674f, 1.11740815f, 1.12043775f, 1.12347557f, 1.12652162f, 1.12957593f, 1.13263852f, 1.13570941f,
    1.13878863f, 1.14187620f, 1.14497214f, 1.14807648f, 1.15118923f, 1.15431042f, 1.15744007f, 1.16057821f,
    1.16372486f, 1.16688004f, 1.17004377f, 1.17321608f, 1.17639699f, 1.17958653f, 1.18278471f, 1.18599157f,
    1.18920712f, 1.19243138f, 1.19566439f, 1.19890617f, 1.20215673f, 1.20541611f, 1.20868432f, 1.21196140f,
    1.21524736f, 1.21854223f, 1.22184603f, 1.22515879f, 1.22848054f, 1.23181128f, 1.23515106f, 1.23849990f,
    1.24185781f, 1.24522483f, 1.24860098f, 1.25198628f, 1.25538076f, 1.25878444f, 1.26219735f, 1.26561951f,
    1.26905096f, 1.27249170f, 1.27594178f, 1.27940121f, 1.28287002f, 1.28634823f, 1.28983587f, 1.29333297f,
    1.29683955f, 1.30035564f, 1.30388127f, 1.30741645f, 1.31096121f, 1.31451559f, 1.31807960f, 1.32165328f,
    1.32523664f, 1.32882972f, 1.33243255f, 1.33604514f, 1.33966752f, 1.34329973f, 1.34694179f, 1.35059372f,
    1.35425555f, 1.35792731f, 1.36160902f, 1.36530072f, 1.36900242f, 1.37271417f, 1.37643597f, 1.38016787f,
    1.38390988f, 1.38766204f, 1.39142438f, 1.39519691f, 1.39897967f, 1.40277269f, 1.40657599f, 1.41038961f,
    1.41421356f, 1.41804788f, 1.42189260f, 1.42574774f, 1.42961334f, 1.43348941f, 1.43737600f, 1.44127312f,
    1.44518081f, 1.44909909f, 1.45302800f, 1.45696755f, 1.46091779f, 1.46487874f, 1.46885043f, 1.47283289f,
    1.47682615f, 1.48083023f, 1.48484517f, 1.48887099f, 1.49290773f, 1.49695541f, 1.50101407f, 1.50508373f,
    1.50916443f, 1.51325619f, 1.51735904f, 1.52147302f, 1.52559815f, 1.52973447f, 1.53388200f, 1.53804077f,
    1.54221083f, 1.54639218f, 1.55058488f, 1.55478894f, 1.55900440f, 1.56323129f, 1.56746964f, 1.57171948f,
    1.57598085f, 1.58025376f, 1.58453827f, 1.58883438f, 1.59314215f, 1.59746160f, 1.60179276f, 1.60613566f,
    1.61049033f, 1.61485681f, 1.61923514f, 1.62362533f, 1.62802742f, 1.63244145f, 1.63686745f, 1.64130545f,
    1.64575548f, 1.65021757f, 1.65469177f, 1.65917809f, 1.66367658f, 1.66818727f, 1.67271018f, 1.67724536f,
    1.68179283f, 1.68635263f, 1.69092480f, 1.69550936f, 1.70010635f, 1.70471581f, 1.70933776f, 1.71397225f,
    1.71861930f, 1.72327895f, 1.72795123f, 1.73263618f, 1.73733384f, 1.74204423f, 1.74676739f, 1.75150335f,
    1.75625216f, 1.76101384f, 1.76578844f, 1.77057597f, 1.77537649f, 1.78019003f, 1.78501661f, 1.78985628f,
    1.79470908f, 1.79957502f, 1.80445417f, 1.80934654f, 1.81425218f, 1.81917111f, 1.82410339f, 1.82904903f,
    1.83400809f, 1.83898059f, 1.84396657f, 1.84896607f, 1.85397913f, 1.85900577f, 1.86404605f, 1.86909999f,
    1.87416763f, 1.87924902f, 1.88434418f, 1.88945315f, 1.89457598f, 1.89971270f, 1.90486334f, 1.91002795f,
    1.91520656f, 1.92039921f, 1.92560594f, 1.93082679f, 1.93606179f, 1.94131099f, 1.94657442f, 1.95185212f,
    1.95714412f, 1.96245048f, 1.96777122f, 1.97310639f, 1.97845603f, 1.98382016f, 1.98919885f, 1.99459211f,
};

// scale Â∑≤Âê´ log2(e): 1/sqrt(128) * 1.44269504 = 0.08838835 * 1.44269504
static const data_t INV_SQRT_DK_LOG2E = (data_t)0.127500604f;  // = (1/sqrt128)*log2(e)

// 2^y ?ü•Ë°?: y in (-inf, ~6], ??ûÂÇ≥ data_t
//   2^y = 2^floor(y) * 2^frac, floor ?î® ldexp (shift), frac ?ü•Ë°?
static inline data_t pow2_lut(const data_t tab[POW2_TSIZE], data_t y) {
    #pragma HLS INLINE
    float yf = (float)y;
    if (yf < -24.0f) return (data_t)0;   // Â§™Â??, Ë∂®Ë?? 0
    int i = (int)hls::floorf(yf);
    float f = yf - (float)i;              // [0,1)
    int idx = (int)(f * POW2_TSIZE);
    if (idx < 0) idx = 0;
    if (idx >= POW2_TSIZE) idx = POW2_TSIZE - 1;
    // 2^f (?ü•Ë°?) * 2^i (ldexp = shift)
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

    // 2^f Â∏∏Êï∏Ë°? ROM (??? 4 bank ?îØ?è¥ u4)
    data_t pow2_tab[POW2_TSIZE];
    #pragma HLS ARRAY_PARTITION variable=pow2_tab cyclic factor=4 dim=1
    Init_ROM: for (int t = 0; t < POW2_TSIZE; t++) {
        #pragma HLS UNROLL factor=4
        pow2_tab[t] = (data_t)POW2_ROM[t];
    }

    data_t E_row[N];

    Row_Loop: for (int i = 0; i < N; i++) {

        // ===== 1) S = Q.K^T : factor=16 + ??†Ê?ïÊ®π (scale ?ê´ log2e) =====
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
            // scale ?ê´ log2(e): ‰πãÂ?? 2^S = exp(??üÂ?? S)
            E_row[j] = (data_t)(group_sum * INV_SQRT_DK_LOG2E);
        }

        // ===== 2) 2^S (base-2 exp, ?Ñ° max) + Sum : softmax u4 =====
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

        // ===== 3) O = (1/denom) * (E . V) : ÂÆåÂÖ®Â±ïÈ?? =====
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