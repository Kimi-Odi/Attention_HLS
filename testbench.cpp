// =============================================================
//  testbench.cpp  --  共用測試平台 (所有 solution 共用)
//
//  做的事:
//   1. 用固定 seed 產生隨機 Q / K / V (FP16)
//   2. 在 host 端用 float 算「黃金答案」O_golden
//   3. 呼叫硬體函式 attention_head 算出 O_dut
//   4. 逐元素比對, 相對誤差 > TOL 就算 FAIL
//
//  在 Vitis HLS:
//   - 把這個檔加入 Testbench 區 (不要加 synthesis flow)
//   - 把某個 attention_sX_*.cpp 加入 Source 區
//   - Run C Simulation 驗證正確性
//   - Run C Synthesis 取得 LUT/FF/BRAM/DSP/Latency
// =============================================================
#include "attention.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// ---- 混合容差 (FP16 attention 的正確驗證方式) ----
// 通過條件: 絕對誤差 <= ABS_TOL  「或」  相對誤差 <= REL_TOL
//   - 對接近 0 的小值 (softmax 後 O 多落在 0.0x 量級) 用絕對容差判斷
//   - 對較大的值用相對容差判斷
// FP16 只有 ~10 bit 尾數, 又經過 128 與 1024 次累加, 單純用相對誤差
// 會在接近 0 的元素上爆炸 (分母太小), 那不代表邏輯錯誤。
static const float ABS_TOL = 5e-3f;   // 絕對容差 0.005
static const float REL_TOL = 5e-2f;   // 相對容差 5%

// ---------- host 端 float 黃金模型 ----------
static void golden_attention(
    const std::vector<float>& Q,   // N*DK
    const std::vector<float>& K,   // N*DK
    const std::vector<float>& V,   // N*DK
    std::vector<float>&       O)   // N*DK
{
    const float inv_sqrt_dk = 1.0f / std::sqrt((float)DK);
    std::vector<float> Srow(N);

    for (int i = 0; i < N; i++) {
        // 1) S = Q.K^T * scale
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < DK; k++)
                s += Q[i*DK + k] * K[j*DK + k];
            Srow[j] = s * inv_sqrt_dk;
        }
        // 2) softmax (max-subtraction)
        float mx = Srow[0];
        for (int j = 1; j < N; j++) if (Srow[j] > mx) mx = Srow[j];
        float denom = 0.0f;
        for (int j = 0; j < N; j++) { Srow[j] = std::exp(Srow[j]-mx); denom += Srow[j]; }
        for (int j = 0; j < N; j++) Srow[j] /= denom;
        // 3) O = P.V
        for (int d = 0; d < DK; d++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++)
                acc += Srow[j] * V[j*DK + d];
            O[i*DK + d] = acc;
        }
    }
}

int main() {
    // ---- 配置記憶體 ----
    // 注意: N*DK = 1024*128 = 131072 個 half. 三個輸入 + 一個輸出.
    // 這在模擬時佔記憶體, 若機器吃緊可把 N 改小做功能驗證 (例如 N=64),
    // 但合成時務必改回 N=1024 取得正確資源/延遲數字。
    static data_t Q [N][DK];
    static data_t K [N][DK];
    static data_t V [N][DK];
    static data_t O [N][DK];

    std::vector<float> Qf(N*DK), Kf(N*DK), Vf(N*DK), Ogold(N*DK);

    // ---- 產生輸入 (固定 seed, 可重現) ----
    srand(1234);
    for (int idx = 0; idx < N*DK; idx++) {
        // 範圍 [-1, 1), 縮小一點避免 exp 溢位
        float q = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        float k = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        float v = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        Qf[idx] = q * 0.5f;
        Kf[idx] = k * 0.5f;
        Vf[idx] = v;
        Q[idx/DK][idx%DK] = (data_t)Qf[idx];
        K[idx/DK][idx%DK] = (data_t)Kf[idx];
        V[idx/DK][idx%DK] = (data_t)Vf[idx];
    }

    // ---- 黃金答案 ----
    golden_attention(Qf, Kf, Vf, Ogold);

    // ---- 待測硬體 ----
    attention_head(Q, K, V, O);

    // ---- 比對 (混合容差 + 統計指標) ----
    int   errors  = 0;
    float max_abs = 0.0f;     // 最大絕對誤差
    float max_rel = 0.0f;     // 最大相對誤差 (僅供參考)
    double sum_abs = 0.0;     // 絕對誤差總和 -> 平均
    double sum_sq  = 0.0;     // 平方誤差總和 -> RMSE
    for (int idx = 0; idx < N*DK; idx++) {
        float dut  = (float)O[idx/DK][idx%DK];
        float gold = Ogold[idx];
        float abs_err = std::fabs(dut - gold);
        float rel_err = abs_err / (std::fabs(gold) + 1e-9f);

        sum_abs += abs_err;
        sum_sq  += (double)abs_err * abs_err;
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel_err > max_rel) max_rel = rel_err;

        // 混合判準: 絕對 OR 相對, 任一通過即可
        bool ok = (abs_err <= ABS_TOL) || (rel_err <= REL_TOL);
        if (!ok) {
            if (errors < 20)
                printf("MISMATCH idx=%d  dut=%.5f  gold=%.5f  abs=%.5f  rel=%.4f\n",
                       idx, dut, gold, abs_err, rel_err);
            errors++;
        }
    }

    float mae  = (float)(sum_abs / (N*DK));
    float rmse = (float)std::sqrt(sum_sq / (N*DK));

    printf("\n==============================\n");
    printf("  max absolute error = %.6f\n", max_abs);
    printf("  mean absolute error (MAE) = %.6f\n", mae);
    printf("  RMSE = %.6f\n", rmse);
    printf("  max relative error = %.2f%% (僅參考, 接近 0 處會偏大)\n", max_rel*100.0f);
    printf("  tolerance: abs<=%.4f OR rel<=%.1f%%\n", ABS_TOL, REL_TOL*100.0f);
    printf("  mismatches = %d / %d\n", errors, N*DK);
    if (errors == 0) {
        printf("  RESULT: PASS\n");
    } else {
        printf("  RESULT: FAIL\n");
    }
    printf("==============================\n");

    return (errors == 0) ? 0 : 1;
}