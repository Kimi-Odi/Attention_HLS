// =============================================================
//  testbench_s20.cpp  --  S20 (定點數) 專用測試平台
//  介面 io_t = ap_fixed<12,1>, 黃金模型 float, 混合容差。
// =============================================================
#include <ap_fixed.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define N   1024
#define DK  128
typedef ap_fixed<12,1> io_t;   // 必須和 S20 的 io_t 一致

void attention_head(io_t Q[N][DK], io_t K[N][DK], io_t V[N][DK], io_t O[N][DK]);

static const float ABS_TOL = 5e-3f;
static const float REL_TOL = 5e-2f;

static void golden_attention(
    const std::vector<float>& Q, const std::vector<float>& K,
    const std::vector<float>& V, std::vector<float>& O)
{
    const float inv = 1.0f/std::sqrt((float)DK);
    std::vector<float> Sr(N);
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            float s=0; for(int k=0;k<DK;k++) s+=Q[i*DK+k]*K[j*DK+k];
            Sr[j]=s*inv;
        }
        float mx=Sr[0]; for(int j=1;j<N;j++) if(Sr[j]>mx)mx=Sr[j];
        float den=0; for(int j=0;j<N;j++){Sr[j]=std::exp(Sr[j]-mx);den+=Sr[j];}
        for(int j=0;j<N;j++) Sr[j]/=den;
        for(int d=0;d<DK;d++){float a=0;for(int j=0;j<N;j++)a+=Sr[j]*V[j*DK+d];O[i*DK+d]=a;}
    }
}

int main(){
    static io_t Q[N][DK], K[N][DK], V[N][DK], O[N][DK];
    std::vector<float> Qf(N*DK), Kf(N*DK), Vf(N*DK), Ogold(N*DK);
    srand(1234);
    for (int idx=0; idx<N*DK; idx++){
        float q=((rand()/(float)RAND_MAX)*2.0f-1.0f)*0.5f;
        float k=((rand()/(float)RAND_MAX)*2.0f-1.0f)*0.5f;
        float v=((rand()/(float)RAND_MAX)*2.0f-1.0f);
        Qf[idx]=q; Kf[idx]=k; Vf[idx]=v;
        Q[idx/DK][idx%DK]=(io_t)q; K[idx/DK][idx%DK]=(io_t)k; V[idx/DK][idx%DK]=(io_t)v;
    }
    golden_attention(Qf, Kf, Vf, Ogold);
    attention_head(Q, K, V, O);
    int errors=0; float max_abs=0; double sum_abs=0, sum_sq=0;
    for (int idx=0; idx<N*DK; idx++){
        float dut=(float)O[idx/DK][idx%DK]; float gold=Ogold[idx];
        float ae=std::fabs(dut-gold); float re=ae/(std::fabs(gold)+1e-9f);
        sum_abs+=ae; sum_sq+=(double)ae*ae; if(ae>max_abs)max_abs=ae;
        bool ok=(ae<=ABS_TOL)||(re<=REL_TOL);
        if(!ok){ if(errors<20) printf("MISMATCH idx=%d dut=%.5f gold=%.5f abs=%.5f\n",idx,dut,gold,ae); errors++; }
    }
    printf("\n==============================\n");
    printf("  [S20 fixed] max_abs=%.6f MAE=%.6f RMSE=%.6f\n", max_abs, sum_abs/(N*DK), std::sqrt(sum_sq/(N*DK)));
    printf("  mismatches = %d / %d\n", errors, N*DK);
    printf("  RESULT: %s\n", errors==0?"PASS":"FAIL");
    printf("==============================\n");
    return errors==0?0:1;
}