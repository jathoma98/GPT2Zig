// Per-row layernorm of a [rows, cols] row-major matrix: (v - mean)/sqrt(var + eps)*gamma + beta.
// Mirrors src/core/op.zig `layernorm` — variance is BIASED (/cols), mean subtracted, beta added.
// One thread per row; flat `rows` global extent. Gamma/Beta are length `cols`.
StructuredBuffer<float> In;
StructuredBuffer<float> Gamma;
StructuredBuffer<float> Beta;
RWStructuredBuffer<float> Out;

struct Push {
    uint rows;
    uint cols;
    float eps;
};
[[vk::push_constant]] Push p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint r = tid.x;
    if (r >= p.rows) return;
    uint base = r * p.cols;
    float fcols = float(p.cols);

    float sum = 0.0;
    for (uint k = 0; k < p.cols; ++k) sum += In[base + k];
    float mean = sum / fcols;

    float sq = 0.0;
    for (uint k = 0; k < p.cols; ++k) {
        float d = In[base + k] - mean;
        sq += d * d;
    }
    float inv_std = 1.0 / sqrt(sq / fcols + p.eps);

    for (uint k = 0; k < p.cols; ++k) {
        Out[base + k] = (In[base + k] - mean) * inv_std * Gamma[k] + Beta[k];
    }
}
