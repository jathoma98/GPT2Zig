// Per-row softmax of a [rows, cols] row-major matrix, with max-subtraction for stability.
// Mirrors src/core/op.zig `softmax`. One thread per row; flat `rows` global extent.
StructuredBuffer<float> In;
RWStructuredBuffer<float> Out;

struct Push {
    uint rows;
    uint cols;
};
[[vk::push_constant]] Push p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint r = tid.x;
    if (r >= p.rows) return;
    uint base = r * p.cols;

    float mx = In[base];
    for (uint k = 1; k < p.cols; ++k) mx = max(mx, In[base + k]);

    float sum = 0.0;
    for (uint k = 0; k < p.cols; ++k) {
        float e = exp(In[base + k] - mx);
        Out[base + k] = e;
        sum += e;
    }

    float inv = 1.0 / sum;
    for (uint k = 0; k < p.cols; ++k) Out[base + k] *= inv;
}
