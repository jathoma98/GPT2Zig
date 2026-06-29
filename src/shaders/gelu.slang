// GELU tanh approximation (gelu_new) — the flavor GPT-2 was trained with. Mirrors src/core/op.zig
// `gelu` constant-for-constant. Elementwise; flat `n` global extent.
StructuredBuffer<float> In;
RWStructuredBuffer<float> Out;

struct Push {
    uint n;
};
[[vk::push_constant]] Push p;

static const float GELU_C = 0.7978845608028654f; // sqrt(2/pi)
static const float GELU_A = 0.044715f;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= p.n) return;
    float v = In[i];
    float inner = GELU_C * (v + GELU_A * v * v * v);
    Out[i] = 0.5 * v * (1.0 + tanh(inner));
}
