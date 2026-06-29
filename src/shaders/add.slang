// In-place residual add: X[i] += Y[i].  Mirrors the `x[idx] += proj[idx]` / `x[idx] += mlp[idx]`
// residual connections in src/core/model.zig. Elementwise; flat `n` global extent.
RWStructuredBuffer<float> X;
StructuredBuffer<float> Y;

struct Push {
    uint n;
};
[[vk::push_constant]] Push p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= p.n) return;
    X[i] += Y[i];
}
