// y = x @ Wᵀ, no bias.  X:[M,K] row-major, W:[N,K] row-major (so Wᵀ is [K,N]), Y:[M,N].
// Mirrors src/core/op.zig `matmulBT` — the tied output projection (logits = x @ wteᵀ). One thread
// per output element; flat M*N global extent.
StructuredBuffer<float> X;
StructuredBuffer<float> W;
RWStructuredBuffer<float> Y;

struct Push {
    uint M;
    uint K;
    uint N;
};
[[vk::push_constant]] Push p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= p.M * p.N) return;
    uint row = t / p.N;
    uint col = t % p.N;
    float acc = 0.0;
    for (uint k = 0; k < p.K; ++k) {
        acc += X[row * p.K + k] * W[col * p.K + k];
    }
    Y[t] = acc;
}
