// y = x @ W + b.  X:[M,K] row-major, W:[K,N] row-major, B:[N], Y:[M,N].
// Mirrors src/core/op.zig `matmul` (naive ijk, acc seeded with bias). One thread per output
// element; the host dispatches a flat M*N global extent (groups = ceil(M*N / 64)).
StructuredBuffer<float> X;
StructuredBuffer<float> W;
StructuredBuffer<float> B;
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
    float acc = B[col];
    for (uint k = 0; k < p.K; ++k) {
        acc += X[row * p.K + k] * W[k * p.N + col];
    }
    Y[t] = acc;
}
