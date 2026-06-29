// Token + position embedding: X[s, d] = Wte[ids[s], d] + Wpe[s, d].  Wte:[vocab,E], Wpe:[n_ctx,E].
// Mirrors src/core/model.zig `embed`. One thread per (s, d) output element; flat S*E global extent.
StructuredBuffer<float> Wte;
StructuredBuffer<float> Wpe;
StructuredBuffer<uint> Ids;
RWStructuredBuffer<float> X;

struct Push {
    uint S;
    uint E;
};
[[vk::push_constant]] Push p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= p.S * p.E) return;
    uint s = t / p.E;
    uint d = t % p.E;
    uint id = Ids[s];
    X[t] = Wte[id * p.E + d] + Wpe[s * p.E + d];
}
