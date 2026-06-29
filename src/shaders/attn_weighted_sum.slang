// Causal attention, pass 3 of 3: out[i, h*head_dim + d] = Σ_{j<=i} p[h,i,j] * v_j[d].
// Mirrors the weighted-sum loop in src/core/model.zig `applyLayer`. Reads softmaxed Scores
// [n_head,S,S] and the fused QKV buffer [S,3*E] (V at col 2*E + h*head_dim). Writes Attn [S, E]
// (heads concatenated). One thread per (h, i, d); flat n_head*S*head_dim global extent.
StructuredBuffer<float> Qkv;
StructuredBuffer<float> Scores;
RWStructuredBuffer<float> Attn;

struct Push {
    uint S;
    uint n_head;
    uint head_dim;
    uint n_embd;
};
[[vk::push_constant]] Push p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= p.n_head * p.S * p.head_dim) return;
    uint h = t / (p.S * p.head_dim);
    uint rem = t % (p.S * p.head_dim);
    uint i = rem / p.head_dim;
    uint d = rem % p.head_dim;

    uint stride = 3 * p.n_embd;
    uint sbase = h * p.S * p.S + i * p.S;
    uint voff = 2 * p.n_embd + h * p.head_dim + d;

    float acc = 0.0;
    for (uint j = 0; j <= i; ++j) {
        acc += Scores[sbase + j] * Qkv[j * stride + voff];
    }
    Attn[i * p.n_embd + h * p.head_dim + d] = acc;
}
