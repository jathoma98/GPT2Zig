// Causal scaled dot-product attention, pass 1 of 3: scores[h,i,j] = (q_i · k_j) * scale for j <= i.
// Mirrors the score loop in src/core/model.zig `applyLayer`. Reads the fused QKV buffer [S, 3*E]
// (Q at col h*head_dim, K at col E + h*head_dim). Scores is [n_head, S, S] row-major (h,i,j).
// One thread per (h,i,j); flat n_head*S*S global extent. Entries with j > i are left 0 (the causal
// upper triangle is never read by the softmax/weighted-sum passes, which bound their loops at j<=i).
StructuredBuffer<float> Qkv;
RWStructuredBuffer<float> Scores;

struct Push {
    uint S;
    uint n_head;
    uint head_dim;
    uint n_embd;
    float scale;
};
[[vk::push_constant]] Push p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= p.n_head * p.S * p.S) return;
    uint h = t / (p.S * p.S);
    uint rem = t % (p.S * p.S);
    uint i = rem / p.S;
    uint j = rem % p.S;

    if (j > i) {
        Scores[t] = 0.0;
        return;
    }

    uint stride = 3 * p.n_embd;
    uint qoff = i * stride + h * p.head_dim;
    uint koff = j * stride + p.n_embd + h * p.head_dim;
    float dot = 0.0;
    for (uint d = 0; d < p.head_dim; ++d) {
        dot += Qkv[qoff + d] * Qkv[koff + d];
    }
    Scores[t] = dot * p.scale;
}
