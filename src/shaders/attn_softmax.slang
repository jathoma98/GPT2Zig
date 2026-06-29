// Causal attention, pass 2 of 3: softmax over the valid keys j <= i for each (h, i), in place on
// the [n_head, S, S] Scores buffer. Row length is i+1 (causal). Mirrors op.zig `softmax` applied to
// the `scores[0..i+1]` slice in model.zig. One thread per (h, i); flat n_head*S global extent.
RWStructuredBuffer<float> Scores;

struct Push {
    uint S;
    uint n_head;
};
[[vk::push_constant]] Push p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= p.n_head * p.S) return;
    uint h = t / p.S;
    uint i = t % p.S;
    uint base = h * p.S * p.S + i * p.S;
    uint len = i + 1; // causal: keys 0..i

    float mx = Scores[base];
    for (uint j = 1; j < len; ++j) mx = max(mx, Scores[base + j]);

    float sum = 0.0;
    for (uint j = 0; j < len; ++j) {
        float e = exp(Scores[base + j] - mx);
        Scores[base + j] = e;
        sum += e;
    }

    float inv = 1.0 / sum;
    for (uint j = 0; j < len; ++j) Scores[base + j] *= inv;
}
