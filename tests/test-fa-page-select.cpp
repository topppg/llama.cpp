// CPU: flash attention with page select (ggml_flash_attn_ext_set_page_select) equals flash attention with the mask
// built by ggml_top_k_mask from the same selection (the block-sparse prefill with and without built masks)

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct fa_case {
    int64_t   hs;
    int64_t   nh;
    int64_t   gqa;
    int64_t   kv;
    int64_t   nt;
    int64_t   n_top;
    int64_t   W;
    ggml_type type_kv;
};

static void fill_uniform(ggml_tensor * t, std::mt19937 & gen) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> f(ggml_nelements(t));
    for (float & x : f) {
        x = u(gen);
    }
    if (t->type == GGML_TYPE_F32) {
        memcpy(t->data, f.data(), f.size()*sizeof(float));
    } else {
        ggml_quantize_chunk(t->type, f.data(), t->data, 0, ggml_nrows(t), t->ne[0], nullptr);
    }
}

static bool run(const fa_case & tc) {
    const int64_t n_page  = tc.kv/64;
    const int64_t n_extra = 64 + tc.W + 128;
    const int64_t c0      = tc.kv - tc.nt;

    ggml_init_params ip = { size_t(1) << 30, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q      = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.hs, tc.nt, tc.nh*tc.gqa, 1);
    ggml_tensor * k      = ggml_new_tensor_4d(ctx, tc.type_kv, tc.hs, tc.kv, tc.nh, 1);
    ggml_tensor * v      = ggml_new_tensor_4d(ctx, tc.type_kv, tc.hs, tc.kv, tc.nh, 1);
    ggml_tensor * m      = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.kv, tc.nt, 1, 1);
    ggml_tensor * picks  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, tc.n_top, tc.nt);
    ggml_tensor * blocks = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 64, n_page);
    ggml_tensor * extra  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_extra, tc.nt);

    std::mt19937 gen(42);
    fill_uniform(q, gen);
    fill_uniform(k, gen);
    fill_uniform(v, gen);

    ggml_fp16_t * mh = (ggml_fp16_t *) m->data;
    for (int64_t r = 0; r < tc.nt; ++r) {
        for (int64_t c = 0; c < tc.kv; ++c) {
            const float x = c > c0 + r || (c*7 + r) % 97 == 0 ? -INFINITY : -0.03125f * float(1 + (c + r) % 7);
            mh[r*tc.kv + c] = ggml_fp32_to_fp16(x);
        }
    }
    int32_t * pk = (int32_t *) picks->data;
    int32_t * ex = (int32_t *) extra->data;
    std::fill(ex, ex + n_extra*tc.nt, -1);
    for (int64_t b = 0; b < (tc.nt + 63)/64; ++b) {
        const int64_t ws  = c0 + 64*b - tc.W;
        const int64_t wsp = ws < 64 ? 1 : ws/64;
        std::vector<int32_t> pages;
        for (int64_t p = 1; p < wsp; ++p) {
            pages.push_back(int32_t(p));
        }
        std::shuffle(pages.begin(), pages.end(), gen);
        pages.resize(std::min<int64_t>(tc.n_top, pages.size()));
        while ((int64_t) pages.size() < tc.n_top) {
            pages.insert(pages.begin(), int32_t(n_page + pages.size()));
        }
        for (int64_t r = 64*b; r < std::min<int64_t>(64*b + 64, tc.nt); ++r) {
            std::copy(pages.begin(), pages.end(), pk + r*tc.n_top);
            int32_t * row = ex + r*n_extra;
            int64_t j = 0;
            for (int64_t c = 0; c < 64; ++c) {
                row[j++] = int32_t(c);
            }
            for (int64_t c = std::max<int64_t>(64, 64*wsp); c <= c0 + r && j < n_extra; ++c) {
                row[j++] = int32_t(c);
            }
        }
    }
    int32_t * blk = (int32_t *) blocks->data;
    for (int64_t c = 0; c < 64*n_page; ++c) {
        blk[c] = int32_t(c);
    }

    const float scale = 1.0f/sqrtf(float(tc.hs));
    ggml_tensor * a = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_page_select(a, picks, blocks, extra);
    ggml_prec_set_acc(a, GGML_PREC_F32);
    ggml_tensor * b = ggml_flash_attn_ext(ctx, q, k, v, ggml_top_k_mask(ctx, m, picks, nullptr, blocks, extra), scale, 0.0f, 0.0f);
    ggml_prec_set_acc(b, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, a);
    ggml_build_forward_expand(gf, b);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const float * fa = (const float *) a->data;
    const float * fb = (const float *) b->data;
    double max_err = 0.0;
    double max_ref = 0.0;
    for (int64_t i = 0; i < ggml_nelements(a); ++i) {
        max_err = std::max(max_err, (double) fabsf(fa[i] - fb[i]));
        max_ref = std::max(max_ref, (double) fabsf(fb[i]));
    }
    const bool ok = std::isfinite(max_err) && max_err <= 1e-4*std::max(1.0, max_ref);
    printf("hs=%lld nh=%lld gqa=%lld kv=%lld nt=%lld n_top=%lld W=%lld type=%s: max |page select - top_k_mask| = %.3g (max %.3g) %s\n",
        (long long) tc.hs, (long long) tc.nh, (long long) tc.gqa, (long long) tc.kv, (long long) tc.nt, (long long) tc.n_top,
        (long long) tc.W, ggml_type_name(tc.type_kv), max_err, max_ref, ok ? "OK" : "FAIL");

    ggml_free(ctx);
    return ok;
}

int main() {
    const fa_case cases[] = {
        { 128, 2, 4, 4096, 256, 6, 256, GGML_TYPE_F16  },
        { 128, 2, 4, 4096, 256, 6, 256, GGML_TYPE_Q8_0 },
        { 64,  1, 8, 4096, 196, 3, 256, GGML_TYPE_F16  },
        { 64,  1, 1, 2048, 1,   4, 256, GGML_TYPE_F16  },
    };
    bool ok = true;
    for (const fa_case & tc : cases) {
        ok &= run(tc);
    }
    return ok ? 0 : 1;
}
