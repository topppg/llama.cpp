#include "page-top-k.cuh"

// page selection of page-sparse decode in two kernels:
//   score:  one warp per page, the scores of all the heads of its KV head, up to PAGE_TOP_K_GQA_CHUNK heads per pass
//   select: one block per KV head, log-softmax per head, max over the group, radix select of the top pages

#define PAGE_TOP_K_GQA_CHUNK    16
#define PAGE_TOP_K_MAX_D        512
#define PAGE_TOP_K_SMEM_FLOATS  10240 // lse and group scores in shared memory up to this many floats, else in global memory
#define PAGE_TOP_K_SCORE_WARPS  8
#define PAGE_TOP_K_SCORE_PAGES  32
#define PAGE_TOP_K_SELECT_SIZE  1024
#define PAGE_TOP_K_LSE_PAD(n)   (((n) + 3) & ~3)

template <typename TS>
static __device__ __forceinline__ float2 page_top_k_load2(const TS * x);

template <>
__device__ __forceinline__ float2 page_top_k_load2<half>(const half * x) {
    return __half22float2(*(const half2 *) x);
}

template <>
__device__ __forceinline__ float2 page_top_k_load2<float>(const float * x) {
    return *(const float2 *) x;
}

// sc[k, g, p] = sum_j max(qs[j], 0)*max[j, p] + min(qs[j], 0)*min[j, p] + pmask[p], qs = scale*q of head k*n_gqa + g
template <typename TS>
static __global__ void page_top_k_score_kernel(
        const float * q, const TS * stats, const float * pmask, float * sc,
        const int d, const int n_gqa, const int n_page,
        const int64_t sq1, const int64_t ss1, const int64_t ss2, const float scale) {
    ggml_cuda_pdl_lc();

    extern __shared__ float qs[]; // [PAGE_TOP_K_GQA_CHUNK, d]

    const int k    = blockIdx.y;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int p1   = min(n_page, (int) (blockIdx.x + 1)*PAGE_TOP_K_SCORE_PAGES);

    ggml_cuda_pdl_sync();

    for (int g0 = 0; g0 < n_gqa; g0 += PAGE_TOP_K_GQA_CHUNK) {
        const int ng = min(PAGE_TOP_K_GQA_CHUNK, n_gqa - g0);

        __syncthreads();
        for (int i = threadIdx.x; i < ng*d; i += blockDim.x) {
            const int g = i / d;
            const int j = i % d;
            qs[i] = scale*q[(int64_t) (k*n_gqa + g0 + g)*sq1 + j];
        }
        __syncthreads();

        for (int p = blockIdx.x*PAGE_TOP_K_SCORE_PAGES + warp; p < p1; p += PAGE_TOP_K_SCORE_WARPS) {
            const TS * st = stats + p*ss1 + k*ss2;

            float acc[PAGE_TOP_K_GQA_CHUNK];
#pragma unroll
            for (int g = 0; g < PAGE_TOP_K_GQA_CHUNK; ++g) {
                acc[g] = 0.0f;
            }

            for (int j = 2*lane; j < d; j += 2*WARP_SIZE) {
                const float2 mx = page_top_k_load2(st + j);
                const float2 mn = page_top_k_load2(st + d + j);
#pragma unroll
                for (int g = 0; g < PAGE_TOP_K_GQA_CHUNK; ++g) {
                    if (g < ng) {
                        const float a = qs[g*d + j];
                        const float b = qs[g*d + j + 1];
                        acc[g] += (a > 0.0f ? a*mx.x : a*mn.x) + (b > 0.0f ? b*mx.y : b*mn.y);
                    }
                }
            }

            const float pm = pmask[p];
#pragma unroll
            for (int g = 0; g < PAGE_TOP_K_GQA_CHUNK; ++g) {
                if (g < ng) {
                    const float s = warp_reduce_sum(acc[g]);
                    if (lane == 0) {
                        sc[((int64_t) k*n_gqa + g0 + g)*n_page + p] = s + pm;
                    }
                }
            }
        }
    }
}

// larger float -> larger key
static __device__ __forceinline__ uint32_t page_top_k_key(const float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// exclusive scan over the block of one value per thread, returns the total in *total
static __device__ __forceinline__ int page_top_k_block_scan(int x, int * red, int * total) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int nw   = blockDim.x / WARP_SIZE;

    int incl = x;
#pragma unroll
    for (int o = 1; o < WARP_SIZE; o *= 2) {
        const int y = __shfl_up_sync(0xFFFFFFFF, incl, o, WARP_SIZE);
        if (lane >= o) {
            incl += y;
        }
    }
    if (lane == WARP_SIZE - 1) {
        red[warp] = incl;
    }
    __syncthreads();
    if (warp == 0) {
        int w = lane < nw ? red[lane] : 0;
#pragma unroll
        for (int o = 1; o < WARP_SIZE; o *= 2) {
            const int y = __shfl_up_sync(0xFFFFFFFF, w, o, WARP_SIZE);
            if (lane >= o) {
                w += y;
            }
        }
        if (lane < nw) {
            red[lane] = w;
        }
    }
    __syncthreads();
    const int res = incl - x + (warp > 0 ? red[warp - 1] : 0);
    *total = red[nw - 1];
    __syncthreads();
    return res;
}

// gs_global: [n_kv_h, n_page] scratch when the group scores do not fit in shared memory, else NULL
static __global__ void __launch_bounds__(PAGE_TOP_K_SELECT_SIZE) page_top_k_select_kernel(
        const float * sc, float * gs_global, int32_t * dst, const int n_gqa, const int n_page, const int n_top, const int64_t sd1) {
    ggml_cuda_pdl_lc();

    extern __shared__ float smem[]; // lse [n_gqa padded to 4], then the group scores [n_page] unless gs_global
    __shared__ float red_f[PAGE_TOP_K_SELECT_SIZE/WARP_SIZE][PAGE_TOP_K_GQA_CHUNK];
    __shared__ int   red_i[PAGE_TOP_K_SELECT_SIZE/WARP_SIZE];
    __shared__ int   hist[256];
    __shared__ uint32_t sel_prefix;
    __shared__ int      sel_rem;

    const int k    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int nt   = blockDim.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int nw   = nt / WARP_SIZE;

    float * lse = smem;
    // the padding keeps the vector loads of lse off the group scores, which other threads write meanwhile
    float * gs  = gs_global ? gs_global + (int64_t) k*n_page : smem + PAGE_TOP_K_LSE_PAD(n_gqa);

    const float * sk = sc + (int64_t) k*n_gqa*n_page;

    ggml_cuda_pdl_sync();

    // per-head max, then per-head log of the sum of exp, PAGE_TOP_K_GQA_CHUNK heads at a time
    for (int g0 = 0; g0 < n_gqa; g0 += PAGE_TOP_K_GQA_CHUNK) {
        const int ng = min(PAGE_TOP_K_GQA_CHUNK, n_gqa - g0);
#pragma unroll
        for (int g = 0; g < PAGE_TOP_K_GQA_CHUNK; ++g) {
            if (g < ng) {
                float m = -INFINITY;
                for (int p = tid; p < n_page; p += nt) {
                    m = fmaxf(m, sk[(g0 + g)*n_page + p]);
                }
                m = warp_reduce_max(m);
                if (lane == 0) {
                    red_f[warp][g] = m;
                }
            }
        }
        __syncthreads();
        if (tid < ng) {
            float x = -INFINITY;
            for (int w = 0; w < nw; ++w) {
                x = fmaxf(x, red_f[w][tid]);
            }
            lse[g0 + tid] = x;
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < PAGE_TOP_K_GQA_CHUNK; ++g) {
            if (g < ng) {
                const float mg = lse[g0 + g];
                float s = 0.0f;
                if (mg > -INFINITY) {
                    for (int p = tid; p < n_page; p += nt) {
                        s += expf(sk[(g0 + g)*n_page + p] - mg);
                    }
                }
                s = warp_reduce_sum(s);
                if (lane == 0) {
                    red_f[warp][g] = s;
                }
            }
        }
        __syncthreads();
        if (tid < ng) {
            float s = 0.0f;
            for (int w = 0; w < nw; ++w) {
                s += red_f[w][tid];
            }
            const float mg = lse[g0 + tid];
            lse[g0 + tid] = mg > -INFINITY ? mg + logf(s) : 0.0f;
        }
        __syncthreads();
    }

    // group score
    for (int p = tid; p < n_page; p += nt) {
        float x = -INFINITY;
        for (int g = 0; g < n_gqa; ++g) {
            x = fmaxf(x, sk[g*n_page + p] - lse[g]);
        }
        gs[p] = x;
    }
    if (tid == 0) {
        sel_prefix = 0;
        sel_rem    = n_top;
    }
    __syncthreads();

    // radix select of the key of the n_top-th largest group score, 8 bits per pass from the top
    uint32_t hmask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += nt) {
            hist[i] = 0;
        }
        __syncthreads();
        const uint32_t prefix = sel_prefix;
        for (int p = tid; p < n_page; p += nt) {
            const uint32_t key = page_top_k_key(gs[p]);
            if ((key & hmask) == prefix) {
                atomicAdd(&hist[(key >> shift) & 0xFF], 1);
            }
        }
        __syncthreads();
        if (warp == 0) {
            // lane l holds bins 255 - 8l ... 248 - 8l, counted from the top
            const int rem = sel_rem;
            int c[8];
            int lsum = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                c[i] = hist[255 - 8*lane - i];
                lsum += c[i];
            }
            int incl = lsum;
#pragma unroll
            for (int o = 1; o < WARP_SIZE; o *= 2) {
                const int y = __shfl_up_sync(0xFFFFFFFF, incl, o, WARP_SIZE);
                if (lane >= o) {
                    incl += y;
                }
            }
            const int excl = incl - lsum;
            __syncwarp();
            if (excl < rem && incl >= rem) {
                int cum = excl;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (cum + c[i] >= rem) {
                        sel_prefix = prefix | ((uint32_t) (255 - 8*lane - i) << shift);
                        sel_rem    = rem - cum;
                        break;
                    }
                    cum += c[i];
                }
            }
        }
        hmask |= 0xFFu << shift;
        __syncthreads();
    }

    // keys above the threshold, then the first sel_rem keys equal to it by page; contiguous pages per thread keep the page order
    const uint32_t thr = sel_prefix;
    const int      rem = sel_rem;
    const int      per = (n_page + nt - 1)/nt;
    const int      p0  = min(n_page, tid*per);
    const int      p1  = min(n_page, p0 + per);

    int n_eq = 0;
    for (int p = p0; p < p1; ++p) {
        n_eq += page_top_k_key(gs[p]) == thr;
    }
    int tot;
    int eq_rank = page_top_k_block_scan(n_eq, red_i, &tot);

    int n_in = 0;
    {
        int r = eq_rank;
        for (int p = p0; p < p1; ++p) {
            const uint32_t key = page_top_k_key(gs[p]);
            if (key > thr || (key == thr && r++ < rem)) {
                n_in++;
            }
        }
    }
    int pos = page_top_k_block_scan(n_in, red_i, &tot);

    for (int p = p0; p < p1; ++p) {
        const uint32_t key = page_top_k_key(gs[p]);
        if (key > thr || (key == thr && eq_rank++ < rem)) {
            for (int g = 0; g < n_gqa; ++g) {
                dst[(int64_t) (k*n_gqa + g)*sd1 + pos] = p;
            }
            pos++;
        }
    }
}

bool ggml_cuda_page_top_k_supported(const ggml_tensor * op) {
    const ggml_tensor * q     = op->src[0];
    const ggml_tensor * stats = op->src[1];
    const int64_t d  = q->ne[0];
    const size_t  ts = ggml_type_size(stats->type);
    return d % 2 == 0 && d <= PAGE_TOP_K_MAX_D && q->ne[1]/stats->ne[2] <= 4096 && stats->view_offs % (2*ts) == 0 &&
        q->nb[1] % sizeof(float) == 0 && stats->nb[1] % (2*ts) == 0 && stats->nb[2] % (2*ts) == 0 &&
        op->nb[1] % sizeof(int32_t) == 0;
}

template <typename TS>
static void page_top_k_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * stats = dst->src[1];
    const ggml_tensor * pmask = dst->src[2];

    const float scale = ggml_get_op_params_f32(dst, 0);

    const int d      = q->ne[0];
    const int n_head = q->ne[1];
    const int n_page = stats->ne[1];
    const int n_kv_h = stats->ne[2];
    const int n_gqa  = n_head/n_kv_h;
    const int n_top  = dst->ne[0];

    const bool gs_smem = PAGE_TOP_K_LSE_PAD(n_gqa) + n_page <= PAGE_TOP_K_SMEM_FLOATS;

    ggml_cuda_pool_alloc<float> sc(ctx.pool(), (size_t) n_kv_h*n_gqa*n_page);
    ggml_cuda_pool_alloc<float> gs(ctx.pool(), gs_smem ? 1 : (size_t) n_kv_h*n_page);

    const dim3 score_grid((n_page + PAGE_TOP_K_SCORE_PAGES - 1)/PAGE_TOP_K_SCORE_PAGES, n_kv_h, 1);
    const ggml_cuda_kernel_launch_params score_params(score_grid, dim3(PAGE_TOP_K_SCORE_WARPS*WARP_SIZE, 1, 1),
            (size_t) std::min(n_gqa, PAGE_TOP_K_GQA_CHUNK)*d*sizeof(float), ctx.stream());
    ggml_cuda_kernel_launch(page_top_k_score_kernel<TS>, score_params,
            (const float *) q->data, (const TS *) stats->data, (const float *) pmask->data, sc.get(),
            d, n_gqa, n_page, (int64_t) (q->nb[1]/sizeof(float)), (int64_t) (stats->nb[1]/sizeof(TS)), (int64_t) (stats->nb[2]/sizeof(TS)),
            scale);

    const ggml_cuda_kernel_launch_params select_params(dim3(n_kv_h, 1, 1), dim3(PAGE_TOP_K_SELECT_SIZE, 1, 1),
            (size_t) (PAGE_TOP_K_LSE_PAD(n_gqa) + (gs_smem ? n_page : 0))*sizeof(float), ctx.stream());
    ggml_cuda_kernel_launch(page_top_k_select_kernel, select_params,
            (const float *) sc.get(), gs_smem ? nullptr : gs.get(), (int32_t *) dst->data, n_gqa, n_page, n_top,
            (int64_t) (dst->nb[1]/sizeof(int32_t)));
}

void ggml_cuda_op_page_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[1]->type) {
        case GGML_TYPE_F16:
            page_top_k_cuda<half>(ctx, dst);
            break;
        case GGML_TYPE_F32:
            page_top_k_cuda<float>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
