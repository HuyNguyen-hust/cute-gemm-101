#include <cuda_fp16.h>
#include <cute/tensor.hpp>

#include "cuda_gemm.hpp"

// kernel
template <
    class T,
    class ProblemShape, class CtaTiler,
    class AStride, class ASmemLayout, class AThreadLayout,
    class BStride, class BSmemLayout, class BThreadLayout,
    class CStride, class CsmemLayout, class CThreadLayout
>
__global__ 
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void official_cute_gemm_v01(
    ProblemShape shape_mnk, CtaTiler cta_tiler,
    const T *A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA_layout,
    const T *B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB_layout,
    T *C, CStride dC, CsmemLayout , CThreadLayout tC_layout,
    T alpha, T beta
)
{
    using namespace cute;
    
    // assert
    CUTE_STATIC_ASSERT_V(rank(shape_mnk) == _3{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == _3{});

    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_mnk), dA));
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_mnk), dB));
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_mnk), dC));

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CsmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<0>(CsmemLayout{}) == size<0>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(CsmemLayout{}) == size<1>(cta_tiler));

    static_assert(is_static<AThreadLayout>::value);
    static_assert(is_static<BThreadLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) % size<0>(AThreadLayout{}) == _0{});
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) % size<1>(AThreadLayout{}) == _0{});
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) % size<0>(BThreadLayout{}) == _0{});
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) % size<1>(BThreadLayout{}) == _0{});

    static_assert(size(tA_layout) == size(tB_layout)); // num threads

    static_assert(is_static<CThreadLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(CsmemLayout{}) % size<0>(CThreadLayout{}) == _0{});
    CUTE_STATIC_ASSERT_V(size<1>(CsmemLayout{}) % size<1>(CThreadLayout{}) == _0{});

    static_assert(size(tC_layout) == size(tA_layout));

    // full tensor
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_mnk), dA); // m x k
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_mnk), dB); // n x k
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_mnk), dC); // m x n

    // global tile tensor
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
    // Tensor gA = local_tile(mA, select<0,2>(shape_mnk), select<0,2>(cta_coord)); // BLOCK_TILE_SIZE_M x BLOCK_TILE_SIZE_K x num_tiles_k
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // BLOCK_TILE_SIZE_M x BLOCK_TILE_SIZE_K x num_tiles_k
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // BLOCK_TILE_SIZE_N x BLOCK_TILE_SIZE_K x num_tiles_k
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // BLOCK_TILE_SIZE_M x BLOCK_TILE_SIZE_N

    // shared memory
    __shared__ T Asmem[cosize_v<ASmemLayout>];
    __shared__ T Bsmem[cosize_v<BSmemLayout>];

    // smem tile tensor
    Tensor sA = make_tensor(make_smem_ptr(Asmem), sA_layout); // BLOCK_TILE_SIZE_M x BLOCK_TILE_SIZE_K
    Tensor sB = make_tensor(make_smem_ptr(Bsmem), sB_layout); // BLOCK_TILE_SIZE_N x BLOCK_TILE_SIZE_K

    // use thread layout tA_layout, tB_layout to partition sA, sB, gA, gB. This is for copy
    // THREAD_TILE_SIZE_M = _32{}
    // THREAD_TILE_SIZE_N = _32{}
    // THREAD_TILE_SIZE_K = _8{}
    Tensor tAsA = local_partition(sA, tA_layout, threadIdx.x); // (BLOCK_TILE_SIZE_M / THREAD_TILE_SIZE_M) x (BLOCK_TILE_SIZE_K / THREAD_TILE_SIZE_K)
    Tensor tAgA = local_partition(gA, tA_layout, threadIdx.x); // (BLOCK_TILE_SIZE_M / THREAD_TILE_SIZE_M) x (BLOCK_TILE_SIZE_K / THREAD_TILE_SIZE_K) x num_tiles_k
    Tensor tBsB = local_partition(sB, tB_layout, threadIdx.x); // (BLOCK_TILE_SIZE_N / THREAD_TILE_SIZE_N) x (BLOCK_TILE_SIZE_K / THREAD_TILE_SIZE_K)
    Tensor tBgB = local_partition(gB, tB_layout, threadIdx.x); // (BLOCK_TILE_SIZE_N / THREAD_TILE_SIZE_N) x (BLOCK_TILE_SIZE_K / THREAD_TILE_SIZE_K) x num_tiles_k

    CUTE_STATIC_ASSERT_V(size<0>(tAsA) == size<0>(tAgA));
    CUTE_STATIC_ASSERT_V(size<0>(tBsB) == size<0>(tBgB));
    CUTE_STATIC_ASSERT_V(size<1>(tAsA) == size<1>(tAgA));
    CUTE_STATIC_ASSERT_V(size<1>(tBsB) == size<1>(tBgB));

    // use thread layout tC_layout to partition sA, sB, gC. This is for compute
    // We have different THREAD_TILE_SIZE_M and THREAD_TILE_SIZE_N for copy purpose and compute purpose
    // THREAD_TILE_SIZE_M = _16{}
    // THREAD_TILE_SIZE_N = _16{}
    Tensor tCsA = local_partition(sA, tC_layout, threadIdx.x, Step<_1, X>{}); // (BLOCK_TILE_SIZE_M / THREAD_TILE_SIZE_M) x BLOCK_TILE_SIZE_K
    Tensor tCsB = local_partition(sB, tC_layout, threadIdx.x, Step<X, _1>{}); // (BLOCK_TILE_SIZE_N / THREAD_TILE_SIZE_N) x BLOCK_TILE_SIZE_K
    Tensor tCgC = local_partition(gC, tC_layout, threadIdx.x, Step<_1, _1>{}); // (BLOCK_TILE_SIZE_M / THREAD_TILE_SIZE_M) x (BLOCK_TILE_SIZE_N / THREAD_TILE_SIZE_N)

    // accumulator
    Tensor tCrC = make_tensor_like(tCgC);

    CUTE_STATIC_ASSERT_V(size<0>(tCsA) == size<0>(tCgC));
    CUTE_STATIC_ASSERT_V(size<0>(tCsB) == size<0>(tCgC));
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB));
    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC));
    CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC));

    clear(tCrC);

    auto K_TILE_MAX = size<2>(tAgA);

    for (size_t k_tile = 0U; k_tile < K_TILE_MAX; ++k_tile)
    {
        // copy from global memory to shared memory
        copy(tAgA(_, _, k_tile), tAsA);
        copy(tBgB(_, _, k_tile), tBsB);

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // compute
        gemm(tCsA, tCsB, tCrC);
        __syncthreads();
    }

    // epilogue
    axpby(alpha, tCrC, beta, tCgC);
}

// launch

template <typename T>
void launch_official_cute_gemm_v01(
    size_t m, size_t n, size_t k,
    const T *alpha,
    const T *A, size_t lda,
    const T *B, size_t ldb,
    const T *beta,
    T *C, size_t ldc,
    cudaStream_t stream
)
{
    using namespace cute;
    unsigned int M = static_cast<unsigned int>(m);
    unsigned int N = static_cast<unsigned int>(n);
    unsigned int K = static_cast<unsigned int>(k);

    // problem shape
    auto shape_mnk = make_shape(M, N, K);

    // stride
    // cublas covenience for TN gemm
    // all matrices are in column major
    // A (m,k) --> transpose --> A(k, m) --> cute layout: A (m, k) : (k, 1) --> lda = k
    // B (k,n) --> cute layout: B (n, k) : (k, 1) --> ldb = k
    // C (m,n) --> cute layout: C (m, n) : (1, n) --> ldc = n

    auto dA = make_stride(lda, _1{});
    auto dB = make_stride(ldb, _1{});
    auto dC = make_stride(_1{}, ldc);

    // cta tile
    auto BLOCK_TILE_SIZE_M = _64{};
    auto BLOCK_TILE_SIZE_N = _64{};
    auto BLOCK_TILE_SIZE_K = _16{};

    // cta tiler
    auto cta_tiler = make_shape(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K);

    // smem layout
    auto sA_layout = make_layout(make_shape(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_K), LayoutRight{});
    auto sB_layout = make_layout(make_shape(BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K), LayoutRight{});
    auto sC_layout = make_layout(make_shape(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N));

    // thread layout for copy
    auto tA_layout = make_layout(make_shape(_32{}, _8{}), LayoutRight{});
    auto tB_layout = make_layout(make_shape(_32{}, _8{}), LayoutRight{});
    
    // thread layout for compute
    auto tC_layout = make_layout(make_shape(_16{}, _16{}));

    // grid, block
    dim3 block{size(tC_layout), 1U, 1U};
    dim3 grid{
        size(ceil_div(N, BLOCK_TILE_SIZE_N)),
        size(ceil_div(M, BLOCK_TILE_SIZE_M)),
        1U
    };

    // kernel
    official_cute_gemm_v01<<<grid, block, 0, stream>>>(
        shape_mnk, cta_tiler,
        A, dA, sA_layout, tA_layout,
        B, dB, sB_layout, tB_layout,
        C, dC, sC_layout, tC_layout,
        *alpha, *beta
    );
}

// explicit instantiation
template void launch_official_cute_gemm_v01<float>(size_t m, size_t n, size_t k,
                                   const float *alpha,
                                   const float *A, size_t lda,
                                   const float *B, size_t ldb,
                                   const float *beta,
                                   float *C, size_t ldc,
                                   cudaStream_t stream);

template void launch_official_cute_gemm_v01<cute::half_t>(size_t m, size_t n, size_t k,
                                    const cute::half_t *alpha,
                                    const cute::half_t *A, size_t lda,
                                    const cute::half_t *B, size_t ldb,
                                    const cute::half_t *beta,
                                    cute::half_t *C, size_t ldc,
                                    cudaStream_t stream);   