#include <cuda_fp16.h>
#include <cute/tensor.hpp>

#include "cuda_gemm.hpp"

// kernel
template <
    class T,
    class ProblemShape, class CtaTiler,
    class AStride, class ASmemLayout, class TiledCopyA,
    class BStride, class BSmemLayout, class TiledCopyB,
    class CStride, class CsmemLayout, class TiledMma
>
__global__ 
__launch_bounds__(decltype(size(TiledMma{}))::value)
void official_cute_gemm_v02(
    ProblemShape shape_mnk, CtaTiler cta_tiler,
    const T *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    const T *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    T *C, CStride dC, CsmemLayout , TiledMma mma,
    T alpha, T beta
)
{
    using namespace cute;
    
    // assert
    CUTE_STATIC_ASSERT_V(rank(shape_mnk) == _3{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == _3{});

    CUTE_STATIC_ASSERT_V(size(mma) == size(copy_a));
    CUTE_STATIC_ASSERT_V(size(mma) == size(copy_b));

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

    // use copy_a to partition gA, sA
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x); // get workload for 1 thread for 1 CPY Atom
    Tensor tAgA = thr_copy_a.partition_S(gA);           // (CPY,CPY_M,CPY_K, k) 
    Tensor tAsA = thr_copy_a.partition_D(sA);           // (CPY,CPY_M,CPY_K)
    Tensor tArA = make_fragment_like(tAsA);

    // use copy_b to partiton gB, sB
    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x); // get workload for 1 thread for 1 CPY Atom
    Tensor tBgB = thr_copy_b.partition_S(gB);           // (CPY,CPY_N,CPY_K, k)
    Tensor tBsB = thr_copy_b.partition_D(sB);           // (CPY,CPY_N,CPY_K)
    Tensor tBrB = make_fragment_like(tBsB);

    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tAsA) == size<1>(tArA)); // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAsA) == size<2>(tArA)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBsB) == size<1>(tBrB)); // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBsB) == size<2>(tBrB)); // CPY_K

    // copy from global memory to thread private memory
    copy(copy_a, tAgA(_, _, _, 0), tArA);
    copy(copy_b, tBgB(_, _, _, 0), tBrB);

    // use mma to partition sA, sB, gC
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);    // get workload for 1 thread for 1 MMA
    Tensor tCsA = thr_mma.partition_A(sA);          // (MMA,MMA_M,MMA_K) 
    Tensor tCsB = thr_mma.partition_B(sB);          // (MMA,MMA_N,MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC);          // (MMA,MMA_M,MMA_N)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    CUTE_STATIC_ASSERT_V(shape(tCgC) == shape(tCrC));
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCgC));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(tCgC));

    clear(tCrC);

    auto K_TILE_MAX = size<3>(tAgA);

    for (size_t k_tile = 0U; k_tile < K_TILE_MAX; ++k_tile)
    {
        __syncthreads();
        // copy from thread private memory to shared memory
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();

        // copy from global memory to thread private memory
        int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
        copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
        copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);

        gemm(mma, tCsA, tCsB, tCrC);
    }

    // epilogue
    axpby(alpha, tCrC, beta, tCgC);
}

// launch
template <typename T>
void launch_official_cute_gemm_v02(
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

    static constexpr int num_units = sizeof(uint128_t) / sizeof(T);
    static constexpr int kCopyEURepeatK = 16 / num_units;

    TiledCopy copy_a = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, T>{},
                                      Layout<Shape<_64,Int<kCopyEURepeatK>>, Stride<Int<kCopyEURepeatK>,_1>>{},
                                      Layout<Shape<_1, Int<num_units>>>{});

    TiledCopy copy_b = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, T>{},
                                      Layout<Shape<_64,Int<kCopyEURepeatK>>, Stride<Int<kCopyEURepeatK>,_1>>{},
                                      Layout<Shape<_1, Int<num_units>>>{});
    // TiledMMA
    // Each UniversalFMA thread will do a 1x1x1 MMA
    // This TiledMMA is for the problem shape: 16x16x1
    static constexpr int kMmaEURepeatN = 64 * kCopyEURepeatK / 16;
    TiledMMA mma_c = make_tiled_mma(UniversalFMA<T, T, T>{},
                                   Layout<Shape<_16,Int<kMmaEURepeatN>,_1>>{});

    // grid, block
    dim3 block{size(copy_a), 1U, 1U};
    dim3 grid{
        size(ceil_div(N, BLOCK_TILE_SIZE_N)),
        size(ceil_div(M, BLOCK_TILE_SIZE_M)),
        1U
    };

    // kernel
    official_cute_gemm_v02<<<grid, block, 0, stream>>>(
        shape_mnk, cta_tiler,
        A, dA, sA_layout, copy_a,
        B, dB, sB_layout, copy_b,
        C, dC, sC_layout, mma_c,
        *alpha, *beta
    );
}

// explicit instantiation
template void launch_official_cute_gemm_v02<float>(size_t m, size_t n, size_t k,
                                   const float *alpha,
                                   const float *A, size_t lda,
                                   const float *B, size_t ldb,
                                   const float *beta,
                                   float *C, size_t ldc,
                                   cudaStream_t stream);
                            
template void launch_official_cute_gemm_v02<cute::half_t>(size_t m, size_t n, size_t k,
                                    const cute::half_t *alpha,
                                    const cute::half_t *A, size_t lda,
                                    const cute::half_t *B, size_t ldb,
                                    const cute::half_t *beta,
                                    cute::half_t *C, size_t ldc,
                                    cudaStream_t stream);                                                                    
