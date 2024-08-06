#include <cute/tensor.hpp>

#include "cuda_gemm.hpp"

// kernel
template <
    class ProblemShape, class CtaTiler,
    class T, class AStride, class BStride, class CStride,
    class TiledMMA
>
__global__ void cute_gemm_v00(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    const T* A, AStride dA,
    const T* B, BStride dB,
    T* C, CStride dC,
    TiledMMA tiled_mma,
    const T alpha, const T beta
)
{   
    using namespace cute;

    // global full tensor
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // M x K
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // N x K
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // M x N

    // global tile tensor
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // BLOCK_TILE_SIZE_M x BLOCK_TILE_SIZE_K x num_tiles_k
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // BLOCK_TILE_SIZE_N x BLOCK_TILE_SIZE_K x num_tiles_k
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // BLOCK_TILE_SIZE_M x BLOCK_TILE_SIZE_N

    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    // partition gA, gB, gC
    Tensor tAgA = thr_mma.partition_A(gA);  // (MMA,MMA_M,MMA_K, num_tiles_k)
    Tensor tBgB = thr_mma.partition_B(gB);  // (MMA,MMA_N,MMA_K, num_tiles_k)
    Tensor tCgC = thr_mma.partition_C(gC);  // (MMA,MMA_M,MMA_N)

    // make fragments in thread registers
    Tensor tArA = thr_mma.partition_fragment_A(gA(_, _, 0));    // MMA x MMA_M x MMA_K
    Tensor tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));    // MMA x MMA_N x MMA_K
    Tensor tCrC = thr_mma.partition_fragment_C(gC);             // MMA x MMA_M x MMA_N
    
    clear(tCrC);

    auto num_tiles_k = size<3>(tAgA);
    
    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
        // copy from global memory to thread private memory
        copy(tAgA(_, _, _, k_tile), tArA);
        copy(tBgB(_, _, _, k_tile), tBrB);

        // no need to sync here because the compute is independent with other threads

        // compute
        gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }

    axpby(alpha, tCrC, beta, tCgC);
}

// launch
template<typename T>
void launch_cute_gemm_kernel_v00(
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
    
    // problem shape
    unsigned int M = static_cast<unsigned int>(m);
    unsigned int N = static_cast<unsigned int>(n);
    unsigned int K = static_cast<unsigned int>(k);
    auto shape_MNK = make_shape(M, N, K);

    // A, B, C stride
    auto dA = make_stride(lda, _1{});
    auto dB = make_stride(ldb, _1{});
    auto dC = make_stride(_1{}, ldc);

    // block shape and cta tiler
    auto BLOCK_TILE_SIZE_M = _64{};
    auto BLOCK_TILE_SIZE_N = _64{};
    auto BLOCK_TILE_SIZE_K = _16{};
    auto cta_tiler = make_shape(BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K);

    // TiledMMA
    using mma_op = SM80_16x8x8_F32TF32TF32F32_TN;
    if (std::is_same_v<T, float>) 
    {
        using mma_op = SM80_16x8x8_F32TF32TF32F32_TN;
    }
    else if (std::is_same_v<T, cute::half_t>)
    {
        using mma_op = SM80_16x8x8_F16F16F16F16_TN;
    }

    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    
    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    static constexpr int kMmaVRepeatM = 1;
    static constexpr int kMmaVRepeatN = 2;
    static constexpr int kMmaVRepeatK = 1;

    // this is for problem shape (16x2) x (8x2x2) x (16x1) = 32x32x16
    
    using MMA_EU_RepeatT = decltype(
        make_layout(
            make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})
        )
    );

    using MMA_V_RepeatT = decltype(
        make_layout(
            make_shape(Int<kMmaVRepeatM>{}, Int<kMmaVRepeatN>{}, Int<kMmaVRepeatK>{})
        )
    );

    using TiledMMA = decltype(
        make_tiled_mma(
            mma_atom{},
            MMA_EU_RepeatT{},
            MMA_V_RepeatT{}
        )
    );

    // grid, block
    dim3 block{size(TiledMMA{}), 1U, 1U};
    dim3 grid{
        size(ceil_div(N, BLOCK_TILE_SIZE_N)),
        size(ceil_div(M, BLOCK_TILE_SIZE_M)),
        1U
    };

    // launch kernel
    cute_gemm_v00<<<grid, block, 0, stream>>>(
        shape_MNK, cta_tiler,
        A, dA,
        B, dB,
        C, dC,
        TiledMMA{},
        *alpha, *beta
    );

}

// explicit instantiation
template void launch_cute_gemm_kernel_v00<float>(size_t m, size_t n, size_t k,
                                   const float *alpha,
                                   const float *A, size_t lda,
                                   const float *B, size_t ldb,
                                   const float *beta,
                                   float *C, size_t ldc,
                                   cudaStream_t stream);
                                   
template void launch_cute_gemm_kernel_v00<cute::half_t>(size_t m, size_t n, size_t k,
                                    const cute::half_t *alpha,
                                    const cute::half_t *A, size_t lda,
                                    const cute::half_t *B, size_t ldb,
                                    const cute::half_t *beta,
                                    cute::half_t *C, size_t ldc,
                                    cudaStream_t stream);                                    