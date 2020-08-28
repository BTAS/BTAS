#ifndef BTAS_RANDOMIZED_DECOMP_H
#define BTAS_RANDOMIZED_DECOMP_H

#include <btas/error.h>
#include <btas/generic/contract.h>
#include <btas/generic/core_contract.h>
#include <btas/generic/linear_algebra.h>
#include <btas/tensor.h>

#include <stdlib.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <random>
#include <vector>

namespace btas {

  /// \param[in,out] A In: An empty matrix of size column dimension of the nth
  /// mode flattened tensor provided to the randomized compression method by the
  /// desired rank of the randmoized compression method.  Out: A random matrix,
  /// column drawn from a random distribution and orthogonalized
  template <typename Tensor>
  void generate_random_metric(Tensor &A) {
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using value_type = typename Tensor::value_type;
    for (ind_t i = 0; i < A.extent(1); i++) {
      std::random_device rd;
      // uncomment for more randomness
      // std::mt19937 gen(rd());
      boost::random::mt19937 gen(random_seed_accessor());
      boost::random::uniform_real_distribution<> distribution(0.0, 10.0);
      value_type norm = 0.0;
      for (ind_t j = 0; j < A.extent(0); j++) {
        auto val = abs(distribution(gen));
        norm += val * val;
        A(i, j) = val;
      }

      norm = sqrt(norm);
      for (ind_t j = 0; j < A.extent(0); j++) {
        A(j, i) /= norm;
      }

      distribution.reset();
    }
  QR_decomp(A);
}

/// Calculates the randomized compression of tensor \c A.
/// <a href=https://arxiv.org/pdf/1703.09074.pdf> See reference </a>
/// \param[in, out] A In: An order-N tensor to be randomly decomposed.
/// Out: The core tensor of random decomposition \param[in, out] transforms
/// In: An empty vector.  Out: The randomized decomposition factor matrices.
/// \param[in] des_rank The rank of each mode of \c A after randomized
/// decomposition. \param[in] oversampl Oversampling added to \c
/// desired_compression_rank required to provide an optimal decomposition.
/// Default = suggested = 10. \param[in] powerit Number of power iterations, as
/// specified in the literature, to scale the spectrum of each mode. Default =
/// suggested = 2.
  template<typename Tensor>
  void randomized_decomposition(Tensor &A, std::vector<Tensor> &transforms,
                                long des_rank, size_t oversampl = 10,
                                size_t powerit = 2) {
    using ind_t = typename Tensor::range_type::index_type::value_type;

    // Add the oversampling to the desired rank
    size_t ndim = A.rank();
    ind_t rank = des_rank + oversampl;
    std::vector<size_t> A_modes;
    for (size_t i = 0; i < ndim; ++i) {
      A_modes.push_back(i);
    }
    std::vector<size_t> final_modes(A_modes);

    // Walk through all the modes of A
    for (size_t n = 0; n < ndim; n++) {
      // Flatten A
      auto An = flatten(A, n);

      // Make and fill the random matrix Gamma
      Tensor G(An.extent(1), rank);
      generate_random_metric(G);

      // Project The random matrix onto the flatten reference tensor
      Tensor Y(An.extent(0), rank);
      gemm(CblasNoTrans, CblasNoTrans, 1.0, An, G, 0.0, Y);

    // Start power iteration
      for (size_t j = 0; j < powerit; j++) {
        // Find L of an LU decomposition of the projected flattened tensor
        LU_decomp(Y);
        Tensor Z(An.extent(1), Y.extent(1));

        // Find the L of an LU decomposition of the L above (called Y) projected
        // onto the flattened reference tensor
        gemm(CblasTrans, CblasNoTrans, 1.0, An, Y, 0.0, Z);
        LU_decomp(Z);

        // Project the second L from above (called Z) onto the flattened reference
      // tensor and start power iteration over again.
      Y.resize(Range{Range1{An.extent(0)}, Range1{Z.extent(1)}});
      gemm(CblasNoTrans, CblasNoTrans, 1.0, An, Z, 0.0, Y);
    }

    // Compute the QR from Y above.  If the QR is non-singular push it into
    // transforms and project the unitary matrix onto the reference tensor
    bool QR_good = true;
    QR_good = QR_decomp(Y);

    if (!QR_good) {
      BTAS_EXCEPTION("QR did not complete successfully due to chosen "
                     "dimension. Choose desired_compression_rank <= smallest "
                     "dimension of tensor A");
    }

    transforms.push_back(Y);
  }
    std::vector<size_t> contract_modes;
  contract_modes.push_back(0); contract_modes.push_back(ndim);
    for (size_t n = 0; n < ndim; n++) {
#ifdef BTAS_HAS_INTEL_MKL
      core_contract(A, transforms[n], n);
#else
      std::vector<ind_t> final_dims;
      for (size_t j = 0; j < ndim; ++j) {
        if (j == n) {
          final_dims.push_back(transforms[n].extent(1));
        } else {
          final_dims.push_back(A.extent(j));
        }
      }
    contract_modes[0] = n;
    final_modes[n] = ndim;
    btas::Range final_range(final_dims);
    Tensor final(final_range);
    contract(1.0, A, A_modes, transforms[n], contract_modes, 0.0, final, final_modes);
    final_modes[n] = n;
    A = final;
#endif //BTAS_HAS_INTEL_MKL
  }
}
} // namespace btas

#endif // BTAS_RANDOMIZED_DECOMP_H
