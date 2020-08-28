//
// Created by Karl Pierce on 7/24/19.
//

#ifndef BTAS_GENERIC_COUPLED_CP_ALS_H
#define BTAS_GENERIC_COUPLED_CP_ALS_H

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace btas{

  /** \brief Computes the Canonical Product (CP) decomposition of two order-N
    tensors using the loss function $f = \|B - \hat{B}\| + \|Z - \hat{Z}\|$
    where $B$ and $Z$ have a coupled dimension $B \in \mathbb{R}^{X \times \dots}$
    and $Z \in \mathbb{R}^{X \times \dots}$ and thus share a factor matrix
    Decomposition optimization will use alternating least squares (ALS).

   \warning this code takes a non-const reference \c tensor_ref but does
   not modify the values. This is a result of API (reshape needs non-const tensor)

    Synopsis:
    \code
    // Constructors
    COUPLED_CP_ALS A(B, Z)              // COUPLED_CP_ALS object with empty factor
                                        // matrices and no symmetries
    COUPLED_CP_ALS A(B, Z, symms)       // COUPLED_CP_ALS object with empty factor
                                        // matrices and symmetries

    // Operations
    A.compute_rank(rank, converge_test)             // Computes the CP_ALS of tensors to
                                                    // rank, rank build and HOSVD options

    A.compute_rank_random(rank, converge_test)      // Computes the CP_ALS of tensors to
                                                    // rank. Factor matrices built at rank
                                                    // with random numbers

    A.compute_error(converge_test, omega)           // Computes the CP_ALS of tensors to
                                                    // 2-norm
                                                    // error < omega.

    A.compute_geometric(rank, converge_test, step)  // Computes CP_ALS of tensors to
                                                    // rank with
                                                    // geometric steps of step between
                                                    // guesses.

    A.compute_PALS(converge_test)                   // Not yet implemented.
                                                    // computes CP_ALS of tensors to
                                                    // rank = 3 * max_dim(tensor)
                                                    // in 4 panels using a modified
                                                    // HOSVD initial guess

   //See documentation for full range of options

    // Accessing Factor Matrices
    A.get_factor_matrices()             // Returns a vector of factor matrices, if
                                        // they have been computed

    A.reconstruct()                     // Returns the tensor $T$ computed using the
                                        // CP factor matrices
    \endcode
  */
  template <typename Tensor, typename ConvClass = NormCheck<Tensor>>
  class COUPLED_CP_ALS : public CP<Tensor, ConvClass> {
  public:
    using CP<Tensor,ConvClass>::A;
    using CP<Tensor,ConvClass>::ndim;
    using CP<Tensor,ConvClass>::normCol;
    using CP<Tensor,ConvClass>::generate_KRP;
    using CP<Tensor,ConvClass>::generate_V;
    using CP<Tensor,ConvClass>::norm;
    using CP<Tensor,ConvClass>::symmetries;
    using typename CP<Tensor,ConvClass>::ind_t;
    using typename CP<Tensor,ConvClass>::ord_t;

    /// Create a COUPLED CP ALS object, child class of the CP object
    /// that stores the reference tensors.
    /// Reference tensor has no symmetries.
    /// \param[in] left the reference tensor, $B$ to be decomposed.
    /// \param[in] right the reference tensor, $Z$ to be decomposed.
    COUPLED_CP_ALS(Tensor& left, Tensor& right) :
            CP<Tensor, ConvClass>(left.rank() + right.rank() - 1),
            tensor_ref_left(left), tensor_ref_right(right), ndimL(left.rank())
    {
      for (size_t i = 0; i < ndim; ++i) {
        symmetries.push_back(i);
      }
    }

    /// Create a CP ALS object, child class of the CP object
    /// that stores the reference tensors.
    /// Reference tensor has symmetries.
    /// Symmetries should be set such that the higher modes index
    /// are set equal to lower mode indices (a 4th order tensor,
    /// where the second & third modes are equal would have a
    /// symmetries of {0,1,1,3}
    /// \param[in] left the reference tensor, $B$ to be decomposed.
    /// \param[in] right the reference tensor, $Z$ to be decomposed.
    /// \param[in] symms the symmetries of the reference tensor.
    COUPLED_CP_ALS(Tensor &left, Tensor &right, std::vector<size_t> &symms) :
            CP<Tensor, ConvClass>(left.rank() + right.rank()),
            tensor_ref_left(left), tensor_ref_right(right), ndimL(left.rank())
    {
      symmetries = symms;
      for (size_t i = 0; i < ndim; ++i) {
        if (symmetries[i] > i) BTAS_EXCEPTION("Symmetries should always refer to factors at earlier positions");
      }
      if (symmetries.size() != ndim) BTAS_EXCEPTION(
              "Tensor describing symmetries must be defined for all dimensions");
    }

    /// \brief Computes decomposition of the order-N tensor \c tensor
    /// with rank = \c RankStep * \c panels *  max_dim(reference_tensor) + max_dim(reference_tensor)
    /// Initial guess for factor matrices start at rank = max_dim(reference_tensor)
    /// and builds rank \c panel times by \c RankStep * max_dim(reference_tensor) increments

    /// \param[in, out] converge_list Tests to see if ALS is converged, holds the value of fit.
    /// should be as many tests as there are panels
    /// \param[in] RankStep CP_ALS increment of the panel
    /// \param[in] panels number of times the rank will be built
    /// \param[in]
    /// max_als Max number of iterations allowed to converge the ALS approximation default = 1e4
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// default = true
    /// \param[in]
    /// calculate_epsilon Should the 2-norm error be calculated \f$ ||T_{\rm exact} -
    /// T_{\rm approx}|| = \epsilon. \f$ Default = false.
    /// \param[in] direct Should the CP decomposition be computed without
    /// calculating the Khatri-Rao product? Default = true.
    /// \returns 2-norm
    /// error between exact and approximate tensor, -1 if calculate_epsilon =
    /// false && ConvClass != FitCheck.
    double compute_PALS(std::vector<ConvClass> &converge_list, double RankStep = 0.5, size_t panels = 4,
                        int max_als = 20, bool fast_pI = false, bool calculate_epsilon = false,
                        bool direct = true) override {
      BTAS_EXCEPTION("Function not yet implemented");
    }

  protected:
    Tensor &tensor_ref_left;        // Tensor in first term of the loss function
    Tensor &tensor_ref_right;       // Tensor in second term of the loss function
    size_t ndimL;                      // Number of dimensions the left tensor has

    /// Creates an initial guess by computing the SVD of each mode
    /// If the rank of the mode is smaller than the CP rank requested
    /// The rest of the factor matrix is filled with random numbers
    /// Builds factor matricies starting with R=(1 or SVD_rank)
    /// and moves to R = \c rank
    /// incrementing column dimension, R, by \c step

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit.
    /// \param[in] direct The CP decomposition be computed without calculating the
    /// Khatri-Rao product?
    /// \param[in] max_als If CP decomposition is to finite
    /// error, max_als is the highest rank approximation computed before giving up
    /// on CP-ALS.
    /// \param[in] calculate_epsilon Should the 2-norm
    /// error be calculated \f$ ||T_{\rm exact} - T_{\rm approx}|| = \epsilon \f$ .
    /// \param[in] step
    /// CP_ALS built from r =1 to r = rank. r increments by step.
    /// \param[in, out] epsilon The 2-norm
    /// error between the exact and approximated reference tensor
    /// \param[in] SVD_initial_guess build inital guess from left singular vectors
    /// \param[in] SVD_rank rank of the initial guess using left singular vector
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    // TODO make use of symmetries in this function
    void build(ind_t rank, ConvClass &converge_test, bool direct, ind_t max_als,
               bool calculate_epsilon,
               ind_t step, double &epsilon, bool SVD_initial_guess, ind_t SVD_rank,
               bool &fast_pI) override {
      // If its the first time into build and SVD_initial_guess
      // build and optimize the initial guess based on the left
      // singular vectors of the reference tensor.
      if (A.empty() && SVD_initial_guess) {
        if (SVD_rank == 0) BTAS_EXCEPTION("Must specify the rank of the initial approximation using SVD");

        std::vector<int> modes_w_dim_LT_svd;
        A = std::vector<Tensor>(ndim);

        // Determine which factor matrices one can fill using SVD initial guess
        // start with left then do right
        size_t ndimR = tensor_ref_right.rank();
        for (size_t i = 0; i < ndimL; i++) {
          if (tensor_ref_left.extent(i) < SVD_rank) {
            modes_w_dim_LT_svd.push_back(i);
          }
        }
        for (size_t i = 1; i < ndimR; i++) {
          if (tensor_ref_right.extent(i) < SVD_rank) {
            modes_w_dim_LT_svd.push_back(i + ndimL - 1);
          }
        }

        for (size_t tensor = 0; tensor < 2; ++tensor) {
          auto &tensor_ref = tensor == 0 ? tensor_ref_left : tensor_ref_right;
          size_t ndim_curr = tensor_ref.rank();
          // Fill all factor matrices with their singular vectors,
          // because we contract X X^T (where X is reference tensor) to make finding
          // singular vectors an eigenvalue problem some factor matrices will not be
          // full rank;
          bool left = tensor == 0;
          for (size_t i = left ? 0 : 1; i < ndim_curr; i++) {
            ind_t R = tensor_ref.extent(i);
            Tensor S(R, R), lambda(R);

            // Contract refrence tensor to make it square matrix of mode i
            gemm(CblasNoTrans, CblasTrans, 1.0, flatten(tensor_ref, i), flatten(tensor_ref, i), 0.0, S);

            // Find the Singular vectors of the matrix using eigenvalue decomposition
            eigenvalue_decomp(S, lambda);

            // Fill a factor matrix with the singular vectors with the largest corresponding singular
            // values
            lambda = Tensor(R, SVD_rank);
            lambda.fill(0.0);
            auto lower_bound = {0,0};
            auto upper_bound = {R, ((R > SVD_rank) ? SVD_rank : R)};
            auto view = make_view(S.range().slice(lower_bound, upper_bound), S.storage());
            auto l_iter = lambda.begin();
            for (auto iter = view.begin(); iter != view.end(); ++iter, ++l_iter) {
              *(l_iter) = *(iter);
            }

            size_t A_dim = left ? i : i + ndimL - 1;
            A[A_dim] = lambda;
          }
        }

        // srand(3);
        boost::random::mt19937 generator(random_seed_accessor());
        boost::random::uniform_real_distribution<> distribution(-1.0, 1.0);
        // Fill the remaining columns in the set of factor matrices with dimension < SVD_rank with random numbers
        for (auto &i : modes_w_dim_LT_svd) {
          size_t dim = i < ndimL ? i : i - ndimL + 1;
          auto &tensor_ref = i < ndimL ? tensor_ref_left : tensor_ref_right;
          ind_t R = tensor_ref.extent(dim), zero = 0;
          auto lower_bound = {zero, R};
          auto upper_bound = {R, SVD_rank};
          auto view = make_view(A[i].range().slice(lower_bound, upper_bound), A[i].storage());
          for (auto iter = view.begin(); iter != view.end(); ++iter) {
            *(iter) = distribution(generator);
          }
        }

        // Normalize the columns of the factor matrices and
        // set the values al lambda, the weigt of each order 1 tensor
        Tensor lambda(Range{Range1{SVD_rank}});
        A.push_back(lambda);
        for (size_t i = 0; i < ndim; ++i) {
          this->normCol(A[i]);
        }

        // Optimize this initial guess.
        ALS(SVD_rank, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI);
      }
      // This loop keeps track of column dimension
      for (ind_t i = (A.empty()) ? 0 : A.at(0).extent(1); i < rank; i += step) {
        ind_t rank_new = i + 1;
        // This loop walks through the factor matrices
        for (size_t j = 0; j < ndim; ++j) {  // select a factor matrix
          // If no factor matrices exists, make a set of factor matrices
          // and fill them with random numbers that are column normalized
          // and create the weighting vector lambda
          auto left = (j < ndimL);
          auto &tensor_ref = left ? tensor_ref_left : tensor_ref_right;
          if (i == 0) {
            Tensor a(Range{tensor_ref.range().range((left ? j : j - ndimL + 1)), Range1{rank_new}});
            a.fill(rand());
            A.push_back(a);
            this->normCol(j);
          }

            // If the factor matrices have memory allocated, rebuild each matrix
            // with new column dimension col_dimension_old + skip
            // fill the new columns with random numbers and normalize the columns
          else {
            ind_t row_extent = A[0].extent(0), rank_old = A[0].extent(1), zero = 0;
            Tensor b(Range{A[0].range().range(0), Range1{rank_new}});

            {
              auto lower_old = {zero, zero}, upper_old = {row_extent, rank_old};
              auto old_view = make_view(b.range().slice(lower_old, upper_old), b.storage());
              auto A_itr = A[0].begin();
              for(auto iter = old_view.begin(); iter != old_view.end(); ++iter, ++A_itr){
                *(iter) = *(A_itr);
              }
            }

            {
              auto lower_new = {zero, rank_old}, upper_new = {row_extent, rank_new};
              auto new_view = make_view(b.range().slice(lower_new, upper_new), b.storage());
              boost::random::mt19937 generator(random_seed_accessor());
              boost::random::uniform_real_distribution<> distribution(-1.0, 1.0);
              for (auto iter = new_view.begin(); iter != new_view.end(); ++iter) {
                *(iter) = distribution(generator);
              }
            }

            A.erase(A.begin());
            A.push_back(b);
            if (j + 1 == ndim) {
              A.erase(A.begin());
            }
          }
        }
        {
          Tensor lam(Range{Range1{rank_new}});
          A.push_back(lam);
        }
        // compute the ALS of factor matrices with rank = i + 1.
        ALS(rank_new, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI);
      }
    }

    /// Create a rank \c rank initial guess using
    /// random numbers from a uniform distribution

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit.
    /// \param[in] direct The CP decomposition be computed without calculating the
    /// Khatri-Rao product?
    /// \param[in] max_als If CP decomposition is to finite
    /// error, max_als is the highest rank approximation computed before giving up
    /// on CP-ALS.
    /// \param[in] calculate_epsilon Should the 2-norm
    /// error be calculated \f$ ||T_{\rm exact} - T_{\rm approx}|| = \epsilon \f$ .
    /// \param[in] step
    /// CP_ALS built from r =1 to r = rank. r increments by step.
    /// \param[in, out] epsilon The 2-norm
    /// error between the exact and approximated reference tensor
    /// \param[in] SVD_initial_guess build inital guess from left singular vectors
    /// \param[in] SVD_rank rank of the initial guess using left singular vector
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    void build_random(ind_t rank, ConvClass &converge_test, bool direct, ind_t max_als,
                      bool calculate_epsilon, double &epsilon,
                      bool &fast_pI) override {
      BTAS_EXCEPTION("Function not yet implemented");
    }

    /// performs the ALS method to minimize the loss function for a single rank
    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit.
    /// \param[in] dir The CP decomposition be computed without calculating the
    /// Khatri-Rao product?
    /// \param[in] max_als If CP decomposition is to finite
    /// error, max_als is the highest rank approximation computed before giving up
    /// on CP-ALS. Default = 1e5.
    /// \param[in] calculate_epsilon Should the 2-norm
    /// error be calculated ||T_exact - T_approx|| = epsilon.
    /// \param[in] tcutALS
    /// How small difference in factor matrices must be to consider ALS of a
    /// single rank converged. Default = 0.1.
    /// \param[in, out] epsilon The 2-norm
    /// error between the exact and approximated reference tensor
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    void
    ALS(ind_t rank, ConvClass &converge_test, bool dir, unsigned int max_als, bool calculate_epsilon,
        double &epsilon, bool &fast_pI) {

      size_t count = 0;
      // Until either the initial guess is converged or it runs out of iterations
      // update the factor matrices with or without Khatri-Rao product
      // intermediate
      bool is_converged = false;
      bool matlab = fast_pI;
      while (count < max_als && !is_converged) {
        count++;
        this->num_ALS++;
        for (size_t i = 0; i < ndim; i++) {
          auto tmp = symmetries[i];
          if (tmp == i) {
            direct(i, rank, fast_pI, matlab, converge_test);
          } else {
            A[i] = A[tmp];
          }
        }
        detail::get_fit(converge_test, epsilon);
        is_converged = converge_test(A);
      }
    }

    /// Computes an optimized factor matrix holding all others constant.
    /// No Khatri-Rao product computed, immediate contraction
    /// Does this by first contracting a factor matrix with the refrence tensor
    /// Then computes hadamard/contraction products along all other modes except n.

    /// In this we are minimizing a sum of two norms.  If n = coupled dimension then we have to minimize
    /// f = ||B^{X}_{abcd...} - \hat{B}^X_{abcd...} || + || B^{X}_{ijkl...} - \hat{B}^X_{ijkl...}||
    /// where X is the coupled dimension. otherwise we just minimize one of the two terms.

    /// \param[in] n The mode being optimized, all other modes held constant
    /// \param[in] rank The current rank, column dimension of the factor matrices
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// return if computing the fast_pI was successful.
    /// \param[in, out] matlab If \c fast_pI = true then try to solve VA = B instead of taking pseudoinverse
    /// in the same manner that matlab would compute the inverse.
    /// return if computing the inverse in this was was successful
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit.
    void direct(size_t n, ind_t rank, bool &fast_pI, bool &matlab, ConvClass &converge_test) {
      if (n == 0) {
        // Start by computing (B^{X}_{abcd...} C^{-X}_{abcd...}) + B^{X}_{ijkl...} C^{-X}_{ijkl...}) = K
        // where C^{-X}_{abcd...} = C^{a} \odot C^{b} \odot C^{c} \odot C^{d} \dots ( the khatri-rao
        // product without the factor matrix C^X
        ind_t coupled_dim = tensor_ref_left.extent(0);
        Tensor K(coupled_dim, rank);
        K.fill(0.0);

        for (int tensor = 0; tensor < 2; ++tensor) {
          auto left = tensor == 0;
          auto &tensor_ref = left ? tensor_ref_left : tensor_ref_right;


          size_t ndim_curr = tensor_ref.rank(),
                  A_dim = left ? ndimL - 1 : this->ndim - 1,
                  contract_size = tensor_ref.extent(ndim_curr - 1),
                  LHSsize = tensor_ref.size() / contract_size;

          auto R = tensor_ref.range();

          Tensor contract_tensor(LHSsize, rank);
          tensor_ref.resize(Range{Range1{LHSsize}, Range1{contract_size}});
          gemm(CblasNoTrans, CblasNoTrans, 1.0, tensor_ref, A[A_dim], 0.0, contract_tensor);
          tensor_ref.resize(R);
          --A_dim;
          for (size_t contract_dim = ndim_curr - 2; contract_dim > 0; --contract_dim, --A_dim) {
            contract_size = tensor_ref.extent(contract_dim);
            LHSsize /= contract_size;

            contract_tensor.resize(Range{Range1{LHSsize}, Range1{contract_size}, Range1{rank}});
            Tensor temp(LHSsize, rank);
            temp.fill(0.0);
            const auto &a = A[A_dim];
            for (ord_t i = 0; i < LHSsize; ++i) {
              for (ord_t k = 0; k < contract_size; ++k) {
                for (ord_t r = 0; r < rank; ++r) {
                  temp(i, r) += contract_tensor(i, k, r) * a(k, r);
                }
              }
            }
            contract_tensor = temp;
          }
          K += contract_tensor;
        }

        // Next form the Hadamard product sum
        // J = (C^{a\quadT} C^a * C^{b\quadT} C^b * \dots + C^{i\quadT} C^i * C^{j\quadT} C^j + \dots
        Tensor J1(rank, rank);
        J1.fill(1.0);
        auto rank2 = rank * (ord_t) rank;
        {
          for (size_t i = 1; i < ndimL; ++i) {
            Tensor temp(rank, rank);
            gemm(CblasTrans, CblasNoTrans, 1.0, A[i], A[i], 0.0, temp);
            for (ord_t j = 0; j < rank2; ++j) {
              *(J1.data() + j) *= *(temp.data() + j);
            }
          }
          Tensor J2(rank, rank);
          J2.fill(1.0);
          for (size_t i = ndimL; i < ndim; ++i) {
            Tensor temp(rank, rank);
            gemm(CblasTrans, CblasNoTrans, 1.0, A[i], A[i], 0.0, temp);
            for (ord_t j = 0; j < rank2; ++j) {
              *(J2.data() + j) *= *(temp.data() + j);
            }
          }
          J1 += J2;
        }
        // Finally Form the product of K * J^\dagger
        Tensor a0(coupled_dim, rank);
        gemm(CblasNoTrans, CblasNoTrans, 1.0, K, pseudoInverse(J1, fast_pI), 0.0, a0);
        this->normCol(a0);
        A[0] = a0;
      }
      else {
        bool left = n < ndimL;
        Tensor &tensor_ref = left ? tensor_ref_left : tensor_ref_right;

        size_t ndim_curr = tensor_ref.rank(),
                A_dim = 0,
                contract_size = tensor_ref.extent(0),
                LHSsize = tensor_ref.size() / contract_size,
                pseudo_rank = rank,
                skip_dim = A[n].extent(0);
        auto R = tensor_ref.range();

        tensor_ref.resize(Range{Range1{contract_size}, Range1{LHSsize}});
        Tensor contract_tensor(LHSsize, rank);
        gemm(CblasTrans, CblasNoTrans, 1.0, tensor_ref, A[A_dim], 0.0, contract_tensor);

        tensor_ref.resize(R);
        A_dim = left ? ndimL - 1 : ndim - 1;
        // TODO Use pointer arithmetic here instead of () operator
        for (size_t contract_dim = ndim_curr - 1; contract_dim > 0; --contract_dim, --A_dim) {
          contract_size = tensor_ref.extent(contract_dim);
          LHSsize /= contract_size;
          contract_tensor.resize(Range{Range1{LHSsize}, Range1{contract_size}, Range1{pseudo_rank}});
          const auto &currA = A[A_dim];

          if (A_dim == n) {
            pseudo_rank *= contract_size;
          } else if (A_dim > n) {
            Tensor temp(LHSsize, pseudo_rank);
            temp.fill(0.0);
            for (ord_t i = 0; i < LHSsize; ++i) {
              for (ord_t j = 0; j < contract_size; ++j) {
                for (ord_t r = 0; r < rank; ++r) {
                  temp(i, r) += contract_tensor(i, j, r) * currA(j, r);
                }
              }
            }
            contract_tensor = temp;
          }
          else {
            Tensor temp(LHSsize, pseudo_rank);
            temp.fill(0.0);
            for (ord_t i = 0; i < LHSsize; ++i) {
              for (ord_t j = 0; j < contract_size; ++j) {
                for (ord_t k = 0; k < skip_dim; ++k) {
                  for (ord_t r = 0; r < rank; ++r) {
                    temp(i, r + k * rank) += contract_tensor(i, j, r + k * rank) * currA(j, r);
                  }
                }
              }
            }
            contract_tensor = temp;
          }
        }
        contract_tensor.resize(Range{Range1{skip_dim}, Range1{rank}});

        Tensor G(rank, rank);
        gemm(CblasTrans, CblasNoTrans, 1.0, A[0], A[0], 0.0, G);
        auto rank2 = rank * (ord_t) rank;
        for (size_t i = (left ? 1 : ndimL); i < (left ? ndimL : ndim); ++i) {
          if (i != n) {
            Tensor temp(rank, rank);
            gemm(CblasTrans, CblasNoTrans, 1.0, A[i], A[i], 0.0, temp);
            for (ord_t j = 0; j < rank2; ++j) {
              *(G.data() + j) *= *(temp.data() + j);
            }
          }
        }
        if (n == ndimL - 1)
          detail::set_MtKRPL(converge_test, contract_tensor);
        else if(n == this->ndim - 1)
          detail::set_MtKRPR(converge_test, contract_tensor);
        Tensor an(skip_dim, rank);
        gemm(CblasNoTrans, CblasNoTrans, 1.0, contract_tensor, pseudoInverse(G, fast_pI), 0.0, an);
        this->normCol(an);
        A[n] = an;
      }
    }

  };

} // namespace btas

#endif //BTAS_GENERIC_COUPLED_CP_ALS_H
