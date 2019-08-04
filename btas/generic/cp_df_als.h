//
// Created by Karl Pierce on 7/24/19.
//

#ifndef BTAS_GENERIC_CP_DF_ALS_H
#define BTAS_GENERIC_CP_DF_ALS_H

#include <btas/btas.h>
#include <btas/error.h>
#include <btas/generic/cp.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <btas/generic/default_random_seed.h>
#include "core_contract.h"
#include "flatten.h"
#include "khatri_rao_product.h"
#include "randomized.h"
#include "swap.h"
#include "tucker.h"
#include "converge_class.h"
#include "rals_helper.h"
#include "reconstruct.h"

namespace btas{

  /** \brief Computes the Canonical Product (CP) decomposition of an order-N
    tensor where the tensor is represented as $T = B^T Z$
    where $B \in \mathbb{R}^{X \times I^1 \times I^2 \times \dots I^{n}}$
    and $Z \in \mathbb{R}^{X \times I^{n+1} \times \dots \tims I^N}$
    Here $X$ is called the connected dimension, no factor matrix will be recovered
    for this mode.
    Decomposition optimization will use alternating least squares (ALS).

    Synopsis:
    \code
    // Constructors
    CP_DF_ALS A(B, Z)                   // CP_DF_ALS object with empty factor
                                        // matrices and no symmetries
    CP_DF_ALS A(B, Z, symms)            // CP_DF_ALS object with empty factor
                                        // matrices and symmetries

    // Operations
    A.compute_rank(rank, converge_test)             // Computes the CP_ALS of $T$ tensor to
                                                    // rank, rank build and HOSVD options

    A.compute_rank_random(rank, converge_test)      // Computes the CP_ALS of $T$ tensor to
                                                    // rank. Factor matrices built at rank
                                                    // with random numbers

    A.compute_error(converge_test, omega)           // Computes the CP_ALS of $T$ tensor to
                                                    // 2-norm
                                                    // error < omega.

    A.compute_geometric(rank, converge_test, step)  // Computes CP_ALS of $T$ tensor to
                                                    // rank with
                                                    // geometric steps of step between
                                                    // guesses.

    A.compute_PALS(converge_test)                   // computes CP_ALS of $T$ tensor to
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
  template <typename Tensor, class ConvClass = NormCheck<Tensor>>
  class CP_DF_ALS : public CP<Tensor, ConvClass> {

  public:

    using CP<Tensor,ConvClass>::A;
    using CP<Tensor,ConvClass>::ndim;
    using CP<Tensor,ConvClass>::pseudoInverse;
    using CP<Tensor,ConvClass>::normCol;
    using CP<Tensor,ConvClass>::generate_KRP;
    using CP<Tensor,ConvClass>::generate_V;
    using CP<Tensor,ConvClass>::norm;
    using CP<Tensor,ConvClass>::symmetries;


    /// Create a CP DF ALS object, child class of the CP object
    /// that stores the reference tensors.
    /// Reference tensor has no symmetries.
    /// \param[in] left the reference tensor, $B$ to be decomposed.
    /// \param[in] right the reference tensor, $Z$ to be decomposed.
    CP_DF_ALS(Tensor &left, Tensor &right) :
            CP<Tensor,ConvClass>(left.rank() + right.rank() - 2),
            tensor_ref_left(left), tensor_ref_right(right),
            ndimL(left.rank()), ndimR(right.rank()) {
      for(int i = 0; i < ndim; ++i){
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
    CP_DF_ALS(Tensor &left, Tensor &right, std::vector<int> & symms) :
            CP<Tensor,ConvClass>(left.rank() + right.rank() - 2),
            tensor_ref_left(left), tensor_ref_right(right),
            ndimL(left.rank()), ndimR(right.rank())
    {
      symmetries = symms;
      for(int i = 0; i < ndim; ++i){
        if(symmetries[i] > i)
        BTAS_EXCEPTION("Symmetries should always refer to factors at earlier positions");
      }
      if (symmetries.size() != ndim) BTAS_EXCEPTION(
              "Tensor describing symmetries must be equal to number of non-connected dimensions");
    }

    CP_DF_ALS() = default;

    ~CP_DF_ALS() = default;

#ifdef _HAS_INTEL_MKL
    /// \breif Computes decomposition of the order-N tensor \c tensor
    /// with rank = \c RankStep * \c panels *  max_dim(reference_tensor) + max_dim(reference_tensor)
    /// Initial guess for factor matrices start at rank = max_dim(reference_tensor)
    /// and builds rank \c panel times by \c RankStep * max_dim(reference_tensor) increments

    /// \param[in] converge_test Test to see if ALS is converged
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
    double compute_PALS(std::vector<ConvClass> & converge_list, double RankStep = 0.5, int panels = 4,
                        int max_als = 20,bool fast_pI = true, bool calculate_epsilon = false, bool direct = true) override{
      if (RankStep <= 0) BTAS_EXCEPTION("Panel step size cannot be less than or equal to zero");
      if(converge_list.size() < panels) BTAS_EXCEPTION("Too few convergence tests.  Must provide a list of panels convergence tests");
      double epsilon = -1.0;
      int count = 0;
      // Find the largest rank this will be the first panel
      auto max_dim = tensor_ref_left.extent(0);
      for(int i = 1; i < ndimL; ++i){
        auto dim = tensor_ref_left.extent(i);
        max_dim = ( dim > max_dim ? dim : max_dim);
      }
      for(int i = 0; i < ndimR; ++i){
        auto dim = tensor_ref_right.extent(i);
        max_dim = (dim > max_dim ? dim : max_dim);
      }

      while(count < panels){
        auto converge_test = converge_list[count];
        // Use tucker initial guess (SVD) to compute the first panel
        if(count == 0) {
          build(max_dim, converge_test, direct, max_als, calculate_epsilon, 1, epsilon, true, max_dim, fast_pI);
          //build(max_dim, converge_test, max_als, calculate_epsilon, 1, epsilon, true, max_dim, fast_pI);
        }
          // All other panels build the rank buy RankStep variable
        else {
          // Always deal with the first matrix push new factors to the end of A
          // Kick out the first factor when it is replaced.
          // This is the easiest way to resize and preserve the columns
          // (if this is rebuilt with rank as columns this resize would be easier)
          int rank = A[0].extent(1), rank_new = rank +  RankStep * max_dim;
          for (int i = 0; i < ndim; ++i) {
            int row_extent = A[0].extent(0);
            Tensor b(Range{Range1{A[0].extent(0)}, Range1{rank_new}});

            // Move the old factor to the new larger matrix
            {
              auto lower_old = {0, 0}, upper_old = {row_extent, rank};
              auto old_view = make_view(b.range().slice(lower_old, upper_old), b.storage());
              auto A_itr = A[0].begin();
              for(auto iter = old_view.begin(); iter != old_view.end(); ++iter, ++A_itr){
                *(iter) = *(A_itr);
              }
            }

            // Fill in the new columns of the factor with random numbers
            {
              auto lower_new = {0, rank}, upper_new = {row_extent, rank_new};
              auto new_view = make_view(b.range().slice(lower_new, upper_new), b.storage());
              std::mt19937 generator(random_seed_accessor());
              std::uniform_real_distribution<> distribution(-1.0, 1.0);
              for(auto iter = new_view.begin(); iter != new_view.end(); ++iter){
                *(iter) = distribution(generator);
              }
            }

            A.erase(A.begin());
            A.push_back(b);
            // replace the lambda matrix when done with all the factors
            if (i + 1 == ndim) {
              b.resize(Range{Range1{rank_new}});
              for (int k = 0; k < A[0].extent(0); k++) b(k) = A[0](k);
              A.erase(A.begin());
              A.push_back(b);
            }
            // normalize the factor (don't replace the previous lambda matrix)
            normCol(0);
          }
          ALS(rank_new, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
        }
        count++;
        //if (count + 1 == panels) max_als = 1000;
      }
      std::cout << "Number of ALS iterations was " << this->num_ALS << std::endl;
      return epsilon;
    }
#endif // _HAS_INTEL_MKL

  protected:
    Tensor &tensor_ref_left;        // Left connected tensor
    Tensor &tensor_ref_right;       // Right connected tensor
    int ndimL;                      // Number of dimensions in left tensor
    int ndimR;                      // number of dims in the right tensor

    /// Creates an initial guess by computing the SVD of each mode
    /// If the rank of the mode is smaller than the CP rank requested
    /// The rest of the factor matrix is filled with random numbers
    /// Builds factor matricies starting with R=(1 or SVD_rank)
    /// and moves to R = \c rank
    /// incrementing column dimension, R, by \c step

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in] converge_test Test to see if ALS is converged.
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

    // TODO take advantage of symmetries in build
    void build(int rank, ConvClass &converge_test, bool direct, int max_als, bool calculate_epsilon, int step, double &epsilon,
               bool SVD_initial_guess, int SVD_rank, bool & fast_pI) override{
      {
        bool factors_set = false;
        // If its the first time into build and SVD_initial_guess
        // build and optimize the initial guess based on the left
        // singular vectors of the reference tensor.
#ifdef _HAS_INTEL_MKL
        if (A.empty() && SVD_initial_guess) {
          if (SVD_rank == 0) BTAS_EXCEPTION("Must specify the rank of the initial approximation using SVD");

          // easier to do this part by constructing tensor_ref
          // This is an N^5 step but it is only done once so it shouldn't be that expensive.
          // once the code is working this step can be reduced to N^4 (though it might be less accurate that way)
          // it is something to test.
          // Must reconstruct with matrix multiplication so
          // get size of product of dimensions (not connecting) for left and right side
          // Also need tensor dimensions to resize tensor_ref after contraction
          std::vector<int> TRdims(ndim);
          auto trLsize = 1.0, trRsize = 1.0;
          for(auto i = 1; i < ndimL; ++i){
            auto dim = tensor_ref_left.extent(i);
            TRdims[i - 1] = dim;
            trLsize *= dim;
          }
          for(auto i = 1; i < ndimR; ++i){
            auto dim = tensor_ref_right.extent(i);
            // i = 1 must take it to 0; then add left dimensions; then subtract 1 for connecting dimension
            TRdims[i + ndimR - 2] = dim;
            trRsize *= dim;
          }

          // Make TR with correct L/R size
          Tensor tensor_ref(trLsize, trRsize);

          // save ranges for after contraction to resize
          auto TRLrange = tensor_ref_left.range();
          auto TRRrange = tensor_ref_right.range();

          // resize tensors to matrices
          tensor_ref_left.resize(Range{Range1{tensor_ref_left.extent(0)},Range1{trLsize}});
          tensor_ref_right.resize(Range{Range1{tensor_ref_right.extent(0)},Range1{trRsize}});

          // matrix multiplication
          gemm(CblasTrans, CblasNoTrans, 1.0, tensor_ref_left, tensor_ref_right, 0.0, tensor_ref);

          // Resize tensor_ref's back to original dimensions
          tensor_ref_left.resize(TRLrange);
          tensor_ref_right.resize(TRRrange);
          tensor_ref.resize(TRdims);

          std::vector<int> modes_w_dim_LT_svd;
          A = std::vector<Tensor>(ndim);

          // Determine which factor matrices one can fill using SVD initial guess
          for(int i = 0; i < ndim; i++){
            if(tensor_ref.extent(i) < SVD_rank){
              modes_w_dim_LT_svd.push_back(i);
            }
          }

          // Fill all factor matrices with their singular vectors,
          // because we contract X X^T (where X is reference tensor) to make finding
          // singular vectors an eigenvalue problem some factor matrices will not be
          // full rank;
          A[0] = Tensor(tensor_ref.extent(0), SVD_rank);
          A[0].fill(0.0);

          for(int i = 0; i < ndim; i++){
            int R = tensor_ref.extent(i);
            Tensor S(R,R), lambda(R);

            // Contract refrence tensor to make it square matrix of mode i
            gemm(CblasNoTrans, CblasTrans, 1.0, flatten(tensor_ref, i), flatten(tensor_ref, i), 0.0, S);

            // Find the Singular vectors of the matrix using eigenvalue decomposition
            auto info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', R, S.data(), R, lambda.data());
            if (info) BTAS_EXCEPTION("Error in computing the SVD initial guess");

            // Fill a factor matrix with the singular vectors with the largest corresponding singular
            // values
            lambda = Tensor(R, SVD_rank);
            lambda.fill(0.0);
            auto lower_bound = {0,0};
            auto upper_bound = {R, ((R > SVD_rank) ? SVD_rank : R)};
            auto view = make_view(S.range().slice(lower_bound, upper_bound), S.storage());
            auto l_iter = lambda.begin();
            for(auto iter = view.begin(); iter != view.end(); ++iter, ++l_iter){
              *(l_iter) = *(iter);
            }

            A[i] = lambda;
          }

          //srand(3);
          std::mt19937 generator(random_seed_accessor());
          std::uniform_real_distribution<> distribution(-1.0, 1.0);
          // Fill the remaining columns in the set of factor matrices with dimension < SVD_rank with random numbers
          for(auto& i: modes_w_dim_LT_svd){
            int R = tensor_ref.extent(i);
            auto lower_bound = {0, R};
            auto upper_bound = {R, SVD_rank};
            auto view = make_view(A[i].range().slice(lower_bound, upper_bound), A[i].storage());
            for(auto iter = view.begin(); iter != view.end(); ++iter){
              *(iter) = distribution(generator);
            }
          }

          // Normalize the columns of the factor matrices and
          // set the values al lambda, the weigt of each order 1 tensor
          Tensor lambda(Range{Range1{SVD_rank}});
          A.push_back(lambda);
          for(auto i = 0; i < ndim; ++i){
            normCol(A[i]);
          }

          // Optimize this initial guess.
          ALS(SVD_rank, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
        }
#else  // _HAS_INTEL_MKL
        if (SVD_initial_guess) BTAS_EXCEPTION("Computing the SVD requires LAPACK");
#endif // _HAS_INTEL_MKL
        // This loop keeps track of column dimension
        bool opt_in_for_loop = false;
        for (auto i = (A.empty()) ? 0 : A.at(0).extent(1); i < rank; i += step) {
          opt_in_for_loop = true;
          // This loop walks through the factor matrices
          for (auto j = 0; j < ndim; ++j) {  // select a factor matrix
            // If no factor matrices exists, make a set of factor matrices
            // and fill them with random numbers that are column normalized
            // and create the weighting vector lambda
            if (i == 0) {
              Tensor a;
              if(j < ndimL - 1){
                a = Tensor(Range{tensor_ref_left.range(j+1), Range1{i + 1}});
              }
              else{
                a = Tensor(Range{tensor_ref_right.range(j - ndimL + 2), Range1{i + 1}});
              }
              a.fill(rand());
              A.push_back(a);
              normCol(j);
              if (j  == ndim - 1) {
                Tensor lam(Range{Range1{i + 1}});
                A.push_back(lam);
              }
            }

              // If the factor matrices have memory allocated, rebuild each matrix
              // with new column dimension col_dimension_old + skip
              // fill the new columns with random numbers and normalize the columns
            else {
              int row_extent = A[0].extent(0), rank_old = A[0].extent(1);
              Tensor b(Range{A[0].range(0), Range1{i + 1}});

              {
                auto lower_old = {0, 0}, upper_old = {row_extent, rank_old};
                auto old_view = make_view(b.range().slice(lower_old, upper_old), b.storage());
                auto A_itr = A[0].begin();
                for(auto iter = old_view.begin(); iter != old_view.end(); ++iter, ++A_itr){
                  *(iter) = *(A_itr);
                }
              }

              {
                auto lower_new = {0, rank_old}, upper_new = {row_extent, (int) i+1};
                auto new_view = make_view(b.range().slice(lower_new, upper_new), b.storage());
                std::mt19937 generator(random_seed_accessor());
                std::uniform_real_distribution<> distribution(-1.0, 1.0);
                for(auto iter = new_view.begin(); iter != new_view.end(); ++iter){
                  *(iter) = distribution(generator);
                }
              }

              A.erase(A.begin());
              A.push_back(b);
              if (j + 1 == ndim) {
                b.resize(Range{Range1{i + 1}});
                for (int k = 0; k < A[0].extent(0); k++) b(k) = A[0](k);
                A.erase(A.begin());
                A.push_back(b);
              }
            }
          }
          // compute the ALS of factor matrices with rank = i + 1.
          ALS(i + 1, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
        }
        if(factors_set && ! opt_in_for_loop){
          ALS(rank, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
        }
      }
    }

    /// Create a rank \c rank initial guess using
    /// random numbers from a uniform distribution

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in] converge_test Test to see if ALS is converged.
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
    void build_random(int rank, ConvClass & converge_test, bool direct, int max_als, bool calculate_epsilon, double &epsilon,
                      bool & fast_pI) override{

      std::mt19937 generator(random_seed_accessor());
      std::uniform_real_distribution<> distribution(-1.0, 1.0);
      for(int i = 1; i < ndimL; ++i){
        auto & tensor_ref = tensor_ref_left;
        Tensor a(tensor_ref.extent(i), rank);
        //std::uniform_int_distribution<unsigned int> distribution(0, std::numeric_limits<unsigned int>::max() - 1);
        for(auto iter = a.begin(); iter != a.end(); ++iter){
          *(iter) = distribution(generator);
        }
        this->A.push_back(a);
        this->normCol(i);
      }
      for(int i = 1; i < ndimR; ++i){
        auto & tensor_ref = tensor_ref_right;

        Tensor a(tensor_ref.extent(i), rank);
        //std::uniform_int_distribution<unsigned int> distribution(0, std::numeric_limits<unsigned int>::max() - 1);
        for(auto iter = a.begin(); iter != a.end(); ++iter){
          *(iter) = distribution(generator);
        }
        this->A.push_back(a);
        this->normCol(i);
      }

      Tensor lambda(rank);
      lambda.fill(0.0);
      this->A.push_back(lambda);

      ALS(rank, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
    }

    /// performs the ALS method to minimize the loss function for a single rank
    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in] converge_test Test to see if ALS is converged
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
    void ALS(int rank, ConvClass & converge_test, int max_als, bool calculate_epsilon, double &epsilon, bool & fast_pI) {
      auto count = 0;
      // Until either the initial guess is converged or it runs out of iterations
      // update the factor matrices with or without Khatri-Rao product
      // intermediate
      bool is_converged = false;
      bool matlab = fast_pI;
      Tensor MtKRP(A[ndim - 1].extent(0), rank);
      std::cout << "count\tfit\tchange" << std::endl;
      while(count < max_als && !is_converged){
        count++;
        this->num_ALS++;
        for (auto i = 0; i < ndim; i++) {
          auto tmp = symmetries[i];
          if(tmp == i) {
            direct(i, rank, matlab, converge_test);
          }
          else if(tmp < i){
            A[i] = A[tmp];
          }
          else{
            BTAS_EXCEPTION("Incorrectly defined symmetry");
          }
        }
        is_converged = converge_test(A);
      }

      // Checks loss function if required
      if (calculate_epsilon) {
        if (typeid(converge_test) == typeid(btas::FitCheck<Tensor>)) {
          detail::get_fit(converge_test, epsilon);
          epsilon = 1 - epsilon;
        }
      }
    }

    /// Computes an optimized factor matrix holding all others constant.
    /// No Khatri-Rao product computed, immediate contraction
    /// Does this by first contracting a factor matrix with the refrence tensor
    /// Then computes hadamard/contraction products along all other modes except n.

    /// Want A(I2, R)
    /// T(I1, I2, I3, I4) = B(X, I1, I2) Z(X, I3, I4)
    /// B(X, I1, I2) (Z(X, I3, I4) * A(I4, R)) = B(X, I1, I2) Z'(X, I3, R)
    /// B(X, I1, I2) (Z'(X, I3, R) (*) A(I3, R)) = B(X, I1, I2) Z'(X, R) (contract along I3, Hadamard along R)
    /// B(X, I1, I2) * Z'(X, R) = B'(I1, I2, R)
    /// B'(I1, I2, R) (*) A(I1, R) = B'(I2, R) = A(I2, R) * V(R, R)

    /// \param[in] n The mode being optimized, all other modes held constant
    /// \param[in] rank The current rank, column dimension of the factor matrices
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// return if computing the fast_pI was successful.
    /// \param[in, out] matlab If \c fast_pI = true then try to solve VA = B instead of taking pseudoinverse
    /// in the same manner that matlab would compute the inverse.
    /// return if computing the inverse in this was was successful
    /// \param[in] converge_test test to see if the ALS is converged
    void direct(int n, int rank, bool & matlab, ConvClass & converge_test) {

      // Determine if n is in the left or the right tensor
      bool leftTensor = n < (ndimL - 1);

      // form the intermediate tensor K
      Tensor K(tensor_ref_left.extent(0), rank);
      // works for leftTensor=true
      //
      {
        // want the tensor without n if n is in the left tensor take the right one and vice versa
        auto & tensor_ref = leftTensor ? tensor_ref_right : tensor_ref_left;

        // How many dimension in this side of the tensor
        auto ndimCurr = tensor_ref.rank();
        auto sizeCurr = tensor_ref.size();
        // save range for resize at the end.
        auto R = tensor_ref.range();

        // Start by contracting with the last dimension of tensor without n
        // This is important for picking the correct factor matrix
        // not for picking from tensor_ref
        int contract_dim_inter = leftTensor ? ndim - 1 : ndimL - 2;

        // This is for size of the dimension being contracted
        // picked from tensor_ref
        auto contract_size = tensor_ref.extent(ndimCurr -1);

        // Make the intermediate that will be contracted then hadamard contracted
        // Also resize the tensor for gemm contraction
        Tensor contract_tensor(sizeCurr / contract_size, rank);
        tensor_ref.resize(Range{Range1{sizeCurr / contract_size}, Range1{contract_size}});

        // Contract out the last dimension
        gemm(CblasNoTrans, CblasNoTrans, 1.0, tensor_ref, A[contract_dim_inter], 0.0, contract_tensor);
        // Resize tensor_ref back to original size
        tensor_ref.resize(R);

        // This is the size of the LH dimension of contract_tensor
        sizeCurr /= tensor_ref.extent(ndimCurr - 1);
        // for A now choose the next factor matrix
        --contract_dim_inter;

        // Now want to hadamard contract all the dimension that aren't the connecting dimension
        for(int i = 0; i < ndimCurr - 2; ++i, --contract_dim_inter){
          // The contract_size now starts at the second last dimension
          contract_size = tensor_ref.extent(ndimCurr - 2 - i);
          // Store the LH most dimension size in idx1
          auto idx1 = sizeCurr / contract_size;
          contract_tensor.resize(Range{Range1{idx1},Range1{contract_size},Range1{rank}});
          // After hadamard product middle dimension is gone
          Tensor temp(idx1, rank);
          temp.fill(0.0);

          for(int j = 0; j < idx1; ++j){
            //auto * temp_ptr = temp.data() + j * rank;
            for(int k = 0; k < contract_size; ++k){
              //const auto * contract_ptr = contract_tensor.data() + j * contract_size * rank + k * rank;
              //const auto * A_ptr = A[contract_dim_inter].data() + k * rank;
              for(int r = 0; r < rank; ++r){
                //*(temp_ptr + r) += *(contract_ptr + r) * *(A_ptr + r);
                temp(j,r) += contract_tensor(j,k,r) * A[contract_dim_inter](k,r);
              }
            }
          }
          // After hadamard contract reset contract_tensor with new product
          contract_tensor = temp;
          // Remove the contracted dimension from the current size.
          sizeCurr = idx1;
        }

        // set the hadamard contracted tensor to the intermediate K
        K = contract_tensor;
      }

      // contract K with the other side tensor
      // Tensor_ref now can be the side that contains n
      Tensor & tensor_ref = leftTensor ? tensor_ref_left : tensor_ref_right;
      // Modifying the dimension of tensor_ref so store the range here to resize
      // after contraction.
      Range R = tensor_ref.range();
      // make the new factor matrix for after process
      Tensor an(A[n].range());

      // LH side of tensor after contracting (doesn't include rank or connecting dimension)
      auto LH_size = tensor_ref.size() / tensor_ref.extent(0);
      // Temp holds the intermediate after contracting out the connecting dimension
      // It will be set up to enter hadamard product loop
      Tensor contract_tensor = Tensor(LH_size, rank);
      // resize tensor_ref to remove connecting dimension
      tensor_ref.resize(Range{Range1{tensor_ref.extent(0)},Range1{LH_size}});

      gemm(CblasTrans, CblasNoTrans, 1.0, tensor_ref, K, 0.0, contract_tensor);
      // resize tensor_ref back to original dimensions
      tensor_ref.resize(R);

      std::vector<int> dims(tensor_ref.rank());
      for(int i = 1; i < tensor_ref.rank(); ++i){
        dims[i - 1] = tensor_ref.extent(i);
      }
      dims[dims.size() - 1] = rank;

      // If hadamard loop has to skip a dimension it is stored here.
      auto pseudo_rank = rank;
      // number of dimensions in tensor_ref
      auto ndimCurr = tensor_ref.rank();
      // the dimension that is being hadamard contracted out.
      auto contract_dim = ndimCurr - 2;
      auto nInTensor = leftTensor ? n : n - ndimL + 1;
      auto a_dim = leftTensor ? contract_dim : ndim - 1;
      auto offset = 0;

      // TODO fix the hadamard contraction loop
      // go through hadamard contract on all dimensions excluding rank (will skip one dimension)
      for(int i = 0; i < ndimCurr - 2; ++i, --contract_dim, --a_dim){
        auto contract_size = dims[contract_dim];
        auto idx1 = LH_size / contract_size;
        contract_tensor.resize(Range{Range1{idx1}, Range1{contract_size}, Range1{pseudo_rank}});
        Tensor temp(Range{Range1{idx1}, Range1{pseudo_rank}});

        temp.fill(0.0);

        // If the middle dimension is the mode not being contracted, I will move
        // it to the right hand side temp((size of tensor_ref/product of
        // dimension contracted, rank * mode n dimension)
        if(nInTensor == contract_dim){
          pseudo_rank *= contract_size;
          offset = contract_size;
        }

          // If the code hasn't hit the mode of interest yet, it will contract
          // over the middle dimension and sum over the rank.
        else if (contract_dim > nInTensor){
          for(int j = 0; j < idx1; ++j){
            //auto * temp_ptr = temp.data() + j * pseudo_rank;
            for(int k = 0; k < contract_size; ++k){
              //const auto * contract_ptr = contract_tensor.data() + j * contract_size * pseudo_rank + k * pseudo_rank;
              //const auto * A_ptr = A[a_dim].data() + k * rank;
              for(int r = 0; r < rank; ++r){
                //*(temp_ptr + r) += *(contract_ptr + r) + *(A_ptr + r);
                temp(j,r) += contract_tensor(j,k,r) * A[a_dim](k,r);
              }
            }
          }
          contract_tensor = temp;
        }
          // If the code has passed the mode of interest, it will contract over
          // the middle dimension and sum over rank * mode n dimension
        else{
          for(int j = 0; j < idx1; ++j){
            //auto * temp_ptr = temp.data() + j * pseudo_rank;
            for(int k = 0; k < contract_size; ++k){
              //const auto * contract_ptr = contract_tensor.data() + j * contract_size * pseudo_rank + k * pseudo_rank;
              //const auto * A_ptr = A[a_dim].data() + k * rank;
              for(int l = 0; l < offset; ++l){
                for(int r = 0; r < rank; ++r){
                  //*(temp_ptr + l * rank + r) += *(contract_ptr + l * rank + r) * *(A_ptr + r);
                  temp(j, l*rank + r) += contract_tensor(j,k,l*rank+r) * A[a_dim](k,l*rank+r);
                }
              }
            }
          }

          contract_tensor = temp;
        }
      }

      // If the mode of interest is the 0th mode, then the while loop above
      // contracts over all other dimensions and resulting temp is of the
      // correct dimension If the mode of interest isn't 0th mode, must contract
      // out the 0th mode here, the above algorithm can't perform this
      // contraction because the mode of interest is coupled with the rank
      if(nInTensor != 0){
        auto contract_size = contract_tensor.extent(0);
        Tensor temp(Range{Range1{offset}, Range1{rank}});
        temp.fill(0.0);

        for(int i = 0; i < contract_size; i++){
          //const auto * A_ptr = A[a_dim].data() + i * rank;
          for(int j = 0; j < offset; j++){
            //const auto * contract_ptr = contract_tensor.data() + i * offset * rank + j * rank;
            //auto * temp_ptr = temp.data() + j * rank;
            for(int r = 0; r < rank; r++){
              //*(temp_ptr + r) += *(A_ptr + r) * *(contract_ptr + r);
              temp(j,r) += contract_tensor(i,j,r) * A[a_dim](i,r);
            }
          }
        }

        contract_tensor = temp;
      }

      detail::set_MtKRP(converge_test, contract_tensor);
      // multiply resulting matrix temp by pseudoinverse to calculate optimized
      // factor matrix
      //t1 = std::chrono::high_resolution_clock::now();
#ifdef _HAS_INTEL_MKL
      if(matlab) {
        // This method computes the inverse quickly for a square matrix
        // based on MATLAB's implementation of A / B operator.
        btas::Tensor<int, DEFAULT::range, varray<int> > piv(rank);
        piv.fill(0);

        auto a = generate_V(n, rank);
        int LDB = contract_tensor.extent(0);
        auto info = LAPACKE_dgesv(CblasColMajor, rank, LDB, a.data(), rank, piv.data(), contract_tensor.data(), rank);
        if (info == 0) {
          an = contract_tensor;
        }
        else{
          // If inverse fails resort to the pseudoinverse
          std::cout << "Matlab square inverse failed revert to fast inverse" << std::endl;
          matlab = false;
        }
      }
      if(! matlab){
        bool fast = false;
        gemm(CblasNoTrans, CblasNoTrans, 1.0, contract_tensor, pseudoInverse(n, rank, fast), 0.0, an);
      }
#else
      matlab = false;
      if( !matlab){
        gemm(CblasNoTrans, CblasNoTrans, 1.0, contract_tensor, pseudoInverse(n, rank), 0.0, an);
      }
#endif
      //t2 = std::chrono::high_resolution_clock::now();
      //time = t2 - t1;
      //gemm_wPI += time.count();


      // Normalize the columns of the new factor matrix and update
      normCol(an);
      A[n] = an;
    }

  };

} // namepsace btas

#endif //BTAS_GENERIC_CP_DF_ALS_H
