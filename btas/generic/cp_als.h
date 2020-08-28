//
// Created by Karl Pierce on 7/24/19.
//

#ifndef BTAS_GENERIC_CP_ALS_H
#define BTAS_GENERIC_CP_ALS_H

#include <btas/generic/cp.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

namespace btas{
  /** \brief Computes the Canonical Product (CP) decomposition of an order-N
    tensor using alternating least squares (ALS).

    This computes the CP decomposition of btas::Tensor objects with row
    major storage only with fixed (compile-time) and variable (run-time)
    ranks. Also provides Tucker and randomized Tucker-like compressions coupled
    with CP-ALS decomposition. Does not support strided ranges.

   \warning this code takes a non-const reference \c tensor_ref but does
   not modify the values. This is a result of API (reshape needs non-const tensor)

    Synopsis:
    \code
    // Constructors
    CP_ALS A(tensor)                    // CP_ALS object with empty factor
                                        // matrices and no symmetries
    CP_ALS A(tensor, symms)             // CP_ALS object with empty factor
                                        // matrices and symmetries

    // Operations
    A.compute_rank(rank, converge_test)             // Computes the CP_ALS of tensor to
                                                    // rank, rank build and HOSVD options

    A.compute_rank_random(rank, converge_test)      // Computes the CP_ALS of tensor to
                                                    // rank. Factor matrices built at rank
                                                    // with random numbers

    A.compute_error(converge_test, omega)           // Computes the CP_ALS of tensor to
                                                    // 2-norm
                                                    // error < omega.

    A.compute_geometric(rank, converge_test, step)  // Computes CP_ALS of tensor to
                                                    // rank with
                                                    // geometric steps of step between
                                                    // guesses.

    A.compute_PALS(converge_test)                   // computes CP_ALS of tensor to
                                                    // rank = 3 * max_dim(tensor)
                                                    // in 4 panels using a modified
                                                    // HOSVD initial guess

    A.compress_compute_tucker(tcut_SVD, converge_test) // Computes Tucker decomposition
                                                    // using
                                                    // truncated SVD method then
                                                    // computes finite
                                                    // error CP decomposition on core
                                                    // tensor.

    A.compress_compute_rand(rank, converge_test)    // Computes random decomposition on
                                                    // Tensor to
                                                    // make core tensor with every mode
                                                    // size rank
                                                    // Then computes CP decomposition
                                                    // of core.

   //See documentation for full range of options

    // Accessing Factor Matrices
    A.get_factor_matrices()             // Returns a vector of factor matrices, if
                                        // they have been computed

    A.reconstruct()                     // Returns the tensor computed using the
                                        // CP factor matrices
    \endcode
  */
  template <typename Tensor, class ConvClass = NormCheck<Tensor> >
  class CP_ALS : public CP<Tensor, ConvClass>
          {
  public:
    using CP<Tensor,ConvClass>::A;
    using CP<Tensor,ConvClass>::ndim;
    using CP<Tensor,ConvClass>::symmetries;
    using typename CP<Tensor,ConvClass>::ind_t;
    using typename CP<Tensor,ConvClass>::ord_t;

    /// Create a CP ALS object, child class of the CP object
    /// that stores the reference tensor.
    /// Reference tensor has no symmetries.
    /// \param[in] tensor the reference tensor to be decomposed.
    CP_ALS(Tensor& tensor): CP<Tensor,ConvClass>(tensor.rank()), tensor_ref(tensor),
                            size(tensor.size()){
      for (size_t i = 0; i < ndim; ++i) {
        symmetries.push_back(i);
      }
    }

    /// Create a CP ALS object, child class of the CP object
    /// that stores the reference tensor.
    /// Reference tensor has symmetries.
    /// Symmetries should be set such that the higher modes index
    /// are set equal to lower mode indices (a 4th order tensor,
    /// where the second & third modes are equal would have a
    /// symmetries of {0,1,1,3}
    /// \param[in] tensor the reference tensor to be decomposed.
    /// \param[in] symms the symmetries of the reference tensor.
    CP_ALS(Tensor &tensor, std::vector<size_t> &symms) : CP<Tensor, ConvClass>(tensor.rank()),
                                                         tensor_ref(tensor), size(tensor.size()){
      symmetries = symms;
      if (symmetries.size() > ndim) BTAS_EXCEPTION("Too many symmetries provided")
      for (size_t i = 0; i < ndim; ++i) {
        if (symmetries[i] > i) BTAS_EXCEPTION("Symmetries should always refer to factors at earlier positions");
      }

    }

    ~CP_ALS() = default;

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

    double compute_PALS(std::vector <ConvClass> &converge_list, double RankStep = 0.5, size_t panels = 4,
            int max_als = 20, bool fast_pI = false, bool calculate_epsilon = false,
                        bool direct = true) override {
      if (RankStep <= 0) BTAS_EXCEPTION("Panel step size cannot be less than or equal to zero");
      if (converge_list.size() < panels) BTAS_EXCEPTION(
              "Too few convergence tests.  Must provide a list of panels convergence tests");
      double epsilon = -1.0;
      size_t count = 0;
      // Find the largest rank this will be the first panel
      ind_t max_dim = tensor_ref.extent(0);
      for (size_t i = 1; i < ndim; ++i) {
        ind_t dim = tensor_ref.extent(i);
        max_dim = (dim > max_dim ? dim : max_dim);
      }

      while (count < panels) {
        auto converge_test = converge_list[count];
        // Use tucker initial guess (SVD) to compute the first panel
        if (count == 0) {
          this->build(max_dim, converge_test, direct, max_als, calculate_epsilon, 1, epsilon, true, max_dim, fast_pI);
        }
          // All other panels build the rank buy RankStep variable
        else {
          // Always deal with the first matrix push new factors to the end of A
          // Kick out the first factor when it is replaced.
          // This is the easiest way to resize and preserve the columns
          // (if this is rebuilt with rank as columns this resize would be easier)
          ind_t rank = A[0].extent(1), rank_new = rank + RankStep * max_dim;
          for (int i = 0; i < ndim; ++i) {
            ind_t row_extent = A[0].extent(0), zero = 0;
            Tensor b(Range{Range1{A[0].extent(0)}, Range1{rank_new}});

            // Move the old factor to the new larger matrix
            {
              auto lower_old = {zero, zero}, upper_old = {row_extent, rank};
              auto old_view = make_view(b.range().slice(lower_old, upper_old), b.storage());
              auto A_itr = A[0].begin();
              for(auto iter = old_view.begin(); iter != old_view.end(); ++iter, ++A_itr){
                *(iter) = *(A_itr);
              }
            }

            // Fill in the new columns of the factor with random numbers
            {
              auto lower_new = {zero, rank}, upper_new = {row_extent, rank_new};
              auto new_view = make_view(b.range().slice(lower_new, upper_new), b.storage());
              boost::random::mt19937 generator(random_seed_accessor());
              boost::random::uniform_real_distribution<> distribution(-1.0, 1.0);
              for (auto iter = new_view.begin(); iter != new_view.end(); ++iter) {
                *(iter) = distribution(generator);
              }
            }

            A.erase(A.begin());
            A.push_back(b);
            // replace the lambda matrix when done with all the factors
            if (i + 1 == ndim) {
              b.resize(Range{Range1{rank_new}});
              for (ind_t k = 0; k < A[0].extent(0); k++) b(k) = A[0](k);
              A.erase(A.begin());
              A.push_back(b);
            }
            // normalize the factor (don't replace the previous lambda matrix)
            this->normCol(0);
          }
          ALS(rank_new, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI);
        }
        count++;
      }
      return epsilon;
    }

    /// \brief Computes an approximate core tensor using
    /// Tucker decomposition, e.g.
    ///  \f$ T(I_1 \dots I_N) \approx T(R_1 \dots R_N) U^{(1)} (R_1, I_1) \dots U^{(N)} (R_N, I_N) \f$
    /// where \f$ \mathrm{rank} R_1 \leq \mathrm{rank } I_1 \f$ , etc.
    /// Reference: <a href="http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7516088">
    /// here</a>. Using this approximation the CP decomposition is
    /// computed to either finite error or finite rank. Default settings
    /// calculate to finite error. Factor matrices from get_factor_matrices() are
    /// scaled by the Tucker transformations.

    /// \param[in] tcutSVD Truncation threshold for SVD of each mode in Tucker
    /// decomposition.
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit.
    /// \param[in] rank If finding CP
    /// decomposition to finite rank, define CP rank. Default 0 will throw error
    /// for compute_rank.
    /// \param[in] direct The CP decomposition be computed
    /// without calculating the Khatri-Rao product? Default = true.
    /// \param[in]
    /// calculate_epsilon Should the 2-norm error be calculated \f$ ||T_{exact} -
    /// T_{approx}|| = \epsilon \f$ . Default = false.
    /// \param[in] max_als If CP decomposition is to finite
    /// error, max_als is the highest rank approximation computed before giving up
    /// on CP-ALS. Default = 1e4.
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// default = false
    /// \returns 2-norm error, \f$ \epsilon \f$,
    /// between exact and approximate tensor, -1.0 if calculate_epsilon = false &&
    ///  ConvClass != FitCheck.
    double compress_compute_tucker(double tcutSVD, ConvClass &converge_test, ind_t rank = 0,
                                   bool direct = true, bool calculate_epsilon = false,
                                   ind_t max_als = 1e4, bool fast_pI = false) {
      // Tensor compression
      std::vector<Tensor> transforms;
      tucker_compression(tensor_ref, tcutSVD, transforms);
      size = tensor_ref.size();
      double epsilon = -1.0;

      // CP decomposition
      epsilon = this->compute_rank_random(rank, converge_test, max_als, fast_pI, calculate_epsilon, direct);

      // scale factor matrices
      for (size_t i = 0; i < ndim; i++) {
        Tensor tt(transforms[i].extent(0), A[i].extent(1));
        gemm(CblasNoTrans, CblasNoTrans, 1.0, transforms[i], A[i], 0.0, tt);
        A[i] = tt;
      }

      return epsilon;
    }

    /// \brief Computes an approximate core tensor using
    /// random projection, i.e.
    /// \f$ T(I_1 \dots I_N) \approx T(R_1 \dots R_N) U^{(1)} (R_1, I_1) \dots U^{(N)} (R_N, I_N) \f$
    /// where \f$ \mathrm{rank } R_1 \leq \mathrm{rank } I_1 \f$ , etc.

    /// Reference: <a href="https://arxiv.org/pdf/1703.09074.pdf">arXiv:1703.09074</a>
    /// Using this approximation the CP decomposition is computed to
    /// either finite error or finite rank.
    /// Default settings calculate to finite error.
    /// Factor matrices are scaled by randomized transformation.

    /// \param[in] desired_compression_rank The new dimension of each mode after
    /// randomized compression.
    /// /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit..
    /// \param[in] oversampl Oversampling added to the
    /// desired_compression_rank required to provide a more optimal decomposition.
    /// Default = suggested = 10.
    /// \param[in] powerit Number of power iterations,
    /// as specified in the literature, to scale the spectrum of each mode.
    /// Default = suggested = 2.
    /// \param[in] rank If finding CP
    /// decomposition to finite rank, define CP rank. Default 0 will throw error
    /// for compute_rank.
    /// \param[in] direct Should the CP decomposition be
    /// computed without calculating the Khatri-Rao product? Default = true.
    /// \param[in] calculate_epsilon Should the 2-norm error be calculated
    /// \f$ ||T_exact - T_approx|| = \epsilon \f$. Default = false.
    /// \param[in] max_als If CP decomposition is to
    /// finite error, max_als is the highest rank approximation computed before
    /// giving up on CP-ALS. Default = 1e5.
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// default = false
    /// \returns 2-norm error, \f$ \epsilon \f$,
    /// between exact and approximate tensor, -1.0 if calculate_epsilon = false &&
    ///  ConvClass != FitCheck.
    double
    compress_compute_rand(ind_t desired_compression_rank, ConvClass &converge_test, long oversampl = 10,
                          size_t powerit = 2,
                          ind_t rank = 0, bool direct = true, bool calculate_epsilon = false,
                          ind_t max_als = 1e5, bool fast_pI = false) {
      std::vector<Tensor> transforms;
      randomized_decomposition(tensor_ref, transforms, desired_compression_rank, oversampl, powerit);
      size = tensor_ref.size();

      auto epsilon = this->compute_rank_random(rank, converge_test, max_als, fast_pI, calculate_epsilon, direct);

      // scale factor matrices
      for (size_t i = 0; i < ndim; i++) {
        Tensor tt(transforms[i].extent(0), A[i].extent(1));
        gemm(CblasNoTrans, CblasNoTrans, 1.0, transforms[i], A[i], 0.0, tt);
        A[i] = tt;
      }

      return epsilon;
    }


  protected:
    Tensor &tensor_ref;         // Tensor to be decomposed
    ord_t size;                   // Total number of elements
    bool factors_set = false;   // Are the factors preset (not implemented yet).

    /// Creates an initial guess by computing the SVD of each mode
    /// If the rank of the mode is smaller than the CP rank requested
    /// The rest of the factor matrix is filled with random numbers
    /// Builds factor matricies starting with R=(1 or SVD_rank)
    /// and moves to R = \c rank
    /// incrementing column dimension, R, by \c step

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit..
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

    void build(ind_t rank, ConvClass &converge_test, bool direct, ind_t max_als, bool calculate_epsilon,
               ind_t step, double &epsilon,
               bool SVD_initial_guess, ind_t SVD_rank, bool &fast_pI) override {
      // If its the first time into build and SVD_initial_guess
      // build and optimize the initial guess based on the left
      // singular vectors of the reference tensor.
      if (A.empty() && SVD_initial_guess) {
        if (SVD_rank == 0) BTAS_EXCEPTION("Must specify the rank of the initial approximation using SVD");

        std::vector<size_t> modes_w_dim_LT_svd;
        A = std::vector<Tensor>(ndim);

        // Determine which factor matrices one can fill using SVD initial guess
        // Don't do the modes that are symmetric to other modes
        for (size_t i = 0; i < ndim; i++) {
          auto tmp = symmetries[i];
          if (tmp != i) continue;
          if (tensor_ref.extent(i) < SVD_rank) {
            modes_w_dim_LT_svd.push_back(i);
          }
        }

        // Fill all factor matrices with their singular vectors,
        // because we contract X X^T (where X is reference tensor) to make finding
        // singular vectors an eigenvalue problem some factor matrices will not be
        // full rank;
        A[0] = Tensor(tensor_ref.extent(0), SVD_rank);
        A[0].fill(0.0);

        for (size_t i = 1; i < ndim; i++) {
          // If a mode is symmetric to another mode skip this whole process
          // Will set the modes equal at the end
          auto tmp = symmetries[i];
          if (tmp != i) continue;
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

          A[i] = lambda;
        }

        // srand(3);
        boost::random::mt19937 generator(random_seed_accessor());
        boost::random::uniform_real_distribution<> distribution(-1.0, 1.0);
        // Fill the remaining columns in the set of factor matrices with dimension < SVD_rank with random numbers
        for (auto &i : modes_w_dim_LT_svd) {
          ind_t R = tensor_ref.extent(i), zero = 0;
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
        for (size_t i = 1; i < ndim; ++i) {
          // normalize the columns of matrices that were set
          // i.e. not symmetric to another mode.
          auto tmp = symmetries[i];
          if (tmp == i)
            this->normCol(A[i]);
          // Then make sure the summetric modes are set here
          A[i] = A[tmp];
        }

        // Optimize this initial guess.
        ALS(SVD_rank, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI);
      }
      // This loop keeps track of column dimension
      bool opt_in_for_loop = false;
      for (ind_t i = (A.empty()) ? 0 : A.at(0).extent(1); i < rank; i += step) {
        opt_in_for_loop = true;
        // This loop walks through the factor matrices
        ind_t rank_new = i + 1;
        for (size_t j = 0; j < ndim; ++j) {  // select a factor matrix
          // If no factor matrices exists, make a set of factor matrices
          // and fill them with random numbers that are column normalized
          // and create the weighting vector lambda
          if (i == 0) {
            Tensor a(Range{tensor_ref.range().range(j), Range1{rank_new}});
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
            if(j == ndim - 1){
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
      if(factors_set && ! opt_in_for_loop){
        ALS(rank, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI);
      }
    }

    /// Create a rank \c rank initial guess using
    /// random numbers from a uniform distribution

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit..
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
      boost::random::mt19937 generator(random_seed_accessor());
      boost::random::uniform_real_distribution<> distribution(-1.0, 1.0);
      for (size_t i = 0; i < this->ndim; ++i) {
        // If this mode is symmetric to a previous mode, set it equal to
        // previous mode, else make a random matrix.
        auto tmp = symmetries[i];
        if (tmp != i) {
          A.push_back(A[tmp]);
        } else {
          Tensor a(tensor_ref.extent(i), rank);
          for (auto iter = a.begin(); iter != a.end(); ++iter) {
            *(iter) = distribution(generator);
          }
          this->A.push_back(a);
          this->normCol(i);
        }
      }
      Tensor lambda(rank);
      lambda.fill(0.0);
      this->A.push_back(lambda);

      ALS(rank, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI);
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

    void ALS(ind_t rank, ConvClass &converge_test, bool dir, int max_als, bool calculate_epsilon,
             double &epsilon, bool &fast_pI) {

      size_t count = 0;

      // Until either the initial guess is converged or it runs out of iterations
      // update the factor matrices with or without Khatri-Rao product
      // intermediate
      bool is_converged = false;
      bool matlab = fast_pI;
      //std::cout << "count\tfit\tFit Change" << std::endl;
      while (count < max_als && !is_converged) {
        count++;
        this->num_ALS++;
        for (size_t i = 0; i < ndim; i++) {
          auto tmp = symmetries[i];
          if (tmp != i) {
            A[i] = A[tmp];
          } else if (dir) {
            direct(i, rank, fast_pI, matlab, converge_test);
          } else {
            update_w_KRP(i, rank, fast_pI, matlab, converge_test);
          }
        }
        //std::cout << count << "\t";
        is_converged = converge_test(A);
        //T *= 0.6;
      }

      // Checks loss function if required
      if (calculate_epsilon) {
        if (typeid(converge_test) == typeid(btas::FitCheck<Tensor>)) {
          detail::get_fit(converge_test, epsilon);
          epsilon = 1 - epsilon;
        } else{
          epsilon = this->norm(this->reconstruct() - tensor_ref);
        }
      }
    }

    /// Calculates an optimized CP factor matrix using Khatri-Rao product
    /// intermediate
    /// \param[in] n The mode being optimized, all other modes held
    /// constant
    /// \param[in] rank The current rank, column dimension of the factor
    /// matrices
    /// iteration factor matrix
    /// \param[in, out] matlab If \c fast_pI = true then try to solve VA = B instead of taking pseudoinverse
    /// in the same manner that matlab would compute the inverse.
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit. test to see if the ALS is converged
    void update_w_KRP(size_t n, ind_t rank, bool &fast_pI,
                      bool &matlab, ConvClass &converge_test) {
      Tensor temp(A[n].extent(0), rank);
      Tensor an(A[n].range());

#ifdef BTAS_HAS_INTEL_MKL

      // Computes the Khatri-Rao product intermediate
      auto KhatriRao = this->generate_KRP(n, rank, true);

      // moves mode n of the reference tensor to the front to simplify contraction
      swap_to_first(tensor_ref, n);
      std::vector<ind_t> tref_indices, KRP_dims, An_indices;

      // resize the Khatri-Rao product to the proper dimensions
      for (size_t i = 1; i < ndim; i++) {
        KRP_dims.push_back(tensor_ref.extent(i));
      }
      KRP_dims.push_back(rank);
      KhatriRao.resize(KRP_dims);
      KRP_dims.clear();

      // build contraction indices to contract over correct modes
      An_indices.push_back(0);
      An_indices.push_back(ndim);
      tref_indices.push_back(0);
      for (size_t i = 1; i < ndim; i++) {
        tref_indices.push_back(i);
        KRP_dims.push_back(i);
      }
      KRP_dims.push_back(ndim);

      contract(1.0, tensor_ref, tref_indices, KhatriRao, KRP_dims, 0.0, temp, An_indices);

      // move the nth mode of the reference tensor back where it belongs
      swap_to_first(tensor_ref, n, true);

#else  // BTAS_HAS_CBLAS
      // without MKL program cannot perform the swapping algorithm, must compute
      // flattened intermediate
      gemm(CblasNoTrans, CblasNoTrans, 1.0, flatten(tensor_ref, n), this->generate_KRP(n, rank, true), 0.0, temp);
#endif

      detail::set_MtKRP(converge_test, temp);

      // contract the product from above with the pseudoinverse of the Hadamard
      // produce an optimize factor matrix
      this->pseudoinverse_helper(n, fast_pI, matlab, temp);

      // compute the difference between this new factor matrix and the previous
      // iteration
      this->normCol(temp);

      // Replace the old factor matrix with the new optimized result
      A[n] = temp;
    }


    /// Computes an optimized factor matrix holding all others constant.
    /// No Khatri-Rao product computed, immediate contraction
    // Does this by first contracting a factor matrix with the refrence tensor
    // Then computes hadamard/contraction products along all other modes except n.

    // Want A(I2, R)
    // T(I1, I2, I3, I4)
    // T(I1, I2, I3, I4) * A(I4, R) = T'(I1, I2, I3, R)
    // T'(I1, I2, I3, R) (*) A(I3, R) = T'(I1, I2, R) (contract along I3, Hadamard along R)
    // T'(I1, I2, R) (*) A(I1, R) = T'(I2, R) = A(I2, R) * V(R, R)

    /// \param[in] n The mode being optimized, all other modes held constant
    /// \param[in] rank The current rank, column dimension of the factor matrices
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// return if computing the fast_pI was successful.
    /// \param[in, out] matlab If \c fast_pI = true then try to solve VA = B instead of taking pseudoinverse
    /// in the same manner that matlab would compute the inverse.
    /// return if computing the inverse in this was was successful
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit. test to see if the ALS is converged

    void direct(size_t n, ind_t rank, bool &fast_pI, bool &matlab, ConvClass &converge_test) {

      // Determine if n is the last mode, if it is first contract with first mode
      // and transpose the product
      bool last_dim = n == ndim - 1;
      // product of all dimensions
      ord_t LH_size = size;
      size_t contract_dim = last_dim ? 0 : ndim - 1;
      ind_t offset_dim = tensor_ref.extent(n);
      ind_t pseudo_rank = rank;

      // Store the dimensions which are available to hadamard contract
      std::vector<ind_t> dimensions;
      for (size_t i = last_dim ? 1 : 0; i < (last_dim ? ndim : ndim - 1); i++) {
        dimensions.push_back(tensor_ref.extent(i));
      }

      // Modifying the dimension of tensor_ref so store the range here to resize
      Range R = tensor_ref.range();
      Tensor an(A[n].range());

      // Resize the tensor which will store the product of tensor_ref and the first factor matrix
      Tensor temp = Tensor(size / tensor_ref.extent(contract_dim), rank);
      tensor_ref.resize(Range{
              Range1{last_dim ? tensor_ref.extent(contract_dim) : size / tensor_ref.extent(contract_dim)},
              Range1{last_dim ? size / tensor_ref.extent(contract_dim) : tensor_ref.extent(contract_dim)}});

      // contract tensor ref and the first factor matrix
      gemm((last_dim ? CblasTrans : CblasNoTrans), CblasNoTrans, 1.0, tensor_ref, A[contract_dim], 0.0, temp);

      // Resize tensor_ref
      tensor_ref.resize(R);
      // Remove the dimension which was just contracted out
      LH_size /= tensor_ref.extent(contract_dim);

      // n tells which dimension not to contract, and contract_dim says which dimension I am trying to contract.
      // If n == contract_dim then that mode is skipped.
      // if n == ndim - 1, my contract_dim = 0. The gemm transposes to make rank = ndim - 1, so I
      // move the pointer that preserves the last dimension to n = ndim -2.
      // In all cases I want to walk through the orders in tensor_ref backward so contract_dim = ndim - 2
      n = last_dim ? ndim - 2: n;
      contract_dim = ndim - 2;

      while (contract_dim > 0) {
        // Now temp is three index object where temp has size
        // (size of tensor_ref/product of dimension contracted, dimension to be
        // contracted, rank)
        temp.resize(Range{Range1{LH_size / dimensions[contract_dim]}, Range1{dimensions[contract_dim]},
                          Range1{pseudo_rank}});
        Tensor contract_tensor(Range{Range1{temp.extent(0)}, Range1{temp.extent(2)}});
        contract_tensor.fill(0.0);
        const auto &a = A[(last_dim ? contract_dim + 1 : contract_dim)];
        // If the middle dimension is the mode not being contracted, I will move
        // it to the right hand side temp((size of tensor_ref/product of
        // dimension contracted, rank * mode n dimension)
        if (n == contract_dim) {
          pseudo_rank *= offset_dim;
        }

        // If the code hasn't hit the mode of interest yet, it will contract
        // over the middle dimension and sum over the rank.

        else if (contract_dim > n) {
          ind_t idx1 = temp.extent(0), idx2 = temp.extent(1);
          ord_t i_times_rank = 0, i_times_rank_idx2 = 0;
          for (ind_t i = 0; i < idx1; i++, i_times_rank+=rank) {
            auto *contract_ptr = contract_tensor.data() + i_times_rank;
            ord_t j_times_rank= 0;
            for (ind_t j = 0; j < idx2; j++, j_times_rank+=rank) {
              const auto *temp_ptr = temp.data() + i_times_rank_idx2 + j_times_rank;
              const auto *A_ptr = a.data() + j_times_rank;
              for (ind_t r = 0; r < rank; r++) {
                *(contract_ptr + r) += *(temp_ptr + r) * *(A_ptr + r);
              }
            }
            i_times_rank_idx2 += j_times_rank;
          }
          temp = contract_tensor;
        }

          // If the code has passed the mode of interest, it will contract over
          // the middle dimension and sum over rank * mode n dimension
        else {
          ind_t idx1 = temp.extent(0), idx2 = temp.extent(1), offset = offset_dim;
          ord_t i_times_rank = 0, i_times_rank_idx2 = 0;
          for (ind_t i = 0; i < idx1; i++, i_times_rank+=pseudo_rank) {
            auto *contract_ptr = contract_tensor.data() + i_times_rank;
            ord_t j_times_prank = 0, j_times_rank = 0;
            for (ind_t j = 0; j < idx2; j++, j_times_prank+=pseudo_rank, j_times_rank+=rank) {
              const auto *temp_ptr = temp.data() + i_times_rank_idx2 + j_times_prank;
              const auto *A_ptr = a.data() + j_times_rank;
              ord_t k_times_rank = 0;
              for (ind_t k = 0; k < offset; k++, k_times_rank+=rank) {
                for (ind_t r = 0; r < rank; r++) {
                  *(contract_ptr + k_times_rank + r) += *(temp_ptr + k_times_rank + r) * *(A_ptr + r);
                }
              }
            }
            i_times_rank_idx2 += j_times_prank;
          }
          temp = contract_tensor;
        }

        LH_size /= tensor_ref.extent(contract_dim);
        contract_dim--;
      }

      // If the mode of interest is the 0th mode, then the while loop above
      // contracts over all other dimensions and resulting temp is of the
      // correct dimension If the mode of interest isn't 0th mode, must contract
      // out the 0th mode here, the above algorithm can't perform this
      // contraction because the mode of interest is coupled with the rank
      if (n != 0) {
        temp.resize(Range{Range1{dimensions[0]}, Range1{dimensions[n]}, Range1{rank}});
        Tensor contract_tensor(Range{Range1{temp.extent(1)}, Range1{rank}});
        contract_tensor.fill(0.0);

        ind_t idx1 = temp.extent(0), idx2 = temp.extent(1);
        const auto &a = A[(last_dim ? 1 : 0)];
        ord_t i_times_rank = 0, i_times_rank_idx2 = 0;
        for (ind_t i = 0; i < idx1; i++, i_times_rank += rank) {
          const auto *A_ptr = a.data() + i_times_rank;
          ord_t j_times_rank = 0;
          for (ind_t j = 0; j < idx2; j++, j_times_rank+=rank) {
            const auto *temp_ptr = temp.data() + i_times_rank_idx2 + j_times_rank;
            auto *contract_ptr = contract_tensor.data() + j * rank;
            for (ind_t r = 0; r < rank; r++) {
              *(contract_ptr + r) += *(A_ptr + r) * *(temp_ptr + r);
            }
          }
          i_times_rank_idx2 += j_times_rank;
        }
        temp = contract_tensor;
      }

      n = last_dim ? ndim - 1: n;
      // multiply resulting matrix temp by pseudoinverse to calculate optimized
      // factor matrix
      detail::set_MtKRP(converge_test, temp);
      // Temp is then rewritten with unnormalized new A[n] matrix
      this->pseudoinverse_helper(n, fast_pI, matlab, temp);

      // Normalize the columns of the new factor matrix and update
      this->normCol(temp);
      A[n] = temp;
    }

  };

} //namespace btas

#endif //BTAS_GENERIC_CP_ALS_H
