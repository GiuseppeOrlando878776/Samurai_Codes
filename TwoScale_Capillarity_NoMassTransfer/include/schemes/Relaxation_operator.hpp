// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include "flux_base.hpp"

#define DEBUG_RELAXATION

namespace samurai {
  using namespace EquationData;

  /**
   * Implementation of a relaxation operator
   */
  template<class Field>
  class RelaxationOperator {
  public:
    using Number = typename Field::value_type; // Define the shortcut for the arithmetic type

    using cfg = samurai::LocalCellSchemeConfig<SchemeType::NonLinear, Field, Field>;

    /**
     * Class constructor
     * @param EOS_phase1_ phase 1 equation of state
     * @param EOS_phase2_ phase 2 equation of state
     * @param sigma_ surface tension coefficient
     * @param lambda_ bound-preserving parameter
     * @param atol_Newton_ absolute tolerance for dual-time stepping
     * @param rtol_Newton_ relative tolerance for dual-time stepping
     * @param max_Newton_iters_ maximum number of iterations for dual-time stepping
     */
    RelaxationOperator(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                       const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                       const Number sigma_,
                       const Number lambda_ = static_cast<Number>(0.9),
                       const Number atol_Newton_ = static_cast<Number>(1e-14),
                       const Number rtol_Newton_ = static_cast<Number>(1e-12),
                       const std::size_t max_Newton_iters_ = 60);

    /**
     * Perform a Newton step relaxation for a state vector
     * @param H curvature
     * @param dalpha1 variation of volume fraction
     * @param alpha1 volume fraction
     * @param to_be_relaxed auxiliary flag to mark if a field has still to be relaxed or not
     * @param Newton_iterations number of Newton (dual-time stepping) iterations
     * @param relaxation_applied flag to check whether relaxation has been applied or we converged
     */
    template<class Field_Scalar, class Field_Scalar_Unsigned>
    auto make_Newton_step_relaxation(const Field_Scalar& H,
                                     Field_Scalar& dalpha1,
                                     Field_Scalar& alpha1,
                                     Field_Scalar_Unsigned& to_be_relaxed,
                                     Field_Scalar_Unsigned& Newton_iterations);

    /**
     * Set the value of the flag to check whether relaxation has been applied
     * @param global_relaxation_applied flag to check whether relaxation has been applied
     */
    inline void set_relaxation_applied(const bool global_relaxation_applied);

    /**
     * Get the value of the flag to check whether relaxation has been applied
     * @return global_relaxation_applied flag to check whether relaxation has been applied
     */
    inline bool get_relaxation_applied() const;

  protected:
    const LinearizedBarotropicEOS<Number>& EOS_phase1;
    const LinearizedBarotropicEOS<Number>& EOS_phase2;

    const Number sigma; /*!< Surface tension coefficient */

    const Number      lambda;           /*!< Parameter for bound preserving strategy */
    const Number      atol_Newton;      /*!< Absolute tolerance Newton method relaxation */
    const Number      rtol_Newton;      /*!< Relative tolerance Newton method relaxation */
    const std::size_t max_Newton_iters; /*!< Maximum number of Newton iterations */

  private:
    bool relaxation_applied; /*!< Auxiliary flag to check whether relaxation has been applied */
  };

  // Constructor with all relevant parameters
  //
  template<class Field>
  RelaxationOperator<Field>::RelaxationOperator(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                                                const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                                                const Number sigma_,
                                                const Number lambda_,
                                                const Number atol_Newton_,
                                                const Number rtol_Newton_,
                                                const std::size_t max_Newton_iters_):
    EOS_phase1(EOS_phase1_), EOS_phase2(EOS_phase2_), sigma(sigma_),
    lambda(lambda_), atol_Newton(atol_Newton_), rtol_Newton(rtol_Newton_),
    max_Newton_iters(max_Newton_iters_) {}

  // Set the value of the flag to check whether relaxation has been applied
  //
  template<class Field>
  void RelaxationOperator<Field>::set_relaxation_applied(const bool global_relaxation_applied) {
    relaxation_applied = global_relaxation_applied;
  }

  // Get the value of the flag to check whether relaxation has been applied
  //
  template<class Field>
  bool RelaxationOperator<Field>::get_relaxation_applied() const {
    return relaxation_applied;
  }

  // Implement the contribution of the discrete relaxation operator
  //
  template<class Field>
  template<class Field_Scalar, class Field_Scalar_Unsigned>
  auto RelaxationOperator<Field>::make_Newton_step_relaxation(const Field_Scalar& H,
                                                              Field_Scalar& dalpha1,
                                                              Field_Scalar& alpha1,
                                                              Field_Scalar_Unsigned& to_be_relaxed,
                                                              Field_Scalar_Unsigned& Newton_iterations) {
    auto relaxation_step = samurai::make_cell_based_scheme<typename RelaxationOperator::cfg>();
    relaxation_step.set_name("Relaxation");
    relaxation_step.set_scheme_function([&](samurai::SchemeValue<cfg>& result, const auto& cell, const auto& field)
                                           {
                                             const auto local_field = field[cell];
                                             result = field[cell];

                                             to_be_relaxed[cell] = 0;

                                             if(!std::isnan(H[cell])) {
                                               // Pre-fetch some variables used multiple times in order to exploit possible vectorization
                                               const auto m1_loc = local_field(M1_INDEX);
                                               const auto m2_loc = local_field(M2_INDEX);
                                               auto alpha1_loc   = alpha1[cell];
                                               auto alpha2_loc   = static_cast<Number>(1.0) - alpha1_loc;

                                               // Update auxiliary values affected by the nonlinear function for which we seek a zero
                                               const auto rho1_loc = m1_loc/alpha1_loc; // TODO: Add a check in case of zero volume fraction
                                               const auto p1_loc   = EOS_phase1.pres_value(rho1_loc);

                                               const auto rho2_loc = m2_loc/alpha2_loc; // TODO: Add a check in case of zero volume fraction
                                               const auto p2_loc   = EOS_phase2.pres_value(rho2_loc);

                                               // Compute the nonlinear function for which we seek the zero (basically the Laplace law)
                                               const auto F = p1_loc - p2_loc - sigma*H[cell];

                                               // Perform the relaxation only where really needed
                                               if(std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase1.get_p0(), sigma*std::abs(H[cell])) &&
                                                  std::abs(dalpha1[cell]) > atol_Newton) {
                                                 to_be_relaxed[cell] = 1;
                                                 Newton_iterations[cell]++;
                                                 relaxation_applied = true;

                                                 // Compute the derivative w.r.t large-scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
                                                 const auto c1_loc = EOS_phase1.c_value(rho1_loc);
                                                 const auto c2_loc = EOS_phase2.c_value(rho2_loc);

                                                 const auto dF_dalpha1 = -m1_loc/(alpha1_loc*alpha1_loc)*
                                                                          c1_loc*c1_loc
                                                                         -m2_loc/(alpha2_loc*alpha2_loc)*
                                                                          c2_loc*c2_loc;

                                                 // Compute the large-scale volume fraction update
                                                 auto dalpha1_loc = F/dF_dalpha1;
                                                 if(dalpha1_loc > static_cast<Number>(0.0)) {
                                                   dalpha1_loc = std::min(dalpha1_loc, lambda*alpha2_loc);
                                                 }
                                                 else if(dalpha1_loc < static_cast<Number>(0.0)) {
                                                   dalpha1_loc = std::max(dalpha1_loc, -lambda*alpha1_loc);
                                                 }
                                                 dalpha1[cell] = dalpha1_loc;

                                                 #ifdef DEBUG_RELAXATION
                                                   if(alpha1_loc + dalpha1_loc < static_cast<Number>(0.0) ||
                                                      alpha1_loc + dalpha1_loc > static_cast<Number>(1.0)) {
                                                        // I should never get here. Added only for the sake of safety!!
                                                        throw std::runtime_error("Bounds exceeding value for large-scale volume fraction inside Newton step ");
                                                   }
                                                 #endif
                                                 alpha1_loc += dalpha1_loc;
                                                 alpha1[cell] = alpha1_loc;
                                               }

                                               // Update the vector of conserved variables
                                               // (probably not the optimal choice since I need this update only at the end of the Newton loop)
                                               result(RHO_ALPHA1_INDEX) = (m1_loc + m2_loc)*alpha1_loc;
                                             }
                                           });

    return relaxation_step;
  }

} // end of namespace
