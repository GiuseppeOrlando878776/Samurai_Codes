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
     * @param Hmax_ threshold length scale
     * @param kappa_ parameter related to the radius of small-scale droplets
     * @param alpha1d_max_ maximum threshold of small-scale volume fraction
     * @param alpha1_bar_min_ minimum effective volume fraction to identify the mixture region
     * @param alpha1_bar_max_ maximum effective volume fraction to identify the mixture region
     * @param lambda_ bound-preserving parameter
     * @param atol_Newton_ absolute tolerance for dual-time stepping
     * @param rtol_Newton_ relative tolerance for dual-time stepping
     * @param max_Newton_iters_ maximum number of iterations for dual-time stepping
     * @param mass_transfer_NR_ flag to check whether mass transfer inside relaxation is desired
     */
    RelaxationOperator(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                       const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                       const Number sigma_,
                       const Number Hmax_,
                       const Number kappa_,
                       const Number alpha1d_max_,
                       const Number alpha1_bar_min_,
                       const Number alpha1_bar_max_,
                       const Number lambda_ = static_cast<Number>(0.9),
                       const Number atol_Newton_ = static_cast<Number>(1e-14),
                       const Number rtol_Newton_ = static_cast<Number>(1e-12),
                       const std::size_t max_Newton_iters_ = 60,
                       const bool mass_transfer_NR_ = true);

    /**
     * Perform a Newton step relaxation for a state vector
     * @param H_bar effective curvature
     * @param dalpha1_bar variation of effective volume fraction
     * @param alpha1_bar effective volume fraction
     * @param to_be_relaxed auxiliary flag to mark if a field has still to be relaxed or not
     * @param Newton_iterations number of Newton (dual-time stepping) iterations
     * @param grad_alpha1_bar gradient of effective volume fraction
     * @param type_relaxation check if relaxation toward local Laplace law or toward pressure equilibrium
     */
    template<class Field_Scalar, class Field_Scalar_Unsigned, class Field_Vect>
    auto make_Newton_step_relaxation(const Field_Scalar& H_bar,
                                     Field_Scalar& dalpha1_bar,
                                     Field_Scalar& alpha1_bar,
                                     Field_Scalar_Unsigned& to_be_relaxed,
                                     Field_Scalar_Unsigned& Newton_iterations,
                                     Field_Vect& grad_alpha1_bar,
                                     Field_Scalar_Unsigned& type_relaxation);
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

    /**
     * Set the value of the flag to check whether mass transfer inside relaxation has to be done or not
     * @param mass_transfer_NR_ flag to check whether mass trasnfer is desired inside relaxation
     */
    inline void set_mass_transfer_NR(const bool mass_transfer_NR_);

  protected:
    const LinearizedBarotropicEOS<Number>& EOS_phase1;
    const LinearizedBarotropicEOS<Number>& EOS_phase2;

    const Number sigma; /*!< Surface tension coefficient */

    const Number Hmax;        /*!< Threshold length scale */
    const Number kappa;       /*!< Parameter related to the radius of small-scale droplets */
    const Number alpha1d_max; /*!< Maximum threshold of small-scale volume fraction */
    const Number alpha1_bar_min; /*!< Minimum effective volume fraction to identify the mixture region */
    const Number alpha1_bar_max; /*!< Maximum effective volume fraction to identify the mixture region */

    const Number      lambda;           /*!< Parameter for bound preserving strategy */
    const Number      atol_Newton;      /*!< Absolute tolerance Newton method relaxation */
    const Number      rtol_Newton;      /*!< Relative tolerance Newton method relaxation */
    const std::size_t max_Newton_iters; /*!< Maximum number of Newton iterations */

  private:
    bool mass_transfer_NR; /*!< Auxiliary flag to check whether mass transfer inside relaxation is desired */

    bool relaxation_applied; /*!< Auxiliary flag to check whether relaxation has been applied */
  };

  // Constructor with all relevant parameters
  //
  template<class Field>
  RelaxationOperator<Field>::RelaxationOperator(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                                                const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                                                const Number sigma_,
                                                const Number Hmax_,
                                                const Number kappa_,
                                                const Number alpha1d_max_,
                                                const Number alpha1_bar_min_,
                                                const Number alpha1_bar_max_,
                                                const Number lambda_,
                                                const Number atol_Newton_,
                                                const Number rtol_Newton_,
                                                const std::size_t max_Newton_iters_,
                                                const bool mass_transfer_NR_):
    EOS_phase1(EOS_phase1_), EOS_phase2(EOS_phase2_), sigma(sigma_),
    Hmax(Hmax_), kappa(kappa_),
    alpha1d_max(alpha1d_max_), alpha1_bar_min(alpha1_bar_min_), alpha1_bar_max(alpha1_bar_max_),
    lambda(lambda_), atol_Newton(atol_Newton_), rtol_Newton(rtol_Newton_),
    max_Newton_iters(max_Newton_iters_), mass_transfer_NR(mass_transfer_NR_) {}

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

  // Set the value of the flag to check whether mass transfer inside relaxation has to be done or not
  //
  template<class Field>
  void RelaxationOperator<Field>::set_mass_transfer_NR(const bool mass_transfer_NR_) {
    mass_transfer_NR = mass_transfer_NR_;
  }

  // Implement the contribution of the discrete relaxation operator
  //
  template<class Field>
  template<class Field_Scalar, class Field_Scalar_Unsigned, class Field_Vect>
  auto RelaxationOperator<Field>::make_Newton_step_relaxation(const Field_Scalar& H_bar,
                                                              Field_Scalar& dalpha1_bar,
                                                              Field_Scalar& alpha1_bar,
                                                              Field_Scalar_Unsigned& to_be_relaxed,
                                                              Field_Scalar_Unsigned& Newton_iterations,
                                                              Field_Vect& grad_alpha1_bar,
                                                              Field_Scalar_Unsigned& type_relaxation) {
    auto relaxation_step = samurai::make_cell_based_scheme<typename RelaxationOperator::cfg>();
    relaxation_step.set_name("Relaxation");
    relaxation_step.set_scheme_function([&](samurai::SchemeValue<cfg>& result, const auto& cell, const auto& field)
                                           {
                                             const auto local_field = field[cell];
                                             result = field[cell];

                                             to_be_relaxed[cell] = 0;

                                             const auto H_bar_loc = H_bar[cell];
                                             // Pre-fetch some variables used multiple times in order to exploit possible vectorization
                                             auto alpha1_bar_loc  = alpha1_bar[cell];
                                             auto dalpha1_bar_loc = dalpha1_bar[cell];

                                             const auto m1_loc       = local_field(M1_INDEX);
                                             const auto m2_loc       = local_field(M2_INDEX);
                                             const auto m1_d_loc     = local_field(M1_D_INDEX);
                                             const auto alpha1_d_loc = local_field(ALPHA1_D_INDEX);

                                             // Update auxiliary values affected by the nonlinear function for which we seek a zero
                                             const auto alpha1_loc = alpha1_bar_loc*(static_cast<Number>(1.0) - alpha1_d_loc);
                                             const auto rho1_loc   = m1_loc/alpha1_loc; // TODO: Add a check in case of zero volume fraction
                                             const auto p1_loc     = EOS_phase1.pres_value(rho1_loc);

                                             const auto alpha2_loc = static_cast<Number>(1.0) - alpha1_loc - alpha1_d_loc;
                                             const auto rho2_loc   = m2_loc/alpha2_loc; // TODO: Add a check in case of zero volume fraction
                                             const auto p2_loc     = EOS_phase2.pres_value(rho2_loc);

                                             const auto rho1d_loc     = (m1_d_loc > static_cast<Number>(0.0) &&
                                                                        alpha1_d_loc > static_cast<Number>(0.0)) ?
                                                                        m1_d_loc/alpha1_d_loc : EOS_phase1.get_rho0();
                                             const auto inv_rho1d_loc = static_cast<Number>(1.0)/rho1d_loc;

                                             // Prepare for mass transfer if desired
                                             const auto rho_loc = m1_loc + m2_loc + m1_d_loc;

                                             // Compute first ordrer integral reminder "specific enthalpy"
                                             auto alpha2_bar_loc  = static_cast<Number>(1.0) - alpha1_bar_loc;
                                             const auto p_bar_loc = alpha1_bar_loc*p1_loc
                                                                  + alpha2_bar_loc*p2_loc;
                                             const auto p2_minus_p1_times_theta = rho1_loc/alpha2_bar_loc*
                                                                                  (EOS_phase1.e_value(rho1d_loc) - EOS_phase1.e_value(rho1_loc) +
                                                                                   p_bar_loc*inv_rho1d_loc - p1_loc/rho1_loc) -
                                                                                  (p2_loc - p1_loc);
                                             Number H_lim = std::min(H_bar_loc, Hmax);
                                             const auto fac_Ru = sigma*(static_cast<Number>(3.0)*H_lim/(kappa*rho1d_loc))*
                                                                       (rho1_loc/alpha2_bar_loc)
                                                               - sigma*H_lim/(static_cast<Number>(1.0) - alpha1_d_loc)
                                                               + p2_minus_p1_times_theta;
                                             if(mass_transfer_NR) {
                                               if(fac_Ru > static_cast<Number>(0.0) &&
                                                  alpha1_bar_loc > alpha1_bar_min && alpha1_bar_loc < alpha1_bar_max &&
                                                  -grad_alpha1_bar[cell][0]*local_field(RHO_U_INDEX)
                                                  -grad_alpha1_bar[cell][1]*local_field(RHO_U_INDEX + 1) > static_cast<Number>(0.0) &&
                                                  alpha1_d_loc < alpha1d_max) {
                                                 ;
                                               }
                                               else {
                                                 H_lim = H_bar_loc;
                                               }
                                             }
                                             else {
                                               H_lim = H_bar_loc;
                                             }

                                             const auto dH = H_bar_loc - H_lim;

                                             // Compute the nonlinear function for which we seek the zero (basically the Laplace law)
                                             // Compute the nonlinear function for which we seek the zero (basically the Laplace law)
                                             Number F,
                                                    fac_Ru,
                                                    H_lim,
                                                    dH;
                                             if(!std::isnan(H_bar_loc)) {
                                               type_relaxation[cell] = LOCAL_LAPLACE;

                                               H_lim  = std::min(H_bar_loc, Hmax);
                                               fac_Ru = sigma*(static_cast<Number>(3.0)*H_lim/(kappa*rho1d_loc))*
                                                              (rho1_loc/alpha2_bar_loc)
                                                      - sigma*H_lim/(static_cast<Number>(1.0) - alpha1_d_loc)
                                                      + p2_minus_p1_times_theta;

                                               if(mass_transfer_NR) {
                                                 if(fac_Ru > static_cast<Number>(0.0) &&
                                                    alpha1_bar_loc > alpha1_bar_min && alpha1_bar_loc < alpha1_bar_max &&
                                                    -grad_alpha1_bar_loc[0]*local_conserved_variables(RHO_U_INDEX)
                                                    -grad_alpha1_bar_loc[1]*local_conserved_variables(RHO_U_INDEX + 1) > static_cast<Number>(0.0) &&
                                                    alpha1_d_loc < alpha1d_max) {
                                                   ;
                                                 }
                                                 else {
                                                   H_lim = H_bar_loc;
                                                 }
                                               }
                                               else {
                                                 H_lim = H_bar_loc;
                                               }
                                               dH = H_bar_loc - H_lim;

                                               F = (static_cast<Number>(1.0) - alpha1_d_loc)*(p1_loc - p2_loc)
                                                 - sigma*H_lim;
                                             }
                                             else {
                                               type_relaxation[cell] = PRESSURE_EQUILIBRIUM;

                                               dH = static_cast<Number>(0.0);

                                               F = (static_cast<Number>(1.0) - alpha1_d_loc)*(p1_loc - p2_loc);
                                             }

                                             // Perform the relaxation only where really needed
                                             if(std::abs(F) > atol_Newton + rtol_Newton*((type_relaxation[cell] == PRESSURE_EQUILIBRIUM) ?
                                                                                         EOS_phase1.get_p0() :
                                                                                         std::min(EOS_phase1.get_p0(), sigma*std::abs(H_lim))) &&
                                                std::abs(dalpha1_bar_loc) > atol_Newto)) {
                                               to_be_relaxed[cell] = 1;
                                               Newton_iterations[cell]++;
                                               relaxation_applied = true;

                                               // Compute the derivative w.r.t large scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
                                               const auto c1_loc = EOS_phase1.c_value(rho1_loc);
                                               const auto c2_loc = EOS_phase2.c_value(rho2_loc);

                                               const auto dF_dalpha1_bar = -m1_loc/(alpha1_bar_loc*alpha1_bar_loc)*
                                                                            c1_loc*c1_loc
                                                                           -m2_loc/(alpha2_bar_loc*alpha2_bar_loc)*
                                                                            c2_loc*c2_loc;

                                               // Compute the pseudo time step starting as initial guess from the ideal unmodified Newton method
                                               auto dtau_ov_epsilon = std::numeric_limits<Number>::infinity();

                                               // Bound-preserving condition for m1, velocity and small-scale volume fraction
                                               if(dH > static_cast<Number>(0.0)) {
                                                 // Bound-preserving condition for m1
                                                 dtau_ov_epsilon = lambda*(alpha1_loc*alpha2_bar_loc)/(sigma*dH);
                                                 #ifdef DEBUG_RELAXATION
                                                   if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
                                                     throw std::runtime_error("Negative time step found after relaxation of mass of large-scale phase 1");
                                                   }
                                                 #endif

                                                 // Bound preserving for the velocity
                                                 const auto mom_dot_vel   = (local_field(RHO_U_INDEX)*local_field(RHO_U_INDEX) +
                                                                             local_field(RHO_U_INDEX + 1)*local_field(RHO_U_INDEX + 1))/rho_loc;
                                                 auto dtau_ov_epsilon_tmp = lambda*mom_dot_vel/(dH*fac_Ru*sigma);
                                                 dtau_ov_epsilon          = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                                                 #ifdef DEBUG_RELAXATION
                                                   if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
                                                     throw std::runtime_error("Negative time step found after relaxation of velocity");
                                                   }
                                                 #endif

                                                 // Bound preserving for the small-scale volume fraction
                                                 dtau_ov_epsilon_tmp = lambda*(alpha1d_max - alpha1_d_loc)*alpha2_bar_loc*rho1d_loc/
                                                                              (rho1_loc*sigma*dH);
                                                 dtau_ov_epsilon     = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                                                 if(alpha1_d_loc > static_cast<Number>(0.0)) {
                                                   dtau_ov_epsilon_tmp = lambda*alpha1_d_loc*alpha2_bar_loc*rho1d_loc/
                                                                                (rho1_loc*sigma*dH);

                                                   dtau_ov_epsilon     = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                                                 }
                                                 #ifdef DEBUG_RELAXATION
                                                   if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
                                                     throw std::runtime_error("Negative time step found after relaxation of small-scale volume fraction");
                                                   }
                                                 #endif
                                               }

                                               // Bound-preserving condition for large-scale volume fraction
                                               const auto dF_dalpha1d   = p2_loc - p1_loc
                                                                        + EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)*rho1_loc
                                                                        - EOS_phase2.c_value(rho2_loc)*EOS_phase2.c_value(rho2_loc)*rho2_loc;
                                               const auto dF_dm1        = EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)/alpha1_bar_loc;
                                               const auto R             = dF_dalpha1d*inv_rho1d_loc - dF_dm1;
                                               const auto a             = rho1_loc*sigma*dH*R/
                                                                          (alpha2_bar_loc*(static_cast<Number>(1.0) - alpha1_d_loc));
                                               // Upper bound
                                               auto b                   = (F + lambda*alpha2_bar_loc*dF_dalpha1_bar)/
                                                                          (static_cast<Number>(1.0) - alpha1_d_loc);
                                               auto D                   = b*b - static_cast<Number>(4.0)*a*(-lambda*alpha2_bar_loc);
                                               auto dtau_ov_epsilon_tmp = std::numeric_limits<Number>::infinity();
                                               if(D > static_cast<Number>(0.0) &&
                                                  (a > static_cast<Number>(0.0) ||
                                                  (a < static_cast<Number>(0.0) && b > static_cast<Number>(0.0)))) {
                                                 dtau_ov_epsilon_tmp = static_cast<Number>(0.5)*(-b + std::sqrt(D))/a;
                                               }
                                               if(a == static_cast<Number>(0.0) &&
                                                  b > static_cast<Number>(0.0)) {
                                                 dtau_ov_epsilon_tmp = lambda*alpha2_bar_loc/b;
                                               }
                                               dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                                               // Lower bound
                                               dtau_ov_epsilon_tmp = std::numeric_limits<Number>::infinity();
                                               b                   = (F - lambda*alpha1_bar_loc*dF_dalpha1_bar)/
                                                                     (static_cast<Number>(1.0) - alpha1_d_loc);
                                               D                   = b*b - static_cast<Number>(4.0)*a*(lambda*alpha1_bar_loc);
                                               if(D > static_cast<Number>(0.0) &&
                                                  (a < static_cast<Number>(0.0) ||
                                                  (a > static_cast<Number>(0.0) && b < static_cast<Number>(0.0)))) {
                                                 dtau_ov_epsilon_tmp = static_cast<Number>(0.5)*(-b - std::sqrt(D))/a;
                                               }
                                               if(a == static_cast<Number>(0.0) &&
                                                  b < static_cast<Number>(0.0)) {
                                                 dtau_ov_epsilon_tmp = -lambda*alpha1_bar_loc/b;
                                               }
                                               dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                                               #ifdef DEBUG_RELAXATION
                                                 if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
                                                   throw std::runtime_error("Negative time step found after relaxation of large-scale volume fraction");
                                                 }
                                               #endif

                                               // Compute the effective variation of the variables
                                               if(std::isinf(dtau_ov_epsilon)) {
                                                 // If we are in this branch we do not have mass transfer
                                                 // and we do not have other restrictions on the bounds of large scale volume fraction
                                                 dalpha1_bar_loc = -F/dF_dalpha1_bar;
                                               }
                                               else {
                                                 const auto dm1 = -dtau_ov_epsilon/alpha2_bar_loc*
                                                                   (m1_loc/(alpha1_bar_loc*(static_cast<Number>(1.0) - alpha1_d_loc)))*
                                                                   sigma*dH;

                                                 #ifdef DEBUG_RELAXATION
                                                   if(dm1 > static_cast<Number>(0.0)) {
                                                     throw std::runtime_error("Negative sign of mass transfer inside Newton step");
                                                   }
                                                 #endif
                                                 result(M1_INDEX) += dm1;
                                                 #ifdef DEBUG_RELAXATION
                                                   // I should never get here. Added only for the sake of safety!!
                                                   if(result(M1_INDEX) < static_cast<Number>(0.0)) {
                                                     throw std::runtime_error("Negative mass of large-scale phase 1 inside Newton step");
                                                   }
                                                 #endif

                                                 result(M1_D_INDEX) -= dm1;
                                                 #ifdef DEBUG_RELAXATION
                                                   // I should never get here. Added only for the sake of safety!!
                                                   if(result(M1_D_INDEX) < static_cast<Number>(0.0)) {
                                                     throw std::runtime_error("Negative mass of small-scale phase 1 inside Newton step");
                                                   }
                                                 #endif

                                                 #ifdef DEBUG_RELAXATION
                                                   if(alpha1_d_loc - dm1*inv_rho1d_loc > static_cast<Number>(1.0)) {
                                                     throw std::runtime_error("Exceeding value for small-scale volume fraction inside Newton step");
                                                   }
                                                 #endif
                                                 result(ALPHA1_D_INDEX) -= dm1*inv_rho1d_loc;

                                                 result(SIGMA_D_INDEX) -= dm1*static_cast<Number>(3.0)*Hmax/(kappa*rho1d_loc);

                                                 const auto mom_squared = local_field(RHO_U_INDEX)*local_field(RHO_U_INDEX)
                                                                        + local_field(RHO_U_INDEX + 1)*local_field(RHO_U_INDEX + 1);
                                                 const auto drho_fac_Ru = dtau_ov_epsilon*
                                                                          sigma*dH*fac_Ru*rho_loc/mom_squared;
                                                                          /*--- u/u^{2} = rho*u/(rho*(u^{2})) = (rho/(rho*u)^{2})*(rho*u) ---*/

                                                 for(std::size_t d = 0; d < Field::dim; ++d) {
                                                   result(RHO_U_INDEX + d) -= drho_fac_Ru*result(RHO_U_INDEX + d);
                                                 }

                                                 const auto num_dalpha1_bar = dtau_ov_epsilon/(static_cast<Number>(1.0) - alpha1_d_loc);
                                                 const auto den_dalpha1_bar = static_cast<Number>(1.0) - num_dalpha1_bar*dF_dalpha1_bar;
                                                 dalpha1_bar_loc            = (num_dalpha1_bar/den_dalpha1_bar)*(F - dm1*R);
                                               }

                                               #ifdef DEBUG_RELAXATION
                                                 if(alpha1_bar_loc + dalpha1_bar_loc < static_cast<Number>(0.0) ||
                                                    alpha1_bar_loc + dalpha1_bar_loc > static_cast<Number>(1.0)) {
                                                   // I should never get here. Added only for the sake of safety!!
                                                   throw std::runtime_error("Bounds exceeding value for large-scale volume fraction inside Newton step");
                                                 }
                                               #endif
                                               alpha1_bar_loc += dalpha1_bar_loc;
                                             }

                                             // Update "conservative counter part" of large-scale volume fraction.
                                             // Do it outside because this can change either because of mass transfer or
                                             // of relaxation towards Laplace law.
                                             alpha1_bar[cell]  = alpha1_bar_loc;
                                             dalpha1_bar[cell] = dalpha1_bar_loc;
                                             result(RHO_ALPHA1_BAR_INDEX) = rho_loc*alpha1_bar_loc;
                                           });

    return relaxation_step;
  }

} // end of namespace
