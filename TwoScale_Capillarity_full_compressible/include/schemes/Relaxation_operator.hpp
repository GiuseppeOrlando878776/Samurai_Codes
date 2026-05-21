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
     * @param EOS_phase_liq_ liquid equation of state
     * @param EOS_phase_gas_ gas equation of state
     * @param sigma_ surface tension coefficient
     * @param Hmax_ threshold length scale
     * @param kappa_ parameter related to the radius of small-scale droplets
     * @param alpha_d_max_ maximum threshold of small-scale volume fraction
     * @param alpha_l_min_ minimum large-scale volume fraction to identify the mixture region
     * @param alpha_l_max_ maximum large-scale volume fraction to identify the mixture region
     * @param lambda_ bound-preserving parameter
     * @param atol_Newton_ absolute tolerance for dual-time stepping
     * @param rtol_Newton_ relative tolerance for dual-time stepping
     * @param max_Newton_iters_ maximum number of iterations for dual-time stepping
     * @param mass_transfer_NR_ flag to check whether mass transfer inside relaxation is desired
     */
    RelaxationOperator(const LinearizedBarotropicEOS<Number>& EOS_phase_liq_,
                       const LinearizedBarotropicEOS<Number>& EOS_phase_gas_,
                       const Number sigma_,
                       const Number Hmax_,
                       const Number kappa_,
                       const Number alpha_d_max_,
                       const Number alpha_l_min_,
                       const Number alpha_l_max_,
                       const Number lambda_ = static_cast<Number>(0.9),
                       const Number atol_Newton_ = static_cast<Number>(1e-14),
                       const Number rtol_Newton_ = static_cast<Number>(1e-12),
                       const std::size_t max_Newton_iters_ = 60,
                       const bool mass_transfer_NR_ = true);

    /**
     * Perform a Newton step relaxation for a state vector
     * @param H curvature
     * @param dalpha_l variation of volume fraction
     * @param alpha_l volume fraction
     * @param to_be_relaxed auxiliary flag to mark if a field has still to be relaxed or not
     * @param Newton_iterations number of Newton (dual-time stepping) iterations
     * @param grad_alpha_l gradient of large-scale volume fraction (unused here, it may serve in case mass-transfer location changes)
     */
    template<class Field_Scalar, class Field_Scalar_Unsigned, class Field_Vect>
    auto make_Newton_step_relaxation(const Field_Scalar& H,
                                     Field_Scalar& dalpha_l,
                                     Field_Scalar& alpha_l,
                                     Field_Scalar_Unsigned& to_be_relaxed,
                                     Field_Scalar_Unsigned& Newton_iterations,
                                     Field_Vect& grad_alpha_l);
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
    const LinearizedBarotropicEOS<Number>& EOS_phase_liq;
    const LinearizedBarotropicEOS<Number>& EOS_phase_gas;

    const Number sigma; /*!< Surface tension coefficient */

    const Number Hmax;        /*!< Threshold length scale */
    const Number kappa;       /*!< Parameter related to the radius of small-scale droplets */
    const Number alpha_d_max; /*!< Maximum threshold of small-scale volume fraction */
    const Number alpha_l_min; /*!< Minimum large-scale volume fraction to identify the mixture region */
    const Number alpha_l_max; /*!< Maximum large-scale volume fraction to identify the mixture region */

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
  RelaxationOperator<Field>::RelaxationOperator(const LinearizedBarotropicEOS<Number>& EOS_phase_liq_,
                                                const LinearizedBarotropicEOS<Number>& EOS_phase_gas_,
                                                const Number sigma_,
                                                const Number Hmax_,
                                                const Number kappa_,
                                                const Number alpha_d_max_,
                                                const Number alpha_l_min_,
                                                const Number alpha_l_max_,
                                                const Number lambda_,
                                                const Number atol_Newton_,
                                                const Number rtol_Newton_,
                                                const std::size_t max_Newton_iters_,
                                                const bool mass_transfer_NR_):
    EOS_phase_liq(EOS_phase_liq_), EOS_phase_gas(EOS_phase_gas_), sigma(sigma_),
    Hmax(Hmax_), kappa(kappa_),
    alpha_d_max(alpha_d_max_), alpha_l_min(alpha_l_min_), alpha_l_max(alpha_l_max_),
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
  auto RelaxationOperator<Field>::make_Newton_step_relaxation(const Field_Scalar& H,
                                                              Field_Scalar& dalpha_l,
                                                              Field_Scalar& alpha_l,
                                                              Field_Scalar_Unsigned& to_be_relaxed,
                                                              Field_Scalar_Unsigned& Newton_iterations,
                                                              Field_Vect& grad_alpha_l) {
    auto relaxation_step = samurai::make_cell_based_scheme<typename RelaxationOperator::cfg>();
    relaxation_step.set_name("Relaxation");
    relaxation_step.set_scheme_function([&](samurai::SchemeValue<cfg>& result, const auto& cell, const auto& field)
                                           {
                                             const auto local_field = field[cell];
                                             result = field[cell];

                                             to_be_relaxed[cell] = 0;

                                             const auto H_loc = H[cell];
                                             if(!std::isnan(H_loc)) {
                                               // Pre-fetch some variables used multiple times in order to exploit possible vectorization
                                               auto alpha_l_loc  = alpha_l[cell];
                                               auto dalpha_l_loc = dalpha_l[cell];

                                               const auto m_l_loc = local_field(Ml_INDEX);
                                               const auto m_g_loc = local_field(Mg_INDEX);
                                               const auto m_d_loc = local_field(Md_INDEX);

                                               const auto inv_m_l_loc     = static_cast<Number>(1.0)/m_l_loc;
                                               const auto inv_alpha_l_loc = static_cast<Number>(1.0)/alpha_l_loc;

                                               // Update auxiliary values affected by the nonlinear function for which we seek a zero
                                               const auto alpha_d_loc     = alpha_l_loc*m_d_loc*inv_m_l_loc; // TODO: Add a check in case of zero volume fraction
                                               const auto alpha_liq_loc   = alpha_l_loc + alpha_d_loc;
                                               const auto alpha_g_loc     = static_cast<Number>(1.0) - alpha_liq_loc;
                                               const auto inv_alpha_g_loc = static_cast<Number>(1.0)/alpha_g_loc;

                                               const auto m_liq_loc       = m_l_loc + m_d_loc;
                                               const auto rho_liq_loc     = m_liq_loc/alpha_liq_loc; // TODO: Add a check in case of zero volume fraction
                                               const auto inv_rho_liq_loc = static_cast<Number>(1.0)/rho_liq_loc;
                                               const auto p_liq_loc       = EOS_phase_liq.pres_value(rho_liq_loc);
                                               const auto rho_g_loc       = m_g_loc*inv_alpha_g_loc; // TODO: Add a check in case of zero volume fraction
                                               const auto p_g_loc         = EOS_phase_gas.pres_value(rho_g_loc);

                                               // Compute region where performing inter-scale transfer
                                               const auto rho_loc     = m_liq_loc + m_g_loc;
                                               auto H_lim             = std::min(H_loc, Hmax);
                                               const auto fac_Ru      = sigma*Hmax*(static_cast<Number>(3.0)/kappa - static_cast<Number>(1.0));
                                               const auto mom_squared = local_field(RHO_U_INDEX)*local_field(RHO_U_INDEX)
                                                                      + local_field(RHO_U_INDEX + 1)*local_field(RHO_U_INDEX + 1);
                                               const auto mom_dot_vel = mom_squared/rho_loc;
                                               if(mass_transfer_NR) {
                                                 if(alpha_l_loc > alpha_l_min && alpha_l_loc < alpha_l_max &&
                                                    alpha_d_loc < alpha_d_max &&
                                                    alpha_l_loc*fac_Ru <= static_cast<Number>(0.5)*mom_dot_vel) {
                                                   ;
                                                 }
                                                 else {
                                                   H_lim = H_loc;
                                                 }
                                               }
                                               else {
                                                 H_lim = H_loc;
                                               }

                                               const auto dH = H_loc - H_lim;

                                               // Compute the nonlinear function for which we seek the zero (basically the Laplace law)
                                               const auto delta_p = p_liq_loc - p_g_loc;
                                               const auto F_LS    = alpha_l_loc*(delta_p - sigma*H_lim);
                                               const auto aux_SS  = static_cast<Number>(2.0/3.0)*sigma*
                                                                    local_field(RHO_Z_INDEX)*std::cbrt(inv_m_l_loc*inv_m_l_loc*inv_alpha_l_loc);
                                                                    // TODO: Add a check in case of zero volume fraction
                                               const auto F_SS    = alpha_d_loc*delta_p - alpha_l_loc*aux_SS;
                                               const auto F       = F_LS + F_SS;

                                               // Perform the relaxation only where really needed
                                               if((std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase_liq.get_p0(), sigma*std::abs(H_lim)) &&
                                                   std::abs(dalpha_l_loc) > atol_Newton) || dH > rtol_Newton*Hmax) {
                                                 to_be_relaxed[cell] = 1;
                                                 Newton_iterations[cell]++;
                                                 relaxation_applied = true;

                                                 // Compute the derivative w.r.t large scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
                                                 const auto c_liq_loc = EOS_phase_liq.c_value(rho_liq_loc);

                                                 const auto c_g_loc = EOS_phase_gas.c_value(rho_g_loc);

                                                 const auto ddelta_p_dalpha_l = -m_l_loc*inv_alpha_l_loc*inv_alpha_l_loc*
                                                                                c_liq_loc*c_liq_loc
                                                                                -m_g_loc*inv_alpha_g_loc*inv_alpha_g_loc*
                                                                                c_g_loc*c_g_loc*
                                                                                m_liq_loc*inv_m_l_loc; // TODO: Add a check in case of zero volume fraction
                                                 const auto dF_LS_dalpha_l    = (delta_p - sigma*H_lim) + alpha_l_loc*ddelta_p_dalpha_l;
                                                 const auto dF_SS_dalpha_l    = (m_d_loc*inv_m_l_loc)*delta_p
                                                                              + alpha_d_loc*ddelta_p_dalpha_l
                                                                              - static_cast<Number>(2.0/3.0)*aux_SS;
                                                                              // TODO: Add a check in case of zero volume fraction
                                                 const auto dF_dalpha_l       = dF_LS_dalpha_l + dF_SS_dalpha_l;

                                                 // Compute the pseudo time step starting as initial guess from the ideal unmodified Newton method
                                                 auto dtau_ov_epsilon = std::numeric_limits<Number>::infinity();

                                                 // Bound-preserving condition for m_l, velocity and small-scale volume fraction
                                                 if(dH > static_cast<Number>(0.0)) {
                                                   // Bound-preserving condition for m_l
                                                   dtau_ov_epsilon = lambda/(sigma*dH);
                                                   #ifdef DEBUG_RELAXATION
                                                     if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
                                                       throw std::runtime_error("Negative time step found after relaxation of mass of large-scale liquid phase");
                                                     }
                                                   #endif

                                                   // Bound preserving for the velocity
                                                   auto dtau_ov_epsilon_tmp = lambda*mom_dot_vel/(sigma*alpha_l_loc*dH*fac_Ru);
                                                                              // TODO: Add a check in case of zero volume fraction
                                                   dtau_ov_epsilon          = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                                                   #ifdef DEBUG_RELAXATION
                                                     if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
                                                       throw std::runtime_error("Negative time step found after relaxation of velocity");
                                                     }
                                                   #endif

                                                   /*--- No specific condition to impose for the positivity of alpha_d
                                                         since alpha_d = alpha_l*m_d/m_l and m_d is increasing,
                                                         m_l has already been imposed positive and alpha_l is going to be set with proper bounds later.
                                                         On the other hand, there is no a priori superior limit, apart from the alpha_d_max
                                                         which deactivates the mass transfer.
                                                         Hence, in the first iteration, one can potentially reach alpha_d > alpha_d_max
                                                         (likely unphyisical, but not impossible...) ---*/
                                                 }

                                                 // Bound-preserving condition for large-scale volume fraction
                                                 const auto dF_drhoz     = static_cast<Number>(-2.0/3.0)*sigma*std::cbrt(inv_rho_liq_loc*inv_rho_liq_loc);

                                                 const auto ddelta_p_dmd = -m_g_loc*inv_alpha_g_loc*inv_alpha_g_loc*
                                                                           c_g_loc*c_g_loc*inv_rho_liq_loc;
                                                                           // TODO: Add a check in case of zero volume fraction
                                                 const auto dF_LS_dmd    = alpha_l_loc*ddelta_p_dmd;
                                                 const auto dF_SS_dmd    = (delta_p + m_d_loc*ddelta_p_dmd)*inv_rho_liq_loc;
                                                 const auto dF_dmd       = dF_LS_dmd + dF_SS_dmd;

                                                 const auto ddelta_p_dml = c_liq_loc*c_liq_loc*inv_alpha_l_loc
                                                                         + m_g_loc*inv_alpha_g_loc*inv_alpha_g_loc*
                                                                           c_g_loc*c_g_loc*alpha_d_loc*inv_m_l_loc;
                                                                           // TODO: Add a check in case of zero volume fraction
                                                 const auto dF_LS_dml    = alpha_l_loc*ddelta_p_dml;
                                                 const auto dF_SS_dml    = (m_d_loc*ddelta_p_dml -
                                                                            static_cast<Number>(1.0/3.0)*aux_SS)*inv_rho_liq_loc
                                                                         - F_SS*inv_m_l_loc; // TODO: Add a check in case of zero volume fraction
                                                 const auto dF_dml       = dF_LS_dml + dF_SS_dml;

                                                 const auto R            = dF_dml
                                                                         - dF_dmd
                                                                         - dF_drhoz*((static_cast<Number>(3.0)*Hmax/kappa)*std::cbrt(inv_rho_liq_loc));
                                                                           /*NOTE: equivalent to dF_drhoz*(S_avg/m_avg)*((rho*z/Sigma))
                                                                                   since S_avg/m_avg = 3Hmax/(kappa*rho_liq) and rho*z/Sigma = rho_liq^(2/3)*/
                                                 // Upper bound
                                                 const auto r_ml          = -m_l_loc*sigma*dH;
                                                 const auto a             = r_ml*R;
                                                 auto b                   = F + lambda*(static_cast<Number>(1.0) - alpha_l_loc)*dF_dalpha_l;
                                                 auto D                   = b*b
                                                                          - static_cast<Number>(4.0)*a*(-lambda*(static_cast<Number>(1.0) - alpha_l_loc));
                                                 auto dtau_ov_epsilon_tmp = std::numeric_limits<Number>::infinity();
                                                 if(D > static_cast<Number>(0.0) &&
                                                    (a > static_cast<Number>(0.0) ||
                                                     (a < static_cast<Number>(0.0) &&
                                                      b > static_cast<Number>(0.0)))) {
                                                   dtau_ov_epsilon_tmp = static_cast<Number>(0.5)*(-b + std::sqrt(D))/a;
                                                 }
                                                 if(a == static_cast<Number>(0.0) &&
                                                    b > static_cast<Number>(0.0)) {
                                                   dtau_ov_epsilon_tmp = lambda*(static_cast<Number>(1.0) - alpha_l_loc)/b;
                                                 }
                                                 dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                                                 // Lower bound
                                                 dtau_ov_epsilon_tmp = std::numeric_limits<Number>::infinity();
                                                 b                   = F - lambda*alpha_l_loc*dF_dalpha_l;
                                                 D                   = b*b
                                                                     - static_cast<Number>(4.0)*a*(lambda*alpha_l_loc);
                                                 if(D > static_cast<Number>(0.0) &&
                                                    (a < static_cast<Number>(0.0) ||
                                                     (a > static_cast<Number>(0.0) &&
                                                      b < static_cast<Number>(0.0)))) {
                                                   dtau_ov_epsilon_tmp = static_cast<Number>(0.5)*(-b - std::sqrt(D))/a;
                                                 }
                                                 if(a == static_cast<Number>(0.0) &&
                                                    b < static_cast<Number>(0.0)) {
                                                   dtau_ov_epsilon_tmp = -lambda*alpha_l_loc/b;
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
                                                   dalpha_l_loc = -F/dF_dalpha_l;
                                                 }
                                                 else {
                                                   const auto dm_l = dtau_ov_epsilon*r_ml;

                                                   #ifdef DEBUG_RELAXATION
                                                     if(dm_l > static_cast<Number>(0.0)) {
                                                       throw std::runtime_error("Negative sign of mass transfer inside Newton step");
                                                     }
                                                   #endif
                                                   result(Ml_INDEX) += dm_l;
                                                   #ifdef DEBUG_RELAXATION
                                                     if(result(Ml_INDEX) < static_cast<Number>(0.0)) {
                                                       // I should never get here. Added only for the sake of safety!!
                                                       throw std::runtime_error("Negative mass of large-scale liquid phase inside Newton step");
                                                     }
                                                   #endif

                                                   result(Md_INDEX) -= dm_l;
                                                   #ifdef DEBUG_RELAXATION
                                                     if(result(Md_INDEX) < static_cast<Number>(0.0)) {
                                                       // I should never get here. Added only for the sake of safety!!
                                                       throw std::runtime_error("Negative mass of small-scale liquid phase inside Newton step");
                                                     }
                                                   #endif

                                                   const auto R_Sigma_d = -dm_l*((static_cast<Number>(3.0)*Hmax/kappa)*inv_rho_liq_loc);
                                                   result(RHO_Z_INDEX) += std::cbrt(rho_liq_loc*rho_liq_loc)*R_Sigma_d;

                                                   const auto drho_fac_Ru = dtau_ov_epsilon*
                                                                            (sigma*alpha_l_loc*dH*fac_Ru)*rho_loc/mom_squared;
                                                                            /*--- u/u^{2} = rho*u/(rho*(u^{2})) = (rho/(rho*u)^{2})*(rho*u) ---*/
                                                   for(std::size_t d = 0; d < Field::dim; ++d) {
                                                     result(RHO_U_INDEX + d) -= drho_fac_Ru*result(RHO_U_INDEX + d);
                                                   }

                                                   dalpha_l_loc = dtau_ov_epsilon/(static_cast<Number>(1.0) - dtau_ov_epsilon*dF_dalpha_l)*
                                                                  (F + dm_l*R);
                                                 }

                                                 #ifdef DEBUG_RELAXATION
                                                   if(alpha_l_loc + dalpha_l_loc < static_cast<Number>(0.0) ||
                                                      alpha_l_loc + dalpha_l_loc > static_cast<Number>(1.0)) {
                                                     // I should never get here. Added only for the sake of safety!!
                                                     throw std::runtime_error("Bounds exceeding value for large-scale volume fraction inside Newton step");
                                                   }
                                                 #endif
                                                 alpha_l_loc += dalpha_l_loc;
                                                 alpha_l[cell]  = alpha_l_loc;
                                                 dalpha_l[cell] = dalpha_l_loc;
                                                 result(RHO_ALPHA_l_INDEX) = rho_loc*alpha_l_loc;
                                               }
                                             }
                                           });

    return relaxation_step;
  }

} // end of namespace
