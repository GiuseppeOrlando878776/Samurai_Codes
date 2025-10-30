// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include "relaxation_base.hpp"

namespace samurai {
  /**
    * Implementation of a finite rate for velocity, pressure, and temperature
    */
  template<class Field>
  class FiniteRatePresTempVel: public Source<Field> {
  public:
    using Indices = Source<Field>::Indices; /*--- Shortcut for the indices storage ---*/
    using Number  = Source<Field>::Number;  /*--- Shortcut for the arithmetic type ---*/
    using cfg     = Source<Field>::cfg;     /*--- Shortcut to specify the type of configuration
                                                  for the cell-based scheme (nonlinear in this case) ---*/

    FiniteRatePresTempVel() = default; /*--- Default constructor (not useful here) ---*/

    FiniteRatePresTempVel(const SG_EOS<Number>& EOS_phase1_,
                          const SG_EOS<Number>& EOS_phase2_,
                          const Number atol_Newton_relaxation_ = static_cast<Number>(1e-12),
                          const Number rtol_Newton_relaxation_ = static_cast<Number>(1e-10),
                          const unsigned max_Newton_iters_ = 60,
                          const Number tau_u_ = static_cast<Number>(1e10),
                          const Number tau_p_ = static_cast<Number>(1e10),
                          const Number tau_T_ = static_cast<Number>(1e10)); /*--- Class constructor (EOS of the two phases and tolerances needed here) ---*/

    virtual decltype(make_cell_based_scheme<cfg>()) make_relaxation() override; /*--- Compute the relaxation ---*/

  private:
    const SG_EOS<Number>& EOS_phase1; /*--- EOS phase 1 ---*/
    const SG_EOS<Number>& EOS_phase2; /*--- EOS phase 2 ---*/

    Number   atol_Newton_relaxation; /*--- Absolute tolerance for the Newton method that reupdates the variables ---*/
    Number   rtol_Newton_relaxation; /*--- Relative tolerance for the Newton method that reupdates the variables ---*/
    unsigned max_Newton_iters;       /*--- Maximum number of iteration of Newton method that reupdates the variables ---*/

    Number tau_u; /*--- Relaxation parameter for the velocity ---*/
    Number tau_p; /*--- Relaxation parameter for the pressure ---*/
    Number tau_T; /*--- Relaxation parameter for the temperature ---*/

    using Matrix_Relaxation = std::array<std::array<Number, 2>, 2>; /*--- Matrix for coupled pressure-temperature relaxation ---*/
    using Vector_Relaxation = std::array<Number, 2>; /*--- Vector associated to the source term ---*/

    template<typename State>
    void compute_coefficients_source_relaxation(const State& q,
                                                const std::array<Number, Field::dim>& delta_u_loc,
                                                Matrix_Relaxation& A,
                                                Vector_Relaxation& S); /*--- Compute the coefficients for the relaxation ---*/
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  FiniteRatePresTempVel<Field>::
  FiniteRatePresTempVel(const SG_EOS<Number>& EOS_phase1_,
                        const SG_EOS<Number>& EOS_phase2_,
                        const Number atol_Newton_relaxation_,
                        const Number rtol_Newton_relaxation_,
                        const unsigned max_Newton_iters_,
                        const Number tau_u_,
                        const Number tau_p_,
                        const Number tau_T_):
    Source<Field>(), EOS_phase1(EOS_phase1_), EOS_phase2(EOS_phase2_),
    atol_Newton_relaxation(atol_Newton_relaxation_),
    rtol_Newton_relaxation(rtol_Newton_relaxation_),
    max_Newton_iters(max_Newton_iters_),
    tau_u(tau_u_), tau_p(tau_p_), tau_T(tau_T_) {}

  // Routine to compute the matrix coefficients associated to the coupled finite-rate pressure-temperature relaxation
  //
  template<class Field>
  template<typename State>
  void FiniteRatePresTempVel<Field>::compute_coefficients_source_relaxation(const State& q,
                                                                            const std::array<Number, Field::dim>& delta_u_loc,
                                                                            Matrix_Relaxation& A,
                                                                            Vector_Relaxation& S) {
    /*--- Pre-fetch variables that will be used several times so as to exploit (possible) vectorization
          as well as to enhance readability ---*/
    const auto alpha1_loc = q[Indices::ALPHA1_INDEX];
    const auto m1_loc     = q[Indices::ALPHA1_RHO1_INDEX];
    const auto m1E1_loc   = q[Indices::ALPHA1_RHO1_E1_INDEX];
    const auto m2_loc     = q[Indices::ALPHA2_RHO2_INDEX];
    const auto m2E2_loc   = q[Indices::ALPHA2_RHO2_E2_INDEX];

    /*--- Compute auxiliary variables for phase 1 ---*/
    const auto inv_alpha1_loc = static_cast<Number>(1.0)/alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho1_loc       = m1_loc*inv_alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    std::array<Number, Field::dim> vel1_loc;
    const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1_loc           = m1E1_loc*inv_m1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      vel1_loc[d] = q[Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      e1_loc -= static_cast<Number>(0.5)*vel1_loc[d]*vel1_loc[d];
    }
    const auto p1_loc = EOS_phase1.pres_value_Rhoe(rho1_loc, e1_loc);
    const auto c1_loc = EOS_phase1.c_value_RhoP(rho1_loc, p1_loc);
    const auto kappa1 = EOS_phase1.de_dP_rho(p1_loc, rho1_loc);
    const auto T1_loc = EOS_phase1.T_value_RhoP(rho1_loc, p1_loc);
    const auto cv1    = EOS_phase1.de_dT_rho(T1_loc, rho1_loc);
    const auto Gamma1 = EOS_phase1.de_drho_T(rho1_loc, T1_loc);

    /*--- Compute auxiliary variables for phase 2 ---*/
    const auto alpha2_loc     = static_cast<Number>(1.0) - alpha1_loc;
    const auto inv_alpha2_loc = static_cast<Number>(1.0)/alpha2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2_loc       = m2_loc*inv_alpha2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    std::array<Number, Field::dim> vel2_loc;
    const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2_loc           = m2E2_loc*inv_m2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < Field::dim; ++d) {
      vel2_loc[d] = q[Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
      e2_loc -= static_cast<Number>(0.5)*vel2_loc[d]*vel2_loc[d];
    }
    const auto p2_loc = EOS_phase2.pres_value_Rhoe(rho2_loc, e2_loc);
    const auto c2_loc = EOS_phase2.c_value_RhoP(rho2_loc, p2_loc);
    const auto kappa2 = EOS_phase2.de_dP_rho(p2_loc, rho2_loc);
    const auto T2_loc = EOS_phase2.T_value_RhoP(rho2_loc, p2_loc);
    const auto cv2    = EOS_phase2.de_dT_rho(T2_loc, rho2_loc);
    const auto Gamma2 = EOS_phase2.de_drho_T(rho2_loc, T2_loc);

    /*--- uI = beta*u1 + (1 - beta)*u2
          pI = chi*p1 + (1 - chi)*p2 ---*/
    auto beta = static_cast<Number>(1.0);
    auto chi  = (static_cast<Number>(1.0) - beta)*T2_loc/
                ((static_cast<Number>(1.0) - beta)*T2_loc + beta*T1_loc);
    auto pI_relax = chi*p1_loc
                  + (static_cast<Number>(1.0) - chi)*p2_loc;
    /*--- TODO: Possibly change, a priori this is not necessarily the same of the convective operator, even though
                substituting uI \cdot grad\alpha we get d\alpha/dt... ---*/

    /*--- Compute the coefficients ---*/
    const auto dt = this->get_dt();

    const auto p_ref_loc     = rho1_loc*c1_loc*c1_loc*inv_alpha1_loc
                             + rho2_loc*c2_loc*c2_loc*inv_alpha2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto p_relax_coeff = static_cast<Number>(1.0)/(tau_p*p_ref_loc);
    const auto T_relax_coeff = (m1_loc*cv1*m2_loc*cv2)/
                               (tau_T*(m1_loc*cv1 + m2_loc*cv2));

    const auto a_pp = -p_relax_coeff*(rho1_loc*c1_loc*c1_loc*inv_alpha1_loc +
                                      rho2_loc*c2_loc*c2_loc*inv_alpha2_loc +
                                      ((chi - static_cast<Number>(1.0))*inv_m1_loc/kappa1 +
                                       chi*inv_m2_loc/kappa2)*
                                      (p1_loc - p2_loc)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto a_pT = -T_relax_coeff*(inv_m1_loc/kappa1 + inv_m2_loc/kappa2); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto a_Tp = -p_relax_coeff*((pI_relax - rho1_loc*rho1_loc*Gamma1)*inv_m1_loc/cv1 +
                                      (pI_relax - rho2_loc*rho2_loc*Gamma2)*inv_m2_loc/cv2);
    const auto a_TT = -T_relax_coeff*(inv_m1_loc*cv1 + inv_m2_loc*cv2); /*--- TODO: Add treatment for vanishing volume fraction ---*/

    A[0][0] = static_cast<Number>(1.0) - dt*a_pp;
    A[0][1] = -dt*a_pT;
    A[1][0] = -dt*a_Tp;
    A[1][1] = static_cast<Number>(1.0) - dt*a_TT;

    /*--- Set source term ---*/
    const auto rho_0      = m1_loc + m2_loc;
    const auto Y1_0       = m1_loc/rho_0;
    const auto Y2_0       = static_cast<Number>(1.0) - Y1_0;
    const auto a_tilde_pu = -static_cast<Number>(1.0)/tau_u*
                            ((Y2_0*(beta - static_cast<Number>(1.0)))/kappa1 + (Y1_0*beta)/kappa2);
    const auto a_tilde_Tu = -static_cast<Number>(1.0)/tau_u*
                            ((Y2_0*(beta - static_cast<Number>(1.0)))/cv1 + (Y1_0*beta)/cv2);
    S[0] = static_cast<Number>(0.0);
    S[1] = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      S[0] += (a_tilde_pu*(vel1_loc[d] - vel2_loc[d]))*delta_u_loc[d];
      S[1] += (a_tilde_Tu*(vel1_loc[d] - vel2_loc[d]))*delta_u_loc[d];
    }
  }

  // Implement the contribution of the relaxation operator.
  //
  template<class Field>
  decltype(make_cell_based_scheme<typename FiniteRatePresTempVel<Field>::cfg>())
  FiniteRatePresTempVel<Field>::make_relaxation() {
    auto relaxation_step = samurai::make_cell_based_scheme<cfg>();
    relaxation_step.set_name(this->get_source_name());

    /*--- Perform the instantaneous velocity relaxation ---*/
    relaxation_step.set_scheme_function([&](samurai::SchemeValue<cfg>& local_conserved_variables,
                                            const auto& cell, const auto& field)
                                            {
                                              local_conserved_variables = field[cell];

                                              std::array<Number, Field::dim> vel1_loc;
                                              std::array<Number, Field::dim> vel2_loc;
                                              std::array<Number, Field::dim> delta_u;

                                              Matrix_Relaxation A_relax; // Matrix associated to the relaxation
                                              Vector_Relaxation S_relax; // Vector associated to the source term

                                              Matrix_Relaxation Jac_update; // Matrix associated to the Jacobian of the Newton method
                                                                            // to update consered variables

                                              // Pre-fetch several variables that will be used several times so as to exploit (possible) vectorization
                                              // as well as to enhance readability
                                              auto alpha1_loc   = local_conserved_variables[Indices::ALPHA1_INDEX];
                                              const auto m1_loc = local_conserved_variables[Indices::ALPHA1_RHO1_INDEX];
                                              auto m1E1_loc     = local_conserved_variables[Indices::ALPHA1_RHO1_E1_INDEX];
                                              const auto m2_loc = local_conserved_variables[Indices::ALPHA2_RHO2_INDEX];
                                              auto m2E2_loc     = local_conserved_variables[Indices::ALPHA2_RHO2_E2_INDEX];

                                              // Compute updated delta_u (we have analytical formula)
                                              const auto dt = this->get_dt();

                                              const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc;
                                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                              const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc;
                                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                                vel1_loc[d] = local_conserved_variables[Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc;
                                                              /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                                vel2_loc[d] = local_conserved_variables[Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc;
                                                              /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                                delta_u[d]  = (vel1_loc[d] - vel2_loc[d])*std::exp(-dt/tau_u);
                                              }

                                              // Solve the system for delta_p and delta_T
                                              /*--- Compute the auxiliary fields to initalize delta_p and delta_T ---*/
                                              auto rho1_loc = m1_loc/alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                              auto e1_loc   = m1E1_loc*inv_m1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                                e1_loc -= static_cast<Number>(0.5)*vel1_loc[d]*vel1_loc[d];
                                              }
                                              auto p1_loc = EOS_phase1.pres_value_Rhoe(rho1_loc, e1_loc);
                                              auto T1_loc = EOS_phase1.T_value_RhoP(rho1_loc, p1_loc);

                                              auto rho2_loc = m2_loc/(static_cast<Number>(1.0) - alpha1_loc);
                                              auto e2_loc   = m2E2_loc*inv_m2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                                e2_loc -= static_cast<Number>(0.5)*vel2_loc[d]*vel2_loc[d];
                                                // Recall that vel1 and vel2 are the initial values!!!!
                                              }
                                              auto p2_loc = EOS_phase2.pres_value_Rhoe(rho2_loc, e2_loc);
                                              auto T2_loc = EOS_phase2.T_value_RhoP(rho2_loc, p2_loc);

                                              /*--- Compute matrix relaxation coefficients ---*/
                                              compute_coefficients_source_relaxation(local_conserved_variables,
                                                                                     delta_u, A_relax, S_relax);

                                              /*--- Solve the linear system ---*/
                                              Number delta_p,
                                                     delta_T;
                                              const auto delta_p0 = p1_loc - p2_loc;
                                              const auto delta_T0 = T1_loc - T2_loc;
                                              const auto det_A_pT = A_relax[0][0]*A_relax[1][1]
                                                                  - A_relax[0][1]*A_relax[1][0];
                                              if(std::abs(det_A_pT) > static_cast<Number>(1e-10)) {
                                                delta_p = (static_cast<Number>(1.0)/det_A_pT)*
                                                          (A_relax[1][1]*(delta_p0 + dt*S_relax[0]) -
                                                           A_relax[0][1]*(delta_T0 + dt*S_relax[1]));
                                                delta_T = (static_cast<Number>(1.0)/det_A_pT)*
                                                          (-A_relax[1][0]*(delta_p0 + dt*S_relax[0]) +
                                                            A_relax[0][0]*(delta_T0 + dt*S_relax[1]));
                                              }
                                              else {
                                                throw std::runtime_error("Singular matrix in the relaxation");
                                              }

                                              // Re-update conserved variables
                                              /*--- Start from phasic momentum (compute also useful norms) ---*/
                                              const auto rho_0  = m1_loc + m2_loc;
                                              const auto Y1_0   = m1_loc/rho_0;
                                              const auto Y2_0   = static_cast<Number>(1.0) - Y1_0;
                                              auto norm2_um     = static_cast<Number>(0.0);
                                              auto norm2_deltau = static_cast<Number>(0.0);
                                              auto norm2_vel1   = static_cast<Number>(0.0);
                                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                                const auto um_d = Y1_0*vel1_loc[d]
                                                                + Y2_0*vel2_loc[d];

                                                norm2_um += um_d*um_d;

                                                vel1_loc[d] = um_d + Y2_0*delta_u[d];
                                                local_conserved_variables[Indices::ALPHA1_RHO1_U1_INDEX + d] = m1_loc*vel1_loc[d];
                                                norm2_vel1 += vel1_loc[d]*vel1_loc[d];

                                                vel2_loc[d] = um_d - Y1_0*delta_u[d];
                                                local_conserved_variables[Indices::ALPHA2_RHO2_U2_INDEX + d] = m2_loc*vel2_loc[d];

                                                norm2_deltau += delta_u[d]*delta_u[d];
                                              }

                                              /*--- Newton method loop to compute pressure and temperature
                                                    (so far we have compute just delta_p and delta_T)
                                                    Pay attention to the variable that we employ in the Newton:
                                                    it is basically necessary to employ the most restricting from a thermodynamic point of view
                                                    so as to compute always admissible state for both phases.
                                                    In this case, this means gas phase, i.e. phase 2 (p2, T2).
                                                    Suppose on the contrary that we update p1. If p1 at the beginning is, e.g., -10^3 Pa
                                                    (typically admissible for the liquid) and delta_p is close to zero
                                                    because we are miming an instantaneous relaxation,
                                                    then p2 is set initially to around -10^3 which is not admissible!!! ---*/
                                              const auto rhoE_0 = m1E1_loc + m2E2_loc;
                                              const auto rhoe_0 = rhoE_0
                                                                - static_cast<Number>(0.5)*rho_0*(norm2_um + Y1_0*Y2_0*norm2_deltau);
                                              auto dp2 = std::numeric_limits<Number>::max();
                                              auto dT2 = std::numeric_limits<Number>::max();
                                              unsigned iter;
                                              auto f1 = m1_loc/EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T)
                                                      + m2_loc/EOS_phase2.rho_value_PT(p2_loc, T2_loc)
                                                      - static_cast<Number>(1.0);
                                              auto f2 = m1_loc*EOS_phase1.e_value_PT(p2_loc + delta_p, T2_loc + delta_T)
                                                      + m2_loc*EOS_phase2.e_value_PT(p2_loc, T2_loc)
                                                      - rhoe_0;
                                              for(iter = 0; iter < max_Newton_iters; ++iter) {
                                                if((std::abs(dp2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(p2_loc) ||
                                                    std::abs(dT2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(T2_loc)) &&
                                                   (std::abs(f1) > atol_Newton_relaxation ||
                                                    std::abs(f2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(rhoe_0))) {
                                                  // Compute Jacobian matrix
                                                  Jac_update[0][0] = -m1_loc/
                                                                     (EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T)*
                                                                      EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T))*
                                                                     EOS_phase1.drho_dP_T(p2_loc + delta_p, T2_loc + delta_T)
                                                                     -m2_loc/
                                                                     (EOS_phase2.rho_value_PT(p2_loc, T2_loc)*
                                                                      EOS_phase2.rho_value_PT(p2_loc, T2_loc))*
                                                                     EOS_phase2.drho_dP_T(p2_loc, T2_loc);
                                                  Jac_update[0][1] = -m1_loc/
                                                                     (EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T)*
                                                                      EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T))*
                                                                     EOS_phase1.drho_dT_P(T2_loc + delta_T, p2_loc + delta_p)
                                                                     -m2_loc/
                                                                     (EOS_phase2.rho_value_PT(p2_loc, T2_loc)*
                                                                      EOS_phase2.rho_value_PT(p2_loc, T2_loc))*
                                                                     EOS_phase2.drho_dT_P(T2_loc, p2_loc);
                                                  Jac_update[1][0] = m1_loc*
                                                                     EOS_phase1.de_dP_T(p2_loc + delta_p, T2_loc + delta_T) +
                                                                     m2_loc*
                                                                     EOS_phase2.de_dP_T(p2_loc, T2_loc);
                                                  Jac_update[1][1] = m1_loc*
                                                                     EOS_phase1.de_dT_P(T2_loc + delta_T, p2_loc + delta_p) +
                                                                     m2_loc*
                                                                     EOS_phase2.de_dT_P(T2_loc, p2_loc);

                                                  // Apply the Newton method
                                                  const auto det_Jac = Jac_update[0][0]*Jac_update[1][1]
                                                                     - Jac_update[0][1]*Jac_update[1][0];
                                                  if(std::abs(det_Jac) > static_cast<Number>(1e-10)) {
                                                    dp2 = (static_cast<Number>(1.0)/det_Jac)*
                                                          (Jac_update[1][1]*f1 - Jac_update[0][1]*f2);
                                                    dp2 = std::min(dp2, static_cast<Number>(0.9)*(p2_loc + EOS_phase2.get_pi_infty()));
                                                    dp2 = std::min(dp2, static_cast<Number>(0.9)*(p2_loc + delta_p + EOS_phase1.get_pi_infty()));
                                                    dT2 = (static_cast<Number>(1.0)/det_Jac)*
                                                          (-Jac_update[1][0]*f1 + Jac_update[0][0]*f2);
                                                    dT2 = std::min(dT2, static_cast<Number>(0.9)*T2_loc);
                                                    dT2 = std::min(dT2, static_cast<Number>(0.9)*(T2_loc + delta_T));
                                                    p2_loc -= dp2;
                                                    T2_loc -= dT2;
                                                  }
                                                  else {
                                                    throw std::runtime_error("Non-invertible Jacobian in the Newton method in the re-update of "
                                                                             "conserved variables after relaxation");
                                                  }

                                                  // Recompute functions for which we look for zero for next iteration
                                                  f1 = m1_loc/EOS_phase1.rho_value_PT(p2_loc + delta_p, T2_loc + delta_T)
                                                     + m2_loc/EOS_phase2.rho_value_PT(p2_loc, T2_loc)
                                                     - static_cast<Number>(1.0);
                                                  f2 = m1_loc*EOS_phase1.e_value_PT(p2_loc + delta_p, T2_loc + delta_T)
                                                     + m2_loc*EOS_phase2.e_value_PT(p2_loc, T2_loc)
                                                     - rhoe_0;
                                                }
                                                else {
                                                  break;
                                                }
                                              }
                                              if(iter == max_Newton_iters &&
                                                 (std::abs(dp2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(p2_loc) ||
                                                  std::abs(dT2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(T2_loc)) &&
                                                 (std::abs(f1) > atol_Newton_relaxation ||
                                                  std::abs(f2) > atol_Newton_relaxation + rtol_Newton_relaxation*std::abs(rhoe_0))) {
                                                throw std::runtime_error("Newton method not converged in the re-update of "
                                                                         "conserved variables after relaxation");
                                              }

                                              // Once pressure and temperature have been update, finalize the update
                                              p1_loc = p2_loc + delta_p;
                                              T1_loc = T2_loc + delta_T;

                                              e1_loc = EOS_phase1.e_value_PT(p1_loc, T1_loc);
                                              local_conserved_variables[Indices::ALPHA1_RHO1_E1_INDEX] = m1_loc*(e1_loc +
                                                                                                                 static_cast<Number>(0.5)*norm2_vel1);

                                              rho1_loc = EOS_phase1.rho_value_PT(p1_loc, T1_loc);
                                              local_conserved_variables[Indices::ALPHA1_INDEX] = m1_loc/rho1_loc;

                                              local_conserved_variables[Indices::ALPHA2_RHO2_E2_INDEX] = rhoE_0
                                                                                                       - local_conserved_variables[Indices::ALPHA1_RHO1_E1_INDEX];
                                           });

    return relaxation_step;
  }

} // end of namespace
