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
    * Implementation of a instantaneous relaxation operator (velocity, pressure+temperature)
    */
  template<class Field>
  class RelaxationVelPresTempOperator: public Source<Field> {
  public:
    using Indices = Source<Field>::Indices; /*--- Shortcut for the indices storage ---*/
    using Number  = Source<Field>::Number;  /*--- Shortcut for the arithmetic type ---*/
    using cfg     = Source<Field>::cfg;     /*--- Shortcut to specify the type of configuration
                                                  for the cell-based scheme (nonlinear in this case) ---*/

    RelaxationVelPresTempOperator() = default; /*--- Default constructor (not useful here) ---*/

    RelaxationVelPresTempOperator(const SG_EOS<Number>& EOS_phase1_,
                                  const SG_EOS<Number>& EOS_phase2_); /*--- Class constructor (EOS of the two phases needed here) ---*/

    virtual decltype(make_cell_based_scheme<cfg>()) make_relaxation() override; /*--- Compute the relaxation ---*/

  private:
    const SG_EOS<Number>& EOS_phase1; /*--- EOS phase 1 ---*/
    const SG_EOS<Number>& EOS_phase2; /*--- EOS phase 2 ---*/
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  RelaxationVelPresTempOperator<Field>::
  RelaxationVelPresTempOperator(const SG_EOS<Number>& EOS_phase1_,
                                const SG_EOS<Number>& EOS_phase2_):
    Source<Field>(), EOS_phase1(EOS_phase1_), EOS_phase2(EOS_phase2_) {}

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  decltype(make_cell_based_scheme<typename RelaxationVelPresTempOperator<Field>::cfg>())
  RelaxationVelPresTempOperator<Field>::make_relaxation() {
    auto relaxation_step = samurai::make_cell_based_scheme<cfg>();
    relaxation_step.set_name(this->get_source_name());

    /*--- Perform the instantaneous velocity relaxation ---*/
    relaxation_step.set_scheme_function([&](samurai::SchemeValue<cfg>& local_conserved_variables,
                                            const auto& cell, const auto& field)
                                            {
                                              local_conserved_variables = field[cell];

                                              std::array<Number, Field::dim> vel1_loc;
                                              std::array<Number, Field::dim> vel2_loc;

                                              // Pre-fetch several variables that will be used several times so as to exploit (possible) vectorization
                                              // as well as to enhance readability
                                              auto alpha1_loc   = local_conserved_variables[Indices::ALPHA1_INDEX];
                                              const auto m1_loc = local_conserved_variables[Indices::ALPHA1_RHO1_INDEX];
                                              auto m1E1_loc     = local_conserved_variables[Indices::ALPHA1_RHO1_E1_INDEX];
                                              const auto m2_loc = local_conserved_variables[Indices::ALPHA2_RHO2_INDEX];
                                              auto m2E2_loc     = local_conserved_variables[Indices::ALPHA2_RHO2_E2_INDEX];

                                              // First focus on the velocity relaxation
                                              /*--- Save phasic velocities and initial specific internal energy of phase 1 for the total energy update ---*/
                                              const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc;
                                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                              const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc;
                                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                              auto e1_0             = m1E1_loc*inv_m1_loc;
                                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                                vel1_loc[d] = local_conserved_variables[Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc;
                                                              /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                                vel2_loc[d] = local_conserved_variables[Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc;
                                                              /*--- TODO: Add treatment for vanishing volume fraction ---*/

                                                e1_0 -= static_cast<Number>(0.5)*vel1_loc[d]*vel1_loc[d];
                                              }

                                              /*--- Compute mixture density and (specific) total energy for the updates ---*/
                                              const auto rho_0     = m1_loc + m2_loc;
                                              const auto inv_rho_0 = static_cast<Number>(1.0)/rho_0;
                                              const auto rhoE_0    = m1E1_loc + m2E2_loc;

                                              /*--- Update the momentum (and the kinetic energy of phase 1) ---*/
                                              m1E1_loc            = static_cast<Number>(0.0);
                                              auto norm2_vel_star = static_cast<Number>(0.0);
                                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                                const auto vel_star_d = (local_conserved_variables[Indices::ALPHA1_RHO1_U1_INDEX + d] +
                                                                         local_conserved_variables[Indices::ALPHA2_RHO2_U2_INDEX + d])*inv_rho_0;
                                                norm2_vel_star += vel_star_d*vel_star_d;

                                                local_conserved_variables[Indices::ALPHA1_RHO1_U1_INDEX + d] = m1_loc*vel_star_d;

                                                local_conserved_variables[Indices::ALPHA2_RHO2_U2_INDEX + d] = m2_loc*vel_star_d;

                                                m1E1_loc += static_cast<Number>(0.5)*
                                                            m1_loc*vel_star_d*vel_star_d;
                                              }

                                              /*--- Update total energy of the two phases ---*/
                                              const auto Y2_0 = m2_loc*inv_rho_0;
                                              const auto chi1 = static_cast<Number>(0.0); // uI = (1 - chi1)*u1 + chi1*u2;
                                              auto e1_star    = e1_0;
                                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                                e1_star += static_cast<Number>(0.5)*chi1*
                                                           (vel1_loc[d] - vel2_loc[d])*(vel1_loc[d] - vel2_loc[d])*Y2_0;
                                                           // Recall that vel1_loc and vel2_loc are the initial values!!!!
                                              }
                                              m1E1_loc += m1_loc*e1_star;

                                              m2E2_loc = rhoE_0 - m1E1_loc;

                                              // Focus now on the pressure/temperature relaxation
                                              const auto rhoe_0 = rhoE_0
                                                                - static_cast<Number>(0.5)*rho_0*norm2_vel_star;

                                              const auto a = EOS_phase1.get_cv()*m1_loc
                                                           + EOS_phase2.get_cv()*m2_loc;
                                              const auto b = EOS_phase1.get_q_infty()*EOS_phase1.get_cv()*
                                                             (EOS_phase1.get_gamma() - static_cast<Number>(1.0))*
                                                             m1_loc*m1_loc
                                                           + EOS_phase2.get_q_infty()*EOS_phase2.get_cv()*
                                                             (EOS_phase2.get_gamma() - static_cast<Number>(1.0))*
                                                             m2_loc*m2_loc
                                                           + m1_loc*
                                                             EOS_phase1.get_cv()*(EOS_phase1.get_gamma()*EOS_phase1.get_pi_infty() + EOS_phase2.get_pi_infty())
                                                           + m2_loc*
                                                             EOS_phase2.get_cv()*(EOS_phase2.get_gamma()*EOS_phase2.get_pi_infty() + EOS_phase1.get_pi_infty())
                                                           + m1_loc*m2_loc*
                                                             (EOS_phase1.get_q_infty()*EOS_phase2.get_cv()*
                                                              (EOS_phase2.get_gamma() - static_cast<Number>(1.0)) +
                                                              EOS_phase2.get_q_infty()*EOS_phase1.get_cv()*
                                                              (EOS_phase1.get_gamma() - static_cast<Number>(1.0)))
                                                           - rhoe_0*
                                                             (EOS_phase1.get_cv()*(EOS_phase1.get_gamma() - static_cast<Number>(1.0))*m1_loc +
                                                              EOS_phase2.get_cv()*(EOS_phase2.get_gamma() - static_cast<Number>(1.0))*m2_loc);
                                              const auto d = EOS_phase1.get_q_infty()*EOS_phase1.get_cv()*
                                                             (EOS_phase1.get_gamma() - static_cast<Number>(1.0))*EOS_phase2.get_pi_infty()*
                                                             m1_loc*m1_loc
                                                           + EOS_phase2.get_q_infty()*EOS_phase2.get_cv()*
                                                             (EOS_phase2.get_gamma() - static_cast<Number>(1.0))*EOS_phase1.get_pi_infty()*
                                                             m2_loc*m2_loc
                                                           + (m1_loc*EOS_phase1.get_cv()*EOS_phase1.get_gamma() +
                                                              m2_loc*EOS_phase2.get_cv()*EOS_phase2.get_gamma())*
                                                             EOS_phase1.get_pi_infty()*EOS_phase2.get_pi_infty()
                                                           + m1_loc*m2_loc*
                                                             (EOS_phase1.get_q_infty()*EOS_phase2.get_cv()*
                                                              (EOS_phase2.get_gamma() - static_cast<Number>(1.0))*EOS_phase1.get_pi_infty() +
                                                              EOS_phase2.get_q_infty()*EOS_phase1.get_cv()*
                                                              (EOS_phase1.get_gamma() - static_cast<Number>(1.0))*EOS_phase2.get_pi_infty())
                                                           - rhoe_0*
                                                             (EOS_phase1.get_cv()*(EOS_phase1.get_gamma() - static_cast<Number>(1.0))*
                                                              m1_loc*EOS_phase2.get_pi_infty() +
                                                              EOS_phase2.get_cv()*(EOS_phase2.get_gamma() - static_cast<Number>(1.0))*
                                                              m2_loc*EOS_phase1.get_pi_infty());

                                              const auto p_star = (-b + std::sqrt(b*b - static_cast<Number>(4.0)*a*d))/
                                                                  (static_cast<Number>(2.0)*a);

                                              /*--- Update the volume fraction using the computed pressure ---*/
                                              alpha1_loc = (EOS_phase1.get_cv()*(EOS_phase1.get_gamma() - static_cast<Number>(1.0))*
                                                            (p_star + EOS_phase2.get_pi_infty())*m1_loc)/
                                                           (EOS_phase1.get_cv()*(EOS_phase1.get_gamma() - static_cast<Number>(1.0))*
                                                            (p_star + EOS_phase2.get_pi_infty())*m1_loc +
                                                            EOS_phase2.get_cv()*(EOS_phase2.get_gamma() - static_cast<Number>(1.0))*
                                                            (p_star + EOS_phase1.get_pi_infty())*m2_loc);
                                              local_conserved_variables[Indices::ALPHA1_INDEX] = alpha1_loc;

                                              /*--- Update the total energy of both phases ---*/
                                              const auto E1_loc = EOS_phase1.e_value_RhoP(m1_loc/alpha1_loc, p_star)
                                                                + static_cast<Number>(0.5)*norm2_vel_star;
                                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/

                                              m1E1_loc = m1_loc*E1_loc;
                                              local_conserved_variables[Indices::ALPHA1_RHO1_E1_INDEX] = m1E1_loc;

                                              local_conserved_variables[Indices::ALPHA2_RHO2_E2_INDEX] = rhoE_0 - m1E1_loc;
                                           });

    return relaxation_step;
  }

} // end of namespace
