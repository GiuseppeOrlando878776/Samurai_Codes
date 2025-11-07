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
    * Implementation of a instantaneous pressure relaxation operator
    */
  template<class Field>
  class RelaxationPresOperator: public Source<Field> {
  public:
    using Indices = Source<Field>::Indices; /*--- Shortcut for the indices storage ---*/
    using Number  = Source<Field>::Number;  /*--- Shortcut for the arithmetic type ---*/
    using cfg     = Source<Field>::cfg;     /*--- Shortcut to specify the type of configuration
                                                  for the cell-based scheme (nonlinear in this case) ---*/

    RelaxationPresOperator() = default; /*--- Default constructor (not useful here) ---*/

    RelaxationPresOperator(const SG_EOS<Number>& EOS_phase1_,
                           const SG_EOS<Number>& EOS_phase2_); /*--- Class constructor (EOS of the two phases needed here) ---*/

    virtual decltype(make_cell_based_scheme<cfg>()) make_relaxation() override; /*--- Compute the relaxation ---*/

  private:
    const SG_EOS<Number>& EOS_phase1; /*--- EOS phase 1 ---*/
    const SG_EOS<Number>& EOS_phase2; /*--- EOS phase 2 ---*/
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  RelaxationPresOperator<Field>::
  RelaxationPresOperator(const SG_EOS<Number>& EOS_phase1_,
                         const SG_EOS<Number>& EOS_phase2_):
    Source<Field>(), EOS_phase1(EOS_phase1_), EOS_phase2(EOS_phase2_) {}

  // Implement the contribution of the relaxation operator
  //
  template<class Field>
  decltype(make_cell_based_scheme<typename RelaxationPresOperator<Field>::cfg>())
  RelaxationPresOperator<Field>::make_relaxation() {
    auto relaxation_step = samurai::make_cell_based_scheme<cfg>();
    relaxation_step.set_name(this->get_source_name());

    /*--- Perform the instantaneous velocity relaxation ---*/
    relaxation_step.set_scheme_function([&](samurai::SchemeValue<cfg>& local_conserved_variables,
                                            const auto& cell, const auto& field)
                                            {
                                              local_conserved_variables = field[cell];

                                              // Pre-fetch several variables that will be used several times so as to exploit (possible) vectorization
                                              // as well as to enhance readability
                                              auto alpha1_loc   = local_conserved_variables[Indices::ALPHA1_INDEX];
                                              const auto m1_loc = local_conserved_variables[Indices::ALPHA1_RHO1_INDEX];
                                              auto m1E1_loc     = local_conserved_variables[Indices::ALPHA1_RHO1_E1_INDEX];
                                              const auto m2_loc = local_conserved_variables[Indices::ALPHA2_RHO2_INDEX];
                                              auto m2E2_loc     = local_conserved_variables[Indices::ALPHA2_RHO2_E2_INDEX];

                                              // Focus on the pressure relaxation
                                              /*--- Compute the initial fields for the pressure relaxation ---*/
                                              const auto inv_m1_loc = static_cast<Number>(1.0)/m1_loc;
                                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                              const auto inv_m2_loc = static_cast<Number>(1.0)/m2_loc;
                                                                      /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                              auto e1_0             = m1E1_loc*inv_m1_loc;
                                              auto e2_0             = m2E2_loc*inv_m2_loc;
                                              auto norm2_vel1       = static_cast<Number>(0.0);
                                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                                const auto vel1_d = local_conserved_variables[Indices::ALPHA1_RHO1_U1_INDEX + d]*inv_m1_loc;
                                                                    /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                                const auto vel2_d = local_conserved_variables[Indices::ALPHA2_RHO2_U2_INDEX + d]*inv_m2_loc;
                                                                    /*--- TODO: Add treatment for vanishing volume fraction ---*/

                                                e1_0 -= static_cast<Number>(0.5)*vel1_d*vel1_d;
                                                e2_0 -= static_cast<Number>(0.5)*vel2_d*vel2_d;

                                                norm2_vel1 += vel1_d*vel1_d;
                                              }

                                              const auto rhoE_0 = m1E1_loc + m2E2_loc;

                                              auto rho1_loc   = m1_loc/alpha1_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                              auto p1_loc     = EOS_phase1.pres_value_Rhoe(rho1_loc, e1_0);

                                              auto alpha2_loc = static_cast<Number>(1.0) - alpha1_loc;
                                              auto rho2_loc   = m2_loc/alpha2_loc; /*--- TODO: Add treatment for vanishing volume fraction ---*/
                                              auto p2_loc     = EOS_phase2.pres_value_Rhoe(rho2_loc, e2_0);

                                              /*--- Compute the pressure equilibrium with the linearization method (Pelanti) ---*/
                                              const auto a    = static_cast<Number>(1.0)
                                                              + EOS_phase2.get_gamma()*alpha1_loc
                                                              + EOS_phase1.get_gamma()*alpha2_loc;
                                              const auto pI_0 = p2_loc;
                                              const auto C1   = static_cast<Number>(2.0)*EOS_phase1.get_gamma()*EOS_phase1.get_pi_infty()
                                                              + (EOS_phase1.get_gamma() - static_cast<Number>(1.0))*pI_0;
                                              const auto C2   = static_cast<Number>(2.0)*EOS_phase2.get_gamma()*EOS_phase2.get_pi_infty()
                                                              + (EOS_phase2.get_gamma() - static_cast<Number>(1.0))*pI_0;
                                              const auto b    = C1*alpha2_loc + C2*alpha1_loc
                                                              - (static_cast<Number>(1.0) + EOS_phase2.get_gamma())*alpha1_loc*p1_loc
                                                              - (static_cast<Number>(1.0) + EOS_phase1.get_gamma())*alpha2_loc*p2_loc;
                                              const auto d    = -(C2*alpha1_loc*p1_loc + C1*alpha2_loc*p2_loc);

                                              const auto p_star = (-b + std::sqrt(b*b - static_cast<Number>(4.0)*a*d))/
                                                                  (static_cast<Number>(2.0)*a);

                                              /*--- Update the volume fraction using the computed pressure ---*/
                                              alpha1_loc *= ((EOS_phase1.get_gamma() - static_cast<Number>(1.0))*p_star +
                                                             static_cast<Number>(2.0)*p1_loc + C1)/
                                                            ((EOS_phase1.get_gamma() + static_cast<Number>(1.0))*p_star + C1);
                                              local_conserved_variables[Indices::ALPHA1_INDEX] = alpha1_loc;

                                              /*--- Update the total energy of both phases ---*/
                                              const auto E1_loc = EOS_phase1.e_value_RhoP(m1_loc/alpha1_loc, p_star)
                                                                + static_cast<Number>(0.5)*norm2_vel1;
                                                                /*--- TODO: Add treatment for vanishing volume fraction ---*/

                                              m1E1_loc = m1_loc*E1_loc;
                                              local_conserved_variables[Indices::ALPHA1_RHO1_E1_INDEX] = m1E1_loc;

                                              local_conserved_variables[Indices::ALPHA2_RHO2_E2_INDEX] = rhoE_0 - m1E1_loc;
                                            });

    return relaxation_step;
  }

} // end of namespace
