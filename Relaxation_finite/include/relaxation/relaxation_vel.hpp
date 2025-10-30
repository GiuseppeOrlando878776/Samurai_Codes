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
    * Implementation of a instantaneous velocity relaxation operator
    */
  template<class Field>
  class RelaxationVelOperator: public Source<Field> {
  public:
    using Indices = Source<Field>::Indices; /*--- Shortcut for the indices storage ---*/
    using Number  = Source<Field>::Number;  /*--- Shortcut for the arithmetic type ---*/
    using cfg     = Source<Field>::cfg;     /*--- Shortcut to specify the type of configuration
                                                  for the cell-based scheme (nonlinear in this case) ---*/

    RelaxationVelOperator() = default; /*--- Default constructor (sufficient here) ---*/

    virtual decltype(make_cell_based_scheme<cfg>()) make_relaxation() override; /*--- Compute the relaxation ---*/
  };

  // Implement the contribution of the relaxation operator.
  //
  template<class Field>
  decltype(make_cell_based_scheme<typename RelaxationVelOperator<Field>::cfg>())
  RelaxationVelOperator<Field>::make_relaxation() {
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
                                              const auto m1_loc = local_conserved_variables[Indices::ALPHA1_RHO1_INDEX];
                                              auto m1E1_loc     = local_conserved_variables[Indices::ALPHA1_RHO1_E1_INDEX];
                                              const auto m2_loc = local_conserved_variables[Indices::ALPHA2_RHO2_INDEX];
                                              auto m2E2_loc     = local_conserved_variables[Indices::ALPHA2_RHO2_E2_INDEX];

                                              // Save phasic velocities and initial specific internal energy of phase 1 for the total energy update
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

                                              // Compute mixture density and (specific) total energy for the updates
                                              const auto rho_0     = m1_loc + m2_loc;
                                              const auto inv_rho_0 = static_cast<Number>(1.0)/rho_0;
                                              const auto rhoE_0    = m1E1_loc + m2E2_loc;

                                              // Update the momentum (and the kinetic energy of phase 1)
                                              m1E1_loc = static_cast<Number>(0.0);
                                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                                const auto vel_star_d = (local_conserved_variables[Indices::ALPHA1_RHO1_U1_INDEX + d] +
                                                                         local_conserved_variables[Indices::ALPHA2_RHO2_U2_INDEX + d])*inv_rho_0;

                                                local_conserved_variables[Indices::ALPHA1_RHO1_U1_INDEX + d] = m1_loc*vel_star_d;

                                                local_conserved_variables[Indices::ALPHA2_RHO2_U2_INDEX + d] = m2_loc*vel_star_d;

                                                m1E1_loc += static_cast<Number>(0.5)*
                                                            m1_loc*vel_star_d*vel_star_d;
                                              }

                                              // Update total energy of the two phases
                                              const auto Y2_0 = m2_loc*inv_rho_0;
                                              const auto chi1 = static_cast<Number>(0.0); // uI = (1 - chi1)*u1 + chi1*u2;
                                              auto e1_star    = e1_0;
                                              for(std::size_t d = 0; d < Field::dim; ++d) {
                                                e1_star += static_cast<Number>(0.5)*chi1*
                                                           (vel1_loc[d] - vel2_loc[d])*(vel1_loc[d] - vel2_loc[d])*Y2_0;
                                                           // Recall that vel1_loc and vel2_loc are the initial values!!!!
                                              }
                                              m1E1_loc += m1_loc*e1_star;

                                              local_conserved_variables[Indices::ALPHA1_RHO1_E1_INDEX] = m1E1_loc;
                                              local_conserved_variables[Indices::ALPHA2_RHO2_E2_INDEX] = rhoE_0 - m1E1_loc;
                                            });

    return relaxation_step;
  }

} // end of namespace
