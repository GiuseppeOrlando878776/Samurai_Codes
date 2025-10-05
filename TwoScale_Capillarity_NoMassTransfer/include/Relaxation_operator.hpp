// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include "flux_base.hpp"

namespace samurai {
  using namespace EquationData;

  /**
    * Implementation of a Rusanov flux
    */
  template<class Field>
  class RelaxationOperator {
  public:
    /*--- Definitions and sanity checks ---*/
    static constexpr std::size_t output_field_size = Field::n_comp;

    using Number = typename Field::Number; /*--- Define the shortcut for the arithmetic type ---*/

    using cfg = samurai::LocalCellSchemeConfig<SchemeType::NonLinear, output_field_size, Field>;

    RelaxationOperator(const LinearizedBarotropicEOS<Number>& EOS_phase1_,
                       const LinearizedBarotropicEOS<Number>& EOS_phase2_,
                       const Number sigma_,
                       const Number lambda_ = static_cast<Number>(0.9),
                       const Number atol_Newton_ = static_cast<Number>(1e-14),
                       const Number rtol_Newton_ = static_cast<Number>(1e-12),
                       const std::size_t max_Newton_iters_ = 60); /*--- Constructor which accepts in input the equations of state of the two phases ---*/

    template<typename Field_Scalar, typename Field_Scalar_Unsigned>
    auto make_Newton_step_relaxation(const Field_Scalar& H,
                                     Field_Scalar& dalpha1,
                                     Field_Scalar& alpha1,
                                     Field_Scalar_Unsigned& to_be_relaxed,
                                     Field_Scalar_Unsigned& Newton_iterations,
                                     bool& relaxation_applied); /*--- Compute the flux over all the directions ---*/

  protected:
    const LinearizedBarotropicEOS<Number>& EOS_phase1;
    const LinearizedBarotropicEOS<Number>& EOS_phase2;

    const Number sigma; /*--- Surface tension coefficient ---*/

    const Number      lambda;           /*--- Parameter for bound preserving strategy ---*/
    const Number      atol_Newton;      /*--- Absolute tolerance Newton method relaxation ---*/
    const Number      rtol_Newton;      /*--- Relative tolerance Newton method relaxation ---*/
    const std::size_t max_Newton_iters; /*--- Maximum number of Newton iterations ---*/
  };

  // Constructor derived from the base class
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

  // Implement the contribution of the discrete flux for all the directions.
  //
  template<class Field>
  template<typename Field_Scalar, typename Field_Scalar_Unsigned>
  auto RelaxationOperator<Field>::make_Newton_step_relaxation(const Field_Scalar& H,
                                                              Field_Scalar& dalpha1,
                                                              Field_Scalar& alpha1,
                                                              Field_Scalar_Unsigned& to_be_relaxed,
                                                              Field_Scalar_Unsigned& Newton_iterations,
                                                              bool& relaxation_applied) {
    auto relaxation_step = samurai::make_cell_based_scheme<typename RelaxationOperator::cfg>();
    relaxation_step.set_name("Relaxation");
    relaxation_step.set_scheme_function([&](const auto& cell, const auto& field)
                                           {
                                             samurai::SchemeValue<cfg> local_field = field[cell];
                                             if(!std::isnan(H[cell])) {
                                               /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
                                               const auto m1_loc = local_field(M1_INDEX);
                                               const auto m2_loc = local_field(M2_INDEX);
                                               auto alpha1_loc   = alpha1[cell];
                                               auto alpha2_loc   = static_cast<Number>(1.0) - alpha1_loc;

                                               /*--- Update auxiliary values affected by the nonlinear function for which we seek a zero ---*/
                                               const auto rho1_loc = m1_loc/alpha1_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
                                               const auto p1_loc   = EOS_phase1.pres_value(rho1_loc);

                                               const auto rho2_loc = m2_loc/alpha2_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
                                               const auto p2_loc   = EOS_phase2.pres_value(rho2_loc);

                                               /*--- Compute the nonlinear function for which we seek the zero (basically the Laplace law) ---*/
                                               const auto F = p1_loc - p2_loc - sigma*H[cell];

                                               /*--- Perform the relaxation only where really needed ---*/
                                               if(std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase1.get_p0(), sigma*std::abs(H[cell])) &&
                                                  std::abs(dalpha1[cell]) > atol_Newton) {
                                                 to_be_relaxed[cell] = 1;
                                                 Newton_iterations[cell]++;
                                                 relaxation_applied = true;

                                                 // Compute the derivative w.r.t large-scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
                                                 const auto dF_dalpha1 = -m1_loc/(alpha1_loc*alpha1_loc)*
                                                                          EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)
                                                                         -m2_loc/(alpha2_loc*alpha2_loc)*
                                                                          EOS_phase2.c_value(rho2_loc)*EOS_phase2.c_value(rho2_loc);

                                                 // Compute the large-scale volume fraction update
                                                 auto dalpha1_loc = F/dF_dalpha1;
                                                 if(dalpha1_loc > static_cast<Number>(0.0)) {
                                                   dalpha1_loc = std::min(dalpha1_loc, lambda*alpha2_loc);
                                                 }
                                                 else if(dalpha1[cell] < static_cast<Number>(0.0)) {
                                                   dalpha1_loc = std::max(dalpha1_loc, -lambda*alpha1_loc);
                                                 }
                                                 dalpha1[cell] = dalpha1_loc;

                                                 if(alpha1_loc + dalpha1_loc < static_cast<Number>(0.0) ||
                                                    alpha1_loc + dalpha1_loc > static_cast<Number>(1.0)) {
                                                   throw std::runtime_error("Bounds exceeding value for large-scale volume fraction inside Newton step ");
                                                 }
                                                 else {
                                                   alpha1_loc += dalpha1_loc
                                                   alpha1[cell] = alpha1_loc;
                                                 }
                                               }

                                               /*--- Update the vector of conserved variables
                                                     (probably not the optimal choice since I need this update only at the end of the Newton loop,
                                                      but the most coherent one thinking about the transfer of mass) ---*/
                                               local_field(RHO_ALPHA1_INDEX) = (m1_loc + m2_loc)*alpha1_loc;
                                             }

                                             return local_field;
                                           });

    return relaxation_step;
  }

} // end of namespace
