// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include <samurai/bc.hpp>

#include "schemes/flux_base.hpp"

// Default boundary condition
//
template<class Field>
struct Default: public samurai::Bc<Field> {
  INIT_BC(Default, samurai::Flux<Field>::stencil_size)

  inline stencil_t get_stencil(constant_stencil_size_t) const override {
    return samurai::line_stencil_from<Field::dim, 0, samurai::Flux<Field>::stencil_size>(0);
  }

  inline apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override {
    return [](Field& U, const stencil_cells_t& cells, const value_t& value) {
      U[cells[1]] = value;
    };
  }
};

// 1D Outflow boundary conditions
//
template<class Field>
auto Outflow(const Field& Q,
             const typename Field::value_type p1_D,
             const auto& EOS_phase1,
             const typename Field::value_type p2_D,
             const auto& EOS_phase2) {
  // Set the shortcut for indices storage
  using Indices = samurai::Flux<Field>::Indices;

  // Implement the boundary condition
  return[&Q, p1_D, &EOS_phase1, p2_D, &EOS_phase2]
  (const auto& /*normal*/, const auto& cell_in, const auto& /*coord*/)
  {
    /*--- Pre-fetch some variables in order to exploit possible vectorization ---*/
    const auto alpha1_plus = Q[cell_in](Indices::ALPHA1_INDEX);
    const auto m1_plus     = Q[cell_in](Indices::ALPHA1_RHO1_INDEX);
    const auto m1u1_plus   = Q[cell_in](Indices::ALPHA1_RHO1_U1_INDEX);
    const auto m2_plus     = Q[cell_in](Indices::ALPHA2_RHO2_INDEX);
    const auto m2u2_plus   = Q[cell_in](Indices::ALPHA2_RHO2_U2_INDEX);

    /*--- Compute the corresponding ghost state ---*/
    xt::xtensor_fixed<typename Field::value_type, xt::xshape<Field::n_comp>> Q_ghost;
    Q_ghost[Indices::ALPHA1_INDEX] = alpha1_plus;
    Q_ghost[Indices::ALPHA1_RHO1_INDEX] = m1_plus;
    Q_ghost[Indices::ALPHA1_RHO1_U1_INDEX] = m1u1_plus;
    const auto rho1_plus = m1_plus/alpha1_plus; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto u1_plus = m1u1_plus/m1_plus; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    Q_ghost[Indices::ALPHA1_RHO1_E1_INDEX] = m1_plus*
                                             (EOS_phase1.e_value_RhoP(rho1_plus, p1_D) +
                                              static_cast<typename Field::value_type>(0.5)*u1_plus*u1_plus);

    Q_ghost[Indices::ALPHA2_RHO2_INDEX] = m2_plus;
    Q_ghost[Indices::ALPHA2_RHO2_U2_INDEX] = m2u2_plus;
    const auto rho2_plus = m2_plus/(static_cast<typename Field::value_type>(1.0) - alpha1_plus);
                           /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto u2_plus = m2u2_plus/m2_plus; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    Q_ghost[Indices::ALPHA2_RHO2_E2_INDEX] = m2_plus*
                                             (EOS_phase2.e_value_RhoP(rho2_plus, p2_D) +
                                              static_cast<typename Field::value_type>(0.5)*u2_plus*u2_plus);

    return Q_ghost;
  };
}
