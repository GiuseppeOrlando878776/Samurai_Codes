// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include <samurai/bc.hpp>

#include "flux_base.hpp"

// Specify the use of this namespace where we just store the indices
// and, in this case, some parameters related to EOS
using namespace EquationData;

// Default boundary condition
//
template<class Field>
struct Default: public samurai::Bc<Field> {
  INIT_BC(Default, samurai::Flux<Field>::stencil_size)

  inline stencil_t get_stencil(constant_stencil_size_t) const override {
    #ifdef ORDER_2
      return samurai::line_stencil_from<Field::dim, 0, samurai::Flux<Field>::stencil_size>(-1);
    #else
      return samurai::line_stencil_from<Field::dim, 0, samurai::Flux<Field>::stencil_size>(0);
    #endif
  }

  inline apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override {
    return [](Field& U, const stencil_cells_t& cells, const value_t& value) {
      #ifdef ORDER_2
        U[cells[2]] = value;
        U[cells[3]] = value;
      #else
        U[cells[1]] = value;
      #endif
    };
  }
};

// Inlet boundary condition for the air-blasted liquid column problem
//
template<class Field>
auto Inlet(const Field& Q,
           const typename Field::value_type ux_D,
           const typename Field::value_type uy_D,
           const typename Field::value_type alpha1_D) {
  return[&Q, ux_D, uy_D, alpha1_D]
  (const auto& /*normal*/, const auto& cell_in, const auto& /*coord*/)
  {
    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m1 = Q[cell_in](M1_INDEX);
    const auto m2 = Q[cell_in](M2_INDEX);

    /*--- Compute phasic pressures form the internal state ---*/
    const auto alpha1 = Q[cell_in](RHO_ALPHA1_INDEX)/(m1 + m2);
    const auto rho1   = m1/alpha1; /*--- TODO: Add a check in case of zero volume fraction ---*/

    const auto alpha2 = static_cast<typename Field::value_type>(1.0) - alpha1;
    const auto rho2   = m2/alpha2; /*--- TODO: Add a check in case of zero volume fraction ---*/

    /*--- Compute the corresponding ghost state ---*/
    xt::xtensor_fixed<typename Field::value_type, xt::xshape<Field::n_comp>> Q_ghost;
    const auto alpha2_D       = static_cast<typename Field::value_type>(1.0) - alpha1_D;
    const auto m1_D           = alpha1_D*rho1;
    Q_ghost[M1_INDEX]         = m1_D;
    const auto m2_D           = alpha2_D*rho2;
    Q_ghost[M2_INDEX]         = m2_D;
    const auto rho_D          = m1_D + m2_D;
    Q_ghost[RHO_ALPHA1_INDEX] = rho_D*alpha1_D;
    Q_ghost[RHO_U_INDEX]      = rho_D*ux_D;
    Q_ghost[RHO_U_INDEX + 1]  = rho_D*uy_D;

    return Q_ghost;
  };
}
