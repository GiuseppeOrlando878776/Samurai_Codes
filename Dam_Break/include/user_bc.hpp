// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#ifndef user_bc_hpp
#define user_bc_hpp

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

// Non-reflecting boundary condition for the dam-break
//
template<class Field>
auto NonReflecting(const Field& Q) {
  return [&Q](const auto& normal, const auto& cell_in, const auto& /*coord*/) {
    /*--- Compute the corresponding ghost state ---*/
    xt::xtensor_fixed<typename Field::value_type, xt::xshape<Field::n_comp>> Q_ghost;
    Q_ghost[M1_INDEX]         = Q[cell_in](M1_INDEX);
    Q_ghost[M2_INDEX]         = Q[cell_in](M2_INDEX);
    Q_ghost[RHO_ALPHA1_INDEX] = Q[cell_in](RHO_ALPHA1_INDEX);

    typename Field::value_type rhou_dot_n = 0.0;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      rhou_dot_n += Q[cell_in](RHO_U_INDEX + d)*normal[d];
    }

    for(std::size_t d = 0; d < Field::dim; ++d) {
      Q_ghost[RHO_U_INDEX + d] = Q[cell_in](RHO_U_INDEX + d) - 2.0*rhou_dot_n*normal[d];
    }

    return Q_ghost;
  };
}

#endif
