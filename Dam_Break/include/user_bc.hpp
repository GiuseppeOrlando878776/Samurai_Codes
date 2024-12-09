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
    //return samurai::line_stencil_from<Field::dim, 0, samurai::Flux<Field>::stencil_size>(-1);
    return samurai::line_stencil_from<Field::dim, 0, samurai::Flux<Field>::stencil_size>(0);
  }

  inline apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override {
    return [](Field& U, const stencil_cells_t& cells, const value_t& value) {
      U[cells[1]] = value; //in case of stencil_size = 2
      //U[cells[2]] = value; //in case of stencil_size = 4
      //U[cells[3]] = value; //in case of stencil_size = 4
    };
  }
};

// Non-reflecting boundary condition for the dam-break
//
template<class Field>
auto NonReflecting(const Field& Q) {
  return [&Q](const auto& normal, const auto& cell_in, const auto& /*coord*/) {
    // Compute the corresponding ghost state
    xt::xtensor_fixed<typename Field::value_type, xt::xshape<Field::size>> Q_ghost;
    Q_ghost[M1_INDEX]         = Q[cell_in](M1_INDEX);
    Q_ghost[M2_INDEX]         = Q[cell_in](M2_INDEX);
    Q_ghost[RHO_ALPHA1_INDEX] = Q[cell_in](RHO_ALPHA1_INDEX);

    const auto rhou_dot_n = Q[cell_in](RHO_U_INDEX)*normal[0]
                          + Q[cell_in](RHO_U_INDEX + 1)*normal[1];

    Q_ghost[RHO_U_INDEX]     = Q[cell_in](RHO_U_INDEX) - 2.0*rhou_dot_n*normal[0];
    Q_ghost[RHO_U_INDEX + 1] = Q[cell_in](RHO_U_INDEX + 1) - 2.0*rhou_dot_n*normal[1];

    return Q_ghost;
  };
}

#endif
