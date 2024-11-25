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
    return samurai::line_stencil_from<Field::dim, 0, samurai::Flux<Field>::stencil_size>(-1);
    //return samurai::line_stencil_from<Field::dim, 0, samurai::Flux<Field>::stencil_size>(0);
  }

  inline apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override {
    return [](Field& U, const stencil_cells_t& cells, const value_t& value) {
      //U[cells[1]] = value; //in case of stencil_size = 2
      U[cells[2]] = value; //in case of stencil_size = 4
      U[cells[3]] = value; //in case of stencil_size = 4
    };
  }
};

// Inlet boundary condition for the air-blasted liquid column problem
//
template<class Field>
auto Inlet(const Field& Q,
           const typename Field::value_type ux_D,
           const typename Field::value_type uy_D,
           const typename Field::value_type alpha1_bar_D,
           const typename Field::value_type alpha1_d_D,
           const typename Field::value_type rho1_d_D,
           const typename Field::value_type Sigma_d_D,
           const double eps) {
  return[&Q, ux_D, uy_D, alpha1_bar_D, alpha1_d_D, rho1_d_D, Sigma_d_D, eps]
  (const auto& /*normal*/, const auto& cell_in, const auto& /*coord*/)
  {
    // Compute phasic pressures form the internal state
    const auto alpha1_bar = Q[cell_in](RHO_ALPHA1_BAR_INDEX)/
                            (Q[cell_in](M1_INDEX) + Q[cell_in](M2_INDEX) + Q[cell_in](M1_D_INDEX));
    const auto alpha1     = alpha1_bar*(1.0 - Q[cell_in](ALPHA1_D_INDEX));
    const auto rho1       = (alpha1 > eps) ? Q[cell_in](M1_INDEX)/alpha1 : nan("");

    const auto alpha2     = 1.0 - alpha1 - Q[cell_in](ALPHA1_D_INDEX);
    const auto rho2       = (alpha2 > eps) ? Q[cell_in](M2_INDEX)/alpha2 : nan("");

    // Compute the corresponding ghost state
    xt::xtensor_fixed<typename Field::value_type, xt::xshape<Field::size>> Q_ghost;
    const auto alpha1_D           = alpha1_bar_D*(1.0 - alpha1_d_D);
    const auto alpha2_D           = 1.0 - alpha1_D - alpha1_d_D;
    Q_ghost[M1_INDEX]             = (!std::isnan(rho1)) ? alpha1_D*rho1 : 0.0;
    Q_ghost[M2_INDEX]             = (!std::isnan(rho2)) ? alpha2_D*rho2 : 0.0;
    Q_ghost[M1_D_INDEX]           = alpha1_d_D*rho1_d_D;
    Q_ghost[ALPHA1_D_INDEX]       = alpha1_d_D;
    Q_ghost[SIGMA_D_INDEX]        = Sigma_d_D;
    Q_ghost[RHO_ALPHA1_BAR_INDEX] = (Q_ghost[M1_INDEX] + Q_ghost[M2_INDEX] + Q_ghost[M1_D_INDEX])*alpha1_bar_D;
    Q_ghost[RHO_U_INDEX]          = (Q_ghost[M1_INDEX] + Q_ghost[M2_INDEX] + Q_ghost[M1_D_INDEX])*ux_D;
    Q_ghost[RHO_U_INDEX + 1]      = (Q_ghost[M1_INDEX] + Q_ghost[M2_INDEX] + Q_ghost[M1_D_INDEX])*uy_D;

    return Q_ghost;
  };
}

#endif
