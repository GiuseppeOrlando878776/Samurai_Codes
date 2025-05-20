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

// Inlet boundary condition for the air-blasted liquid column problem
//
template<class Field>
auto Inlet(const Field& Q,
           const typename Field::value_type ux_D,
           const typename Field::value_type uy_D,
           const typename Field::value_type alpha_l_D,
           const typename Field::value_type alpha_d_D,
           const typename Field::value_type z_D) {
  return[&Q, ux_D, uy_D, alpha_l_D, alpha_d_D, z_D]
  (const auto& /*normal*/, const auto& cell_in, const auto& /*coord*/)
  {
    /*--- Compute phasic pressures form the internal state ---*/
    const auto rho     = Q[cell_in](Ml_INDEX) + Q[cell_in](Mg_INDEX) + Q[cell_in](Md_INDEX);
    const auto alpha_l = Q[cell_in](RHO_ALPHA_l_INDEX)/rho;
    const auto alpha_d = alpha_l*Q[cell_in](Md_INDEX)/Q[cell_in](Ml_INDEX); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto alpha_g = 1.0 - alpha_l - alpha_d;
    const auto rho_liq = (Q[cell_in](Ml_INDEX) + Q[cell_in](Md_INDEX))/(alpha_l + alpha_d); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto rho_g   = Q[cell_in](Mg_INDEX)/alpha_g; /*--- TODO: Add a check in case of zero volume fraction ---*/

    /*--- Compute the corresponding ghost state ---*/
    xt::xtensor_fixed<typename Field::value_type, xt::xshape<Field::n_comp>> Q_ghost;
    const auto alpha_g_D       = 1.0 - alpha_l_D - alpha_d_D;
    Q_ghost[Ml_INDEX]          = alpha_l_D*rho_liq;
    Q_ghost[Mg_INDEX]          = alpha_g_D*rho_g;
    Q_ghost[Md_INDEX]          = alpha_d_D*rho_liq;
    const auto rho_ghost       = Q_ghost[Ml_INDEX] + Q_ghost[Mg_INDEX] + Q_ghost[Md_INDEX];
    Q_ghost[RHO_Z_INDEX]       = rho_ghost*z_D;
    Q_ghost[RHO_ALPHA_l_INDEX] = rho_ghost*alpha_l_D;
    Q_ghost[RHO_U_INDEX]       = rho_ghost*ux_D;
    Q_ghost[RHO_U_INDEX + 1]   = rho_ghost*uy_D;

    return Q_ghost;
  };
}

#endif
