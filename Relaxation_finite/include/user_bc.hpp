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

// Outflow boundary condition for the air-blasted liquid column problem
//
template<class Field>
auto Outflow(const Field& Q,
             const typename Field::value_type p1_D,
             const auto& EOS_phase1,
             const typename Field::value_type p2_D,
             const auto& EOS_phase2) {
  return[&Q, p1_D, &EOS_phase1, p2_D, &EOS_phase2]
  (const auto& /*normal*/, const auto& cell_in, const auto& /*coord*/)
  {
    // Compute the corresponding ghost state
    xt::xtensor_fixed<typename Field::value_type, xt::xshape<Field::size>> Q_ghost;
    Q_ghost[ALPHA1_INDEX]         = Q[cell_in](ALPHA1_INDEX);
    Q_ghost[ALPHA1_RHO1_INDEX]    = Q[cell_in](ALPHA1_RHO1_INDEX);
    const auto rho1_plus          = Q[cell_in](ALPHA1_RHO1_INDEX)/Q[cell_in](ALPHA1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto u1_plus            = Q[cell_in](ALPHA1_RHO1_U1_INDEX)/Q[cell_in](ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    Q_ghost[ALPHA1_RHO1_U1_INDEX] = Q_ghost[ALPHA1_RHO1_INDEX]*u1_plus;
    Q_ghost[ALPHA1_RHO1_E1_INDEX] = Q_ghost[ALPHA1_RHO1_INDEX]*
                                    (EOS_phase1.e_value_RhoP(rho1_plus, p1_D) +
                                     0.5*u1_plus*u1_plus);

    Q_ghost[ALPHA2_RHO2_INDEX]    = Q[cell_in](ALPHA2_RHO2_INDEX);
    const auto rho2_plus          = Q[cell_in](ALPHA2_RHO2_INDEX)/(1.0 - Q[cell_in](ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto u2_plus            = Q[cell_in](ALPHA2_RHO2_U2_INDEX)/Q[cell_in](ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    Q_ghost[ALPHA2_RHO2_U2_INDEX] = Q_ghost[ALPHA2_RHO2_INDEX]*u2_plus;
    Q_ghost[ALPHA2_RHO2_E2_INDEX] = Q_ghost[ALPHA2_RHO2_INDEX]*
                                    (EOS_phase2.e_value_RhoP(rho2_plus, p2_D) +
                                     0.5*u2_plus*u2_plus);

    return Q_ghost;
  };
}

#endif
