// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include <samurai/bc.hpp>

#include "schemes/flux_base.hpp"

// Specify the use of this namespace where we just store the indices
using namespace EquationData;

/* TO DO: Modify the configuration of test case for non-isotheral flows (not so trivial a priori) */

/**
 * Default boundary condition
 */
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

/**
 * Inlet boundary condition for the air-blasted liquid column problem
 * @param Q field with conserved variables (i.e. the variables for which we solve the PDE system)
 * @param grad_alpha_l gradient of large-scale volume fraction
 * @param sigma surface tension coefficient
 * @param ux_D boundary horizontal component of the velocity
 * @param uy_D boundary vertical component of the velocity
 * @param alpha_l_D boundary large-scale volume fraction
 * @param alpha_d_D boundary small-scale volume fraction
 * @param Sigma_d_D boundary small-scale interfacial area density
 */
template<class Field>
auto Inlet(const Field& Q,
           const auto& grad_alpha_l,
           const typename Field::value_type sigma,
           const typename Field::value_type ux_D,
           const typename Field::value_type uy_D,
           const typename Field::value_type alpha_l_D,
           const typename Field::value_type alpha_d_D,
           const typename Field::value_type Sigma_d_D) {
  return[&Q, &grad_alpha_l, sigma, ux_D, uy_D, alpha_l_D, alpha_d_D, Sigma_d_D]
  (const auto& /*normal*/, const auto& cell_in, const auto& /*coord*/)
  {
    // Pre-fetch some variables used multiple times in order to exploit possible vectorization
    const auto m_l_loc = Q[cell_in](Ml_INDEX);
    const auto m_g_loc = Q[cell_in](Mg_INDEX);
    const auto m_d_loc = Q[cell_in](Md_INDEX);

    // Compute liquid density
    const auto m_liq_loc     = m_l_loc + m_d_loc;
    const auto rho_loc       = m_liq_loc + m_g_loc;
    const auto inv_rho_loc   = static_cast<typename Field::value_type>(1.0)/rho_loc;
    const auto alpha_l_loc   = Q[cell_in](RHO_ALPHA_l_INDEX)*inv_rho_loc;
    const auto alpha_d_loc   = alpha_l_loc*m_d_loc/m_l_loc; // TODO: Add a check in case of zero volume fraction
    const auto alpha_liq_loc = alpha_l_loc + alpha_d_loc;
    const auto rho_liq_loc   = m_liq_loc/alpha_liq_loc; // TODO: Add a check in case of zero volume fraction

    // Compute liquid internal energy
    const auto Sigma_d_loc = Q[cell_in](RHO_Z_INDEX)/std::cbrt(rho_liq_loc*rho_liq_loc);

    auto norm2_vel_loc = static_cast<typename Field::value_type>(0.0);
    for(std::size_t d = 0; d < dim; ++d) {
      const auto vel_loc_d = Q[cell_in](RHO_U_INDEX + d)*inv_rho_loc;
      norm2_vel_loc += vel_loc_d*vel_loc_d;
    }

    const auto& grad_alpha_l_loc = grad_alpha_l[cell_in];
    auto mod2_grad_alpha_l_loc   = static_cast<typename Field::value_type>(0.0);
    for(std::size_t d = 0; d < dim; ++d) {
      mod2_grad_alpha_l_loc += grad_alpha_l_loc[d]*grad_alpha_l_loc[d];
    }
    const auto mod_grad_alpha_l_loc = std::sqrt(mod2_grad_alpha_l_loc);

    const auto Y_liq_loc   = m_liq_loc*inv_rho_loc;
    const auto chi_liq_loc = Y_liq_loc;
    const auto e_liq_loc   = Q[cell_in](Mliq_Eliq_INDEX)/m_liq_loc
                           - static_cast<typename Field::value_type>(0.5)*norm2_vel_loc
                           - sigma*inv_rho_loc*(chi_liq_loc/Y_liq_loc)*(mod_grad_alpha_l_loc + Sigma_d_loc);
                           // TODO: Add a check in case of zero volume fraction

    // Compute gas density
    const auto alpha_g_loc = static_cast<typename Field::value_type>(1.0) - alpha_liq_loc;
    const auto rho_g_loc   = m_g_loc/alpha_g_loc; // TODO: Add a check in case of zero volume fraction

    // Compute gas internal energy
    const auto Y_g_loc   = static_cast<typename Field::value_type>(1.0) - Y_liq_loc;
    const auto chi_g_loc = Y_g_loc;
    const auto e_g_loc   = Q[cell_in](Mg_Eg_INDEX)/m_g_loc
                         - static_cast<typename Field::value_type>(0.5)*norm2_vel_loc
                         - sigma*inv_rho_loc*(chi_g_loc/Y_g_loc)*(mod_grad_alpha_l_loc + Sigma_d_loc);
                         // TODO: Add a check in case of zero volume fraction

    // Compute the corresponding ghost state
    xt::xtensor_fixed<typename Field::value_type, xt::xshape<Field::n_comp>> Q_ghost;
    const auto alpha_g_D       = static_cast<typename Field::value_type>(1.0) - alpha_l_D - alpha_d_D;
    const auto m_l_D           = alpha_l_D*rho_liq_loc;
    Q_ghost[Ml_INDEX]          = m_l_D;
    const auto m_g_D           = alpha_g_D*rho_g_loc;
    Q_ghost[Mg_INDEX]          = m_g_D;
    const auto m_d_D           = alpha_d_D*rho_liq_loc;
    Q_ghost[Md_INDEX]          = m_d_D;
    Q_ghost[RHO_Z_INDEX]       = Sigma_d_D*std::cbrt(rho_liq_loc*rho_liq_loc);
    const auto m_liq_D         = m_l_D + m_d_D;
    const auto rho_D           = m_liq_D + m_g_D;
    Q_ghost[RHO_ALPHA_l_INDEX] = rho_D*alpha_l_D;
    Q_ghost[RHO_U_INDEX]       = rho_D*ux_D;
    Q_ghost[RHO_U_INDEX + 1]   = rho_D*uy_D;
    const auto inv_rho_D       = static_cast<typename Field::value_type>(1.0)/rho_D;
    const auto Y_liq_D         = m_liq_D*inv_rho_D;
    const auto chi_liq_D       = Y_liq_D;
    Q_ghost[Mliq_Eliq_INDEX]   = m_liq_D*(e_liq_loc +
                                          static_cast<typename Field::value_type>(0.5)*(ux_D*ux_D + uy_D*uy_D) +
                                          sigma*inv_rho_D*(chi_liq_D/Y_liq_D)*(mod_grad_alpha_l_loc + Sigma_d_D));
    const auto Y_g_D           = static_cast<typename Field::value_type>(1.0) - Y_liq_D;
    const auto chi_g_D         = Y_g_D;
    Q_ghost[Mg_Eg_INDEX]       = m_g_D*(e_g_loc +
                                        static_cast<typename Field::value_type>(0.5)*(ux_D*ux_D + uy_D*uy_D) +
                                        sigma*inv_rho_D*(chi_g_D/Y_g_D)*(mod_grad_alpha_l_loc + Sigma_d_D));

    return Q_ghost;
  };
}
