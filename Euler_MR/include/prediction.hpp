// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Authors: Loic Gouarin, 2025
//          Giuseppe Orlando, 2025
//
#pragma once

#include <samurai/numeric/prediction.hpp>
#include <samurai/operators_base.hpp>

#include "schemes/flux_base.hpp"

// Specify the use of this namespace where we just store the indices
using namespace EquationData;

/** This is the class for the high-order prediction for the Euler equations
 */
template<std::size_t dim, class TInterval>
class Euler_prediction_op : public samurai::field_operator_base<dim, TInterval> {
public:
  INIT_OPERATOR(Euler_prediction_op)

  inline void operator()(samurai::Dim<1>,
                         auto& dest,
                         const auto& src,
                         const auto& Euler_EOS) const;
};

// Implement the prediction operator for the Euler equations
//
template<std::size_t dim, class TInterval>
inline void Euler_prediction_op<dim, TInterval>::operator()(samurai::Dim<1>,
                                                            auto& dest,
                                                            const auto& src,
                                                            const auto& Euler_EOS) const {
  using field_t = std::decay_t<decltype(src)>;

  /*--- Compute unchanged prediction ---*/
  constexpr std::size_t pred_order = field_t::mesh_t::config::prediction_order;

  auto ii = i << 1;
  ii.step = 2;

  auto qs_i = samurai::Qs_i<pred_order>(src, level, i);

  dest(level + 1, ii)     = src(level, i) + qs_i;
  dest(level + 1, ii + 1) = src(level, i) - qs_i;

  /*--- Use bound-preserving prediction ---*/
  if constexpr (!field_t::is_scalar) {
    if(src.name() == "conserved") {
      // Bound-preserving prediction for the density
      const auto mask_rho = (dest(RHO_INDEX, level + 1, ii) < 0.0 ||
                             dest(RHO_INDEX, level + 1, ii + 1) < 0.0);

      samurai::apply_on_masked(mask_rho, [&](auto& ie) {
        std::cout << "Negative density at " << ie << ". Prediction corrected."
                  << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });

      // Bound-preserving prediction for the pressure
      auto compute_p = [&](const auto& i_) {
        const auto rho       = dest(RHO_INDEX, level + 1, i_);
        const auto rho_u     = dest(RHOU_INDEX, level + 1, i_);
        const auto rho_E     = dest(RHOE_INDEX, level + 1, i_);
        const auto q_infty   = Euler_EOS.get_q_infty();
        const auto pi_infty  = Euler_EOS.get_pi_infty();
        const auto gamma     = Euler_EOS.get_gamma();
        const auto vel       = rho_u/rho;
        const auto norm2_vel = vel*vel;
        const auto e         = rho_E/rho - 0.5*norm2_vel;

        return xt::eval((gamma - 1.0)*rho*(e - q_infty) -
                        gamma * pi_infty);
      };

      const auto p_ii    = compute_p(ii);
      const auto p_ii_p1 = compute_p(ii + 1);
      const auto mask_p  = (p_ii < 0.0 || p_ii_p1 < 0.0);
      samurai::apply_on_masked(mask_p, [&](auto& ie) {
        std::cout << "Negative pressure at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });
    }
  }
}
