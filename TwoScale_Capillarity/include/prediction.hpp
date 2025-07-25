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

#include "flux_base.hpp"

// Specify the use of this namespace where we just store the indices
// and some parameters related to the equations of state
using namespace EquationData;

/** This is the class for the high-order prediction for the two-scale capillarity model
 */
template<std::size_t dim, class TInterval>
class TwoScaleCapillarity_prediction_op : public samurai::field_operator_base<dim, TInterval> {
public:
  INIT_OPERATOR(TwoScaleCapillarity_prediction_op)

  inline void operator()(samurai::Dim<dim>,
                         auto& dest,
                         const auto& src) const;
};

// Implement the prediction operator for the Euler equations
//
template<std::size_t dim, class TInterval>
inline void TwoScaleCapillarity_prediction_op<dim, TInterval>::operator()(samurai::Dim<dim>,
                                                                          auto& dest,
                                                                          const auto& src) const {
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
      // Bound-preserving prediction for the large-scale liquid mass
      const auto mask_m1 = (dest(M1_INDEX, level + 1, ii) < 0.0 ||
                            dest(M1_INDEX, level + 1, ii + 1) < 0.0);

      samurai::apply_on_masked(mask_m1, [&](auto& ie) {
        std::cout << "Negative large-scale liquid mass at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });

      // Bound-preserving prediction for the gas mass
      const auto mask_m2 = (dest(M2_INDEX, level + 1, ii) < 0.0 ||
                            dest(M2_INDEX, level + 1, ii + 1) < 0.0);

      samurai::apply_on_masked(mask_m2, [&](auto& ie) {
        std::cout << "Negative gas mass at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });

      // Bound-preserving prediction for the small-scale liquid mass
      const auto mask_m1_d = (dest(M1_D_INDEX, level + 1, ii) < -1e-15 ||
                              dest(M1_D_INDEX, level + 1, ii + 1) < -1e-15);

      samurai::apply_on_masked(mask_m1_d, [&](auto& ie) {
        std::cout << "Negative small-scale liquid mass at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });

      // Bound-preserving prediction for the small-scale IAD
      const auto mask_Sigma_d = (dest(SIGMA_D_INDEX, level + 1, ii) < -1e-15 ||
                                 dest(SIGMA_D_INDEX, level + 1, ii + 1) < -1e-15);

      samurai::apply_on_masked(mask_Sigma_d, [&](auto& ie) {
        std::cout << "Negative small-scale liquid IAD at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });

      // Bound-preserving prediction for alpha1_bar
      auto compute_alpha1_bar = [&](const auto& i_) {
        const auto rho = dest(M1_INDEX, level + 1, i_)
                       + dest(M2_INDEX, level + 1, i_)
                       + dest(M1_D_INDEX, level + 1, i_);

        return dest(RHO_ALPHA1_BAR_INDEX, level + 1, i_)/rho;
      };

      const auto alpha1_bar_ii    = compute_alpha1_bar(ii);
      const auto alpha1_bar_ii_p1 = compute_alpha1_bar(ii + 1);
      const auto mask_alpha1_bar  = (alpha1_bar_ii < 0.0 || alpha1_bar_ii > 1.0 ||
                                     alpha1_bar_ii_p1 < 0.0 || alpha1_bar_ii_p1 > 1.0);
      samurai::apply_on_masked(mask_alpha1_bar, [&](auto& ie) {
        std::cout << "Bound-violating large-scale liquid volume fraction at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });

      // Bound-preserving prediction for alpha1_d
      const auto mask_alpha1_d = (dest(ALPHA1_D_INDEX, level + 1, ii) < -1e-15 ||
                                  dest(ALPHA1_D_INDEX, level + 1, ii) > 1.0 ||
                                  dest(ALPHA1_D_INDEX, level + 1, ii + 1) < -1e-15 ||
                                  dest(ALPHA1_D_INDEX, level + 1, ii + 1) > 1.0);

      samurai::apply_on_masked(mask_alpha1_d, [&](auto& ie) {
        std::cout << "Bound-violating small-scale liquid volume fraction at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });
    }
  }
}
