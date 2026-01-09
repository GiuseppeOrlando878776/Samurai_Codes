// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Authors: Loïc Gouarin, 2025
//          Giuseppe Orlando, 2025
//
#pragma once

#include <samurai/numeric/prediction.hpp>
#include <samurai/operators_base.hpp>

#include "scheme/flux_base.hpp"

// Specify the use of this namespace where we just store the indices
using namespace EquationData;

/** This is the class for the high-order prediction for the two-scale capillarity model
 */
template<std::size_t dim, class TInterval>
class DamBreak_prediction_op: public samurai::field_operator_base<dim, TInterval> {
public:
  INIT_OPERATOR(DamBreak_prediction_op)

  inline void operator()(samurai::Dim<dim>,
                         auto& dest,
                         const auto& src) const;
};

// Implement the prediction operator for the dam-break (barotropic 4-equation model)
//
template<std::size_t dim, class TInterval>
inline void DamBreak_prediction_op<dim, TInterval>::operator()(samurai::Dim<dim>,
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
      // Bound-preserving prediction for the liquid mass
      const auto mask_m1 = (dest(M1_INDEX, level + 1, ii) < 0.0 ||
                            dest(M1_INDEX, level + 1, ii + 1) < 0.0);

      samurai::apply_on_masked(mask_m1,
                               [&](auto& ie) {
        std::cout << "Negative liquid mass at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });

      // Bound-preserving prediction for the gas mass
      const auto mask_m2 = (dest(M2_INDEX, level + 1, ii) < 0.0 ||
                            dest(M2_INDEX, level + 1, ii + 1) < 0.0);

      samurai::apply_on_masked(mask_m2,
                               [&](auto& ie) {
        std::cout << "Negative gas mass at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });

      // Bound-preserving prediction for alpha1
      auto compute_alpha1 = [&](const auto& i_) {
        const auto rho = dest(M1_INDEX, level + 1, i_)
                       + dest(M2_INDEX, level + 1, i_);

        return dest(RHO_ALPHA1_INDEX, level + 1, i_)/rho;
      };

      const auto alpha1_ii    = compute_alpha1(ii);
      const auto alpha1_ii_p1 = compute_alpha1(ii + 1);
      const auto mask_alpha1  = (alpha1_ii < 0.0 || alpha1_ii > 1.0 ||
                                 alpha1_ii_p1 < 0.0 || alpha1_ii_p1 > 1.0);
      samurai::apply_on_masked(mask_alpha1,
                               [&](auto& ie) {
        std::cout << "Bound-violating liquid volume fraction at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });
    }
  }
}
