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

#include "schemes/flux_base.hpp"

// Specify the use of this namespace where we just store the indices
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
      const auto mask_ml = (dest(Ml_INDEX, level + 1, ii) < 0.0 ||
                            dest(Ml_INDEX, level + 1, ii + 1) < 0.0);

      samurai::apply_on_masked(mask_ml, [&](auto& ie) {
        std::cout << "Negative large-scale liquid mass at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });

      // Bound-preserving prediction for the gas mass
      const auto mask_mg = (dest(Mg_INDEX, level + 1, ii) < 0.0 ||
                            dest(Mg_INDEX, level + 1, ii + 1) < 0.0);

      samurai::apply_on_masked(mask_mg, [&](auto& ie) {
        std::cout << "Negative gas mass at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });

      // Bound-preserving prediction for the small-scale liquid mass
      const auto mask_md = (dest(Md_INDEX, level + 1, ii) < -1e-15 ||
                            dest(Md_INDEX, level + 1, ii + 1) < -1e-15);

      samurai::apply_on_masked(mask_md, [&](auto& ie) {
        std::cout << "Negative small-scale liquid mass at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });

      // Bound-preserving prediction for the z, the variable related to the small-scale IAD
      const auto mask_z = (dest(RHO_Z_INDEX, level + 1, ii) < -1e-15 ||
                           dest(RHO_Z_INDEX, level + 1, ii + 1) < -1e-15);

      samurai::apply_on_masked(mask_z, [&](auto& ie) {
        std::cout << "Negative small-scale liquid IAD at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });

      // Bound-preserving prediction for alpha_l
      auto compute_alpha = [&](const auto& i_) {
        const auto rho = dest(Ml_INDEX, level + 1, i_)
                       + dest(Mg_INDEX, level + 1, i_)
                       + dest(Md_INDEX, level + 1, i_);

        return dest(RHO_ALPHA_l_INDEX, level + 1, i_)/rho;
      };

      const auto alpha_ii    = compute_alpha(ii);
      const auto alpha_ii_p1 = compute_alpha(ii + 1);
      const auto mask_alpha  = (alpha_ii < 0.0 || alpha_ii > 1.0 ||
                                alpha_ii_p1 < 0.0 || alpha_ii_p1 > 1.0);
      samurai::apply_on_masked(mask_alpha, [&](auto& ie) {
        std::cout << "Bound-violating large-scale liquid volume fraction at " << ie
                  << ". Prediction corrected." << std::endl;
        xt::view(dest(level + 1, ii), ie)     = xt::view(src(level, i), ie);
        xt::view(dest(level + 1, ii + 1), ie) = xt::view(src(level, i), ie);
      });
    }
  }
}
