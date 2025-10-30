// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include <samurai/schemes/fv.hpp>

// Memorize indices for the sake of generality
//
template<std::size_t dim>
struct EquationData {
  /*--- Declare suitable static variables for the sake of generalities in the indices ---*/
  static constexpr std::size_t ALPHA1_INDEX         = 0;
  static constexpr std::size_t ALPHA1_RHO1_INDEX    = 1;
  static constexpr std::size_t ALPHA1_RHO1_U1_INDEX = 2;
  static constexpr std::size_t ALPHA1_RHO1_E1_INDEX = ALPHA1_RHO1_U1_INDEX + dim;
  static constexpr std::size_t ALPHA2_RHO2_INDEX    = ALPHA1_RHO1_E1_INDEX + 1;
  static constexpr std::size_t ALPHA2_RHO2_U2_INDEX = ALPHA2_RHO2_INDEX + 1;
  static constexpr std::size_t ALPHA2_RHO2_E2_INDEX = ALPHA2_RHO2_U2_INDEX + dim;

  static constexpr std::size_t NVARS = ALPHA2_RHO2_E2_INDEX + 1;

  /*--- Use auxiliary variables for the indices also for primitive variables for the sake of generality ---*/
  static constexpr std::size_t RHO1_INDEX = ALPHA1_RHO1_INDEX;
  static constexpr std::size_t U1_INDEX   = ALPHA1_RHO1_U1_INDEX;
  static constexpr std::size_t P1_INDEX   = ALPHA1_RHO1_E1_INDEX;
  static constexpr std::size_t RHO2_INDEX = ALPHA2_RHO2_INDEX;
  static constexpr std::size_t U2_INDEX   = ALPHA2_RHO2_U2_INDEX;
  static constexpr std::size_t P2_INDEX   = ALPHA2_RHO2_E2_INDEX;
};

// Reconstruction for second order scheme
//
template<class Field, typename cfg>
void perform_reconstruction(const samurai::FluxValue<cfg>& primLL,
                            const samurai::FluxValue<cfg>& primL,
                            const samurai::FluxValue<cfg>& primR,
                            const samurai::FluxValue<cfg>& primRR,
                            samurai::FluxValue<cfg>& primL_recon,
                            samurai::FluxValue<cfg>& primR_recon) {
  using Number = typename Field::value_type; /*--- Define the shortcut for the arithmetic type ---*/

  /*--- Initialize with the original state ---*/
  primL_recon = primL;
  primR_recon = primR;

  /*--- Perform the reconstruction ---*/
  const auto beta = static_cast<Number>(1.0); // MINMOD limiter
  for(std::size_t comp = 0; comp < Field::n_comp; ++comp) {
    if(primR(comp) - primL(comp) > static_cast<Number>(0.0)) {
      primL_recon(comp) += static_cast<Number>(0.5)*
                           std::max(static_cast<Number>(0.0),
                                    std::max(std::min(beta*(primL(comp) - primLL(comp)),
                                                      primR(comp) - primL(comp)),
                                             std::min(primL(comp) - primLL(comp),
                                                      beta*(primR(comp) - primL(comp)))));
    }
    else if(primR(comp) - primL(comp) < static_cast<Number>(0.0)) {
      primL_recon(comp) += static_cast<Number>(0.5)*
                           std::min(static_cast<Number>(0.0),
                                    std::min(std::max(beta*(primL(comp) - primLL(comp)),
                                                      primR(comp) - primL(comp)),
                                             std::max(primL(comp) - primLL(comp),
                                                      beta*(primR(comp) - primL(comp)))));
    }

    if(primRR(comp) - primR(comp) > static_cast<Number>(0.0)) {
      primR_recon(comp) -= static_cast<Number>(0.5)*
                           std::max(static_cast<Number>(0.0),
                                    std::max(std::min(beta*(primR(comp) - primL(comp)),
                                                      primRR(comp) - primR(comp)),
                                             std::min(primR(comp) - primL(comp),
                                                      beta*(primRR(comp) - primR(comp)))));
    }
    else if(primRR(comp) - primR(comp) < static_cast<Number>(0.0)) {
      primR_recon(comp) -= static_cast<Number>(0.5)*
                           std::min(static_cast<Number>(0.0),
                                    std::min(std::max(beta*(primR(comp) - primL(comp)),
                                                      primRR(comp) - primR(comp)),
                                             std::max(primR(comp) - primL(comp),
                                                      beta*(primRR(comp) - primR(comp)))));
    }
  }
}
