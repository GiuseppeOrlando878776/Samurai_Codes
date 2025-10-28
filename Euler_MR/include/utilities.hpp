// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include <samurai/schemes/fv.hpp>

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
