// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include <samurai/schemes/fv.hpp>

/**
 * Useful parameters and enumerators
 */
namespace EquationData {
  // Declare spatial dimension
  static constexpr std::size_t dim = 2;

  // Use auxiliary variables for the indices for the sake of generality
  static constexpr std::size_t M1_INDEX         = 0;
  static constexpr std::size_t M2_INDEX         = 1;
  static constexpr std::size_t RHO_ALPHA1_INDEX = 2;
  static constexpr std::size_t RHO_U_INDEX      = 3;

  // Save also the total number of (scalar) variables
  static constexpr std::size_t NVARS = 3 + dim;

  // Use auxiliary variables for the indices also for primitive variables for the sake of generality
  static constexpr std::size_t ALPHA1_INDEX = RHO_ALPHA1_INDEX;
  static constexpr std::size_t P1_INDEX     = M1_INDEX;
  static constexpr std::size_t P2_INDEX     = M2_INDEX;
  static constexpr std::size_t U_INDEX      = RHO_U_INDEX;
}

/**
 * Useful auxiliary functions not related to a specific class or instance
 */
namespace Utilities {
  // Auxiliary function to convert unsigned to string
  //
  template<typename T>
  std::string unsigned_to_string(const T value, const unsigned digits = 5) {
    std::string lc_string = std::to_string(value);

    if(lc_string.size() < digits) {
      // We have to add the padding zeros in front of the number
      const unsigned int padding_position = (lc_string[0] == '-') ? 1 : 0;

      const std::string padding(digits - lc_string.size(), '0');
      lc_string.insert(padding_position, padding);
    }

    return lc_string;
  }

  // Reconstruction for second order scheme
  //
  template<class Field>
  void perform_reconstruction(const auto& primLL,
                              const auto& primL,
                              const auto& primR,
                              const auto& primRR,
                              auto& primL_recon,
                              auto& primR_recon) {
    using Number = typename Field::value_type; // Define the shortcut for the arithmetic type

    // Initialize with the original state
    primL_recon = primL;
    primR_recon = primR;

    // Perform the reconstruction
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
} // end of namespace
