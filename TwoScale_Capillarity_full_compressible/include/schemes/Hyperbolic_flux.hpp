// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include <variant>

#include "hyperbolic_fluxes/Rusanov_flux.hpp"
#include "hyperbolic_fluxes/HLLC_flux.hpp"

// Auxiliary type corresponding to the 'factory' of hyperbolic flux
//
template<class Field>
using HyperbolicFlux = std::variant<samurai::RusanovFlux<Field>,
                                    samurai::HLLCFlux<Field>>;

template<class Field, typename... Args>
HyperbolicFlux<Field> get_numerical_hyperbolic_flux(const std::string& scheme,
                                                    Args&&... args) {
  if(scheme == "Rusanov") {
    return samurai::RusanovFlux<Field>(std::forward<Args>(args)...);
  }
  else if(scheme == "HLLC") {
    return samurai::HLLCFlux<Field>(std::forward<Args>(args)...);
  }

  throw std::runtime_error("Unknown scheme: " + scheme);
}

// Get the hyperbolic flux from name
//
template<class Field, typename... Args>
static HyperbolicFlux<Field> create_hyperbolic_flux(const std::string& scheme,
                                                    Args&&... args) {
  try {
    return get_numerical_hyperbolic_flux<Field>(scheme, std::forward<Args>(args)...);
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }
}
