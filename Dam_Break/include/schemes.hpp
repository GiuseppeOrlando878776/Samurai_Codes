// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include "schemes/Rusanov_flux.hpp"
#include "schemes/HLL_flux.hpp"
#include "schemes/HLLC_flux.hpp"

template<class Field, typename... Args>
std::unique_ptr<samurai::Flux<Field>> get_numerical_flux(const std::string& scheme,
                                                         Args&&... args) {
  if(scheme == "Rusanov") {
    return std::make_unique<samurai::RusanovFlux<Field>>(std::forward<Args>(args)...);
  }
  else if(scheme == "HLL") {
    return std::make_unique<samurai::HLLFlux<Field>>(std::forward<Args>(args)...);
  }
  else if(scheme == "HLLC") {
    return std::make_unique<samurai::HLLCFlux<Field>>(std::forward<Args>(args)...);
  }
  else {
    throw std::runtime_error("Unknown scheme: " + scheme);
  }
}
