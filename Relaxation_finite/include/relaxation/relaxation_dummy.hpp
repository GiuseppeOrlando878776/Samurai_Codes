// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#pragma once

#include "relaxation_base.hpp"

namespace samurai {
  /**
    * Implementation of a instantaneous velocity relaxation operator
    */
  template<class Field>
  class RelaxationDummyOperator: public Source<Field> {
  public:
    using cfg = Source<Field>::cfg;     /*--- Shortcut to specify the type of configuration
                                              for the cell-based scheme (nonlinear in this case) ---*/

    RelaxationDummyOperator() = default; /*--- Default constructor ---*/

    virtual decltype(make_cell_based_scheme<cfg>()) make_relaxation() override; /*--- Compute the relaxation ---*/
  };

  // Implement the contribution of the relaxation operator.
  //
  template<class Field>
  decltype(make_cell_based_scheme<typename RelaxationDummyOperator<Field>::cfg>())
  RelaxationDummyOperator<Field>::make_relaxation() {
    auto relaxation_step = samurai::make_cell_based_scheme<cfg>();

    /*--- Perform the instantaneous velocity relaxation ---*/
    relaxation_step.set_scheme_function([&](samurai::SchemeValue<cfg>& local_conserved_variables,
                                            const auto& cell, const auto& field)
                                            {
                                              local_conserved_variables.fill(0.0);
                                            });

    return relaxation_step;
  }

} // end of namespace
