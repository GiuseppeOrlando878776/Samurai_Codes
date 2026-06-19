// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "test_case.hpp"

#include "utilities.hpp"

#include "cases/liquid_column/liquid_column.hpp"
#include "cases/static_bubble/static_bubble.hpp"

/**
 * @file test_case_factory.hpp
 * @brief Factory function that instantiates the correct TestCaseBase subclass.
 *
 * The solver calls make_test_case() once, passing the case name and the path
 * to the case-specific parameter file. Both values are typically read from
 * the main simulation parameter file, e.g.:
 *
 * @code{.json}
 * [test_case]
 * name_tc    = "liquid_column"
 * param_file = "liquid_column/liquid_column.json"
 * @endcode
 *
 * To add a new test case:
 *   1. Create my_case.hpp implementing TestCase<SolverTraits>.
 *   2. #include it above.
 *   3. Add the corresponding "else if" branch below.
 *
 * The solver, SolverContext, and TestCaseBase never change.
 *
 * @tparam Traits Traits struct defined in the solver header.
 * @param name_tc name of the desidered test case
 * @param param_file name of the parameter files.
                     It can be not used if exception in reading json file is used
                     and defualt values are assigned
 */
template<typename Traits, typename AuxFields>
std::unique_ptr<TestCaseBase<Traits, AuxFields>>
make_test_case(const std::string& name_tc,
               const std::string& param_file = "") {
  if constexpr(EquationData::dim == 2) {
    if(name_tc == "liquid_column") {
      return std::make_unique<LiquidColumn<Traits, AuxFields>>(param_file);
    }
    if(name_tc == "static_bubble") {
      return std::make_unique<StaticBubble<Traits, AuxFields>>(param_file);
    }

    throw std::invalid_argument("make_test_case: unknown test case '" + name_tc + "'.\n"
                                "Available cases: liquid_column, static_bubble");
  }

  throw std::invalid_argument("No available test case in " + std::to_string(EquationData::dim) + " dimensions");
}
