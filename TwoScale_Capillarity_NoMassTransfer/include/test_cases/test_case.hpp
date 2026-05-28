// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include <functional>

#include "solver_context.hpp"

/**
 * @file test_case.hpp
 *
 * @brief Abstract base class for all test cases.
 *
 * This is the only test-case type the solver needs to know about.
 * It exposes three members:
 *
 *   setup(ctx)
 *     Pure virtual method called once in the solver constructor, after
 *     the mesh and all fields have been created. Each concrete test case
 *     implements setup() to assemble init_fn and bc_fn by capturing
 *     references from the SolverContext and its own private parameters.
 *     The signature of setup() is the sole fixed contract between the
 *     solver and the test case hierarchy, and never changes.
 *
 *   init_fn
 *     Callable with no arguments, valid after setup() returns.
 *     Wraps the concrete init_variables() with an arbitrary signature.
 *     Called once in the solver constructor, immediately after setup().
 *
 *   bc_fn
 *     Callable with no arguments, valid after setup() returns.
 *     Wraps the concrete apply_bcs() with an arbitrary signature.
 *     Called once in the solver constructor, after init_fn().
 *
 * Design rationale
 * ----------------
 * The concrete methods init_variables() and apply_bcs() have completely
 * free signatures and are private to each test case class. The solver
 * never sees them. All solver-side state is passed through a single
 * SolverContext<Traits> object, whose structure is determined solely by
 * what the solver owns — never by what a particular test case needs.
 *
 * @tparam Traits Traits struct defined in the solver header.
 */
template<typename Traits, typename AuxFields>
class TestCaseBase {
public:
  using Context = SolverContext<Traits, AuxFields>;

  virtual ~TestCaseBase() = default;

  /**
   * Assemble init_fn and bc_fn from the solver context.
   *
   * Implementations must populate this->init_fn and this->bc_fn before
   * returning. Both callables may capture references from @p ctx; the
   * solver guarantees that @p ctx outlives both calls.
   *
   * Scalar parameters needed by the test case must be registered in the
   * solver via ctx.params["key"] = val before setup() is called. Use
   * ctx.param("key", "MyCase") inside setup() to retrieve them safely.
   *
   * @param ctx Full solver context: mandatory fields, all auxiliary
   *            fields, and the scalar parameter map.
   */
  virtual void setup(Context& ctx) = 0;

  std::function<void()> init_fn; /*!< Initialises all fields. Valid only after setup() has returned. */

  std::function<void()> bc_fn; /*!< Attaches boundary conditions. Valid only after setup() has returned. */
};
