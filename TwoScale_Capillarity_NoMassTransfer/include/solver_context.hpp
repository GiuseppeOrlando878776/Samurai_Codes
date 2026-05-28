// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>

/**
 * @file solver_context.hpp
 *
 * @brief Aggregated view of all fields and parameters owned by the solver.
 *
 * SolverContext<Traits, AuxFields> is the single object passed from the solver to
 * TestCaseBase::setup(). It contains:
 *
 *   Mandatory fields (references)
 *   ------------------------------
 *   All minimal mandatory fields needed to perform a simulation
 *   and owned by the solver
 *
 *   Auxiliary fields (reference)
 *   ------------------------------
 *   All fields that the solver owns are exposed here as references
 *   collected in a struct through the template parameter AuxFields.
 *   Every test case receives the full set; each case uses only the
 *   fields it needs inside setup() and ignores the rest. Unused fields
 *   remain untouched in the solver.
 *
 *   Scalar parameters (by value, arbitrary number)
 *   ------------------------------------------------
 *   Lightweight scalars (surface tension coefficient, residual volume
 *   fraction, gravity, ...) are stored in a string-keyed map.  The solver
 *   populates it with everything it has; each test case retrieves only
 *   the keys it needs via param() or has_param().
 *
 * Solver invariance
 * -----------------
 * The solver constructs one SolverContext, populates params, and calls
 * test_case->setup(ctx). This code never changes regardless of which
 * test case is used:
 *   - Adding a new test case never requires modifying the solver.
 *   - Adding a new auxiliary field to the solver means adding one member
 *     to the strcut in the solver constructor — existing test cases
 *     are unaffected.
 *   - Adding a new scalar parameter means one extra params["key"] = val
 *     line in the solver — existing test cases are unaffected.
 *
 * @tparam Traits Traits struct defined in the solver header, exposing
 *                mesh_type, Number, Field, Field_Scalar, Field_Vect,
 *                EOS_type.
 * @tparam AuxFields Struct defined in the solver header with the auxiliary
 *                   fields that each test case has to initialize properly.
 */
template<typename Traits, typename AuxFields>
struct SolverContext {
  // -------------------------------------------------------------------------
  // Type aliases
  // -------------------------------------------------------------------------

  using mesh_type = typename Traits::mesh_type;

  using Number       = typename Traits::Number;
  using Field        = typename Traits::Field;
  using Field_Scalar = typename Traits::Field_Scalar;
  using Field_Vect   = typename Traits::Field_Vect;

  using EOS_type = typename Traits::EOS_type;

  using gradient_type   = typename Traits::gradient_type;
  using divergence_type = typename Traits::divergence_type;

  // -------------------------------------------------------------------------
  // Members (first mandatory, then optional (aux, params)).
  // Some of the mandatory fields can be computed from conserved_variables,
  // but we consder them as mandatory since they are deeply exploited by the
  // solver and not only, e.g., in post-processing. As an example,
  // the surface tension operator requires the gradient of the volume fraction
  // as Field to assemble the operator.
  // -------------------------------------------------------------------------

  mesh_type&      mesh;                /*!< Computational mesh */
  Field&          conserved_variables; /*!< Conserved variable field */
  const EOS_type& EOS_phase1;          /*!< Equation of state for phase 1 */
  const EOS_type& EOS_phase2;          /*!< Equation of state for phase 2 */

  Field_Scalar& alpha1;      /*!< Volume fraction */
  Field_Vect&   grad_alpha1; /*!< Gradient of volume fraction */
  Field_Vect&   normal;      /*!< Interface normal */
  Field_Scalar& H;           /*!< Curvature */

  gradient_type&   gradient;   /*!< Second-order gradient operator */
  divergence_type& divergence; /*!< Second-order divergence operator */

  AuxFields& aux; /*!< All auxiliary fields (single reference, never changes) */

  std::unordered_map<std::string, Number> params; /*!< Scalar parameters populated by the solver */

  /**
   * Class constructor
   *
   * @param mesh_ computational mesh
   * @param conserved_variables varaibles for which we solve the PDE system
   * @param EOS_phase1_ equation of state phase 1
   * @param EOS_phase2_ equation of state phase 2
   * @param alpha1_ volume fraction
   * @param grad_alpha1_ gradient of volume fraction
   * @param normal_ interface normal
   * @param H_ curvature
   * @param gradient_ gradient operator
   * @param divergence_ divergence oeprator
   * @param aux_ struct with auxiliary fields
   * @param params_ auxiliary scalar parameters
   */
  SolverContext(mesh_type& mesh_,
                Field& conserved_variables_,
                const EOS_type& EOS_phase1_,
                const EOS_type& EOS_phase2_,
                Field_Scalar& alpha1_,
                Field_Vect& grad_alpha1_,
                Field_Vect& normal_,
                Field_Scalar& H_,
                gradient_type& gradient_,
                divergence_type& divergence_,
                AuxFields& aux_ = {},
                std::unordered_map<std::string, Number> params_ = {});

  /**
   * Retrieve the scalar parameter associated with @p key.
   * Throws std::out_of_range with a descriptive message if the key is
   * absent, which is clearer than the default std::unordered_map message.
   *
   * @param key Name of the parameter as registered by the solver.
   * @param caller Optional caller name for the error message (e.g. the
   *               test case class name), to ease debugging.
   */
  Number param(const std::string& key,
               const std::string& caller = "") const;
};

// Class constructor
//
template<typename Traits, typename AuxFields>
SolverContext<Traits, AuxFields>::SolverContext(mesh_type& mesh_,
                                                Field& conserved_variables_,
                                                const EOS_type& EOS_phase1_,
                                                const EOS_type& EOS_phase2_,
                                                Field_Scalar& alpha1_,
                                                Field_Vect& grad_alpha1_,
                                                Field_Vect& normal_,
                                                Field_Scalar& H_,
                                                gradient_type& gradient_,
                                                divergence_type& divergence_,
                                                AuxFields& aux_,
                                                std::unordered_map<std::string, Number> params_):
  mesh(mesh_), conserved_variables(conserved_variables_),
  EOS_phase1(EOS_phase1_), EOS_phase2(EOS_phase2_),
  alpha1(alpha1_), grad_alpha1(grad_alpha1_), normal(normal_), H(H_),
  gradient(gradient_), divergence(divergence_),
  aux(aux_), params(std::move(params_)) {}

// Recover the value from the key for the unordered_map
//
template<typename Traits, typename AuxFields>
typename SolverContext<Traits, AuxFields>::Number
SolverContext<Traits, AuxFields>::param(const std::string& key,
                                        const std::string& caller) const {
  auto it = params.find(key);
  if(it == params.end()) {
    const std::string who = caller.empty() ? "SolverContext" : caller;
    throw std::out_of_range(who + "::param(): key '" + key + "' not found in params map. "
                            "Register it in the solver with ctx.params[\"" + key + "\"] = val.");
  }

  return it->second;
}
