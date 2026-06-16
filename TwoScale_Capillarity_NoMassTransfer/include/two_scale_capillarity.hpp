// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/io/hdf5.hpp>

#include <filesystem>
namespace fs = std::filesystem;

/*--- Add header file for the multiresolution ---*/
#include <samurai/mr/adapt.hpp>
//#include "prediction.hpp"

/*--- Add header with auxiliary structs ---*/
#include "containers.hpp"

/*--- Add header with the possible configurations ---*/
#include "test_case_factory.hpp"

/*--- Include the headers with the numerical fluxes ---*/
#include "Hyperbolic_flux.hpp"
#include "SurfaceTension_flux.hpp"
#include "Relaxation_operator.hpp"

/*--- Specify the use of this namespace where we just store the indices ---*/
using namespace EquationData;

/*--- Define preprocessor to check whether to control data or not ---*/
#define DEBUG

/**
 * Auxiliary traits for type generality and flexibility
 */
template<std::size_t dim>
struct TwoScaleCapillarity_Traits {
  using Config    = samurai::MRConfig<dim, 2, 1, 0>;
  using mesh_type = samurai::MRMesh<Config>;

  using Field        = samurai::VectorField<mesh_type,
                                            double,
                                            EquationData::NVARS,
                                            false>;
  using Number       = samurai::Flux<Field>::Number; // Define the shortcut for the arithmetic type
  using Field_Scalar = samurai::ScalarField<mesh_type, Number>;
  using Field_Vect   = samurai::VectorField<mesh_type, Number, dim, false>;

  using EOS_type = LinearizedBarotropicEOS<Number>;

  using gradient_type   = decltype(samurai::make_gradient_order2<Field_Scalar>());
  using divergence_type = decltype(samurai::make_divergence_order2<Field_Vect>());
};

/**
 * This is the single declaration point for auxiliary fields. To add a
 * new auxiliary field:
 *   1. Add the member here.
 *   2. Initialise it in the solver class (make_*_field call).
 *   3. Add the corresponding resize() call where needed.
 *
 * @tparam Traits Traits struct defined in the solver header.
 */
template<typename Traits>
struct AuxiliaryFields {
  using Number       = typename Traits::Number;
  using Field_Scalar = typename Traits::Field_Scalar;
  using Field_Vect   = typename Traits::Field_Vect;
  using mesh_type    = typename Traits::mesh_type;

  using gradient_type   = typename Traits::gradient_type;
  using divergence_type = typename Traits::divergence_type;

  Field_Scalar p1, /*!< Phase 1 pressure */
               p2, /*!< Phase 2 pressure */
               p;  /*!< Mixture pressure */

  Field_Vect vel; /*!< Velocity field */
};

/**
 * This is the class for the simulation for a two-fluid model with capillarity
 */
template<std::size_t dim>
class TwoScaleCapillarity {
public:
  using Traits = TwoScaleCapillarity_Traits<dim>;

  using AuxFields = AuxiliaryFields<Traits>;

  using Config    = typename Traits::Config;
  using mesh_type = typename Traits::mesh_type;

  using Field        = typename Traits::Field;
  using Number       = typename Traits::Number;
  using Field_Scalar = typename Traits::Field_Scalar;
  using Field_Vect   = typename Traits::Field_Vect;

  using EOS_type = typename Traits::EOS_type;

  using gradient_type   = typename Traits::gradient_type;
  using divergence_type = typename Traits::divergence_type;

  /**
   * Default constructor. This will do nothing and basically will never be used
   */
  TwoScaleCapillarity() = default;

  /**
   * Class constructor with the arguments related to the grid, to the physics, and to the relaxation.
   * @param min_corner lower-left domain coordinates
   * @param max_corner upper-right domain coordinates
   * @param sim_param list of parameters for the configuration
   * @param eos_param parameters related to EOS (linearized barotropic EOS)
   * @param tc pointer to the test case configuration
   */
  TwoScaleCapillarity(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                      const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                      const Simulation_Parameters<Number>& sim_param,
                      const EOS_Parameters<Number>& eos_param,
                      std::unique_ptr<TestCaseBase<Traits, AuxFields>> tc);

  /**
   * Function which actually executes the temporal loop
   * @param num_flux_hyp name of flux for the hyperbolic subsystem. It can be read by parameter files (default value HLLC),
                         but since it is needed only here, we pass it as parameter rather than storing it
   * @param nfiles number of output files. It can be read by parameter files (default value 10),
                   but since it is needed only here, we pass it as parameter rather than storing it
   */
  void run(const std::string& num_flux_hyp,
           const std::size_t nfiles = 10);

  /**
   * Routine to save the results
   * @param suffix suffix to be added to the name
   * @param fields (variadic template) to specify the fields to be saved
   */
  template<class... Variables>
  void save(const std::string& suffix,
            const Variables&... fields);

private:
  /*--- Now we declare some relevant variables ---*/
  const samurai::Box<double, dim> box;

  mesh_type mesh; /*!< Variable to store the mesh */

  const Number t0; /*!< Initial time of the simulation */
  const Number Tf; /*!< Final time of the simulation */

  const Number sigma; /*!< Surface tension coefficient */

  bool apply_relax; /*!< Choose whether to apply or not the relaxation */

  Number cfl; /*!< Courant number of the simulation so as to compute the time step */
  Number dt; /*!< Time step */

  const Number mod_grad_alpha1_min; /*!< Minimum threshold for which not computing anymore the unit normal */

  const std::size_t max_Newton_iters; /*!< Maximum number of Newton iterations */

  double MR_param;      /*!< Multiresolution parameter */
  double MR_regularity; /*!< Multiresolution regularity */

  EOS_type EOS_phase1,
           EOS_phase2; // The two variables which take care of the
                       // barotropic EOS to compute the speed of sound

  std::unique_ptr<TestCaseBase<Traits, AuxFields>> test_case; /*!< Auxiliary variable to configurate the test case */

  HyperbolicFlux<Field> Hyperbolic_flux; /*!< Auxiliary variable to compute the contribution associated with hyperbolic operator */
  samurai::SurfaceTensionFlux<Field, Field_Vect> SurfaceTension_flux; /*!< Auxiliary variable to compute the contribution associated with surface tension */
  samurai::RelaxationOperator<Field> Relaxation_operator; /*!< Auxiliary variable to compute the contribution associated with source term (relaxation) */

  fs::path    path;     /*!< Auxiliary variable to store the output directory */
  std::string filename; /*!< Auxiliary variable to store the name of output */

  Field conserved_variables; /*!< The variable which stores the conserved variables,
                                  namely the varialbes for which we solve a PDE system */
  Field conserved_variables_tmp; /*!< Auxiliary field since we are solving a time-dependent PDE */

  /*--- Now we declare a bunch of fields which depend from the state, but it is useful
        to have it so as to avoid recomputation ---*/
  Field_Scalar alpha1,
               dalpha1,
               H;

  Field_Vect grad_alpha1,
             normal;

  AuxFields aux_fields;

  samurai::ScalarField<mesh_type, std::size_t> to_be_relaxed;
  samurai::ScalarField<mesh_type, std::size_t> Newton_iterations;

  gradient_type gradient;

  divergence_type divergence;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  /**
   * Auxiliary routine to compute gradient of volume fraction
   */
  void update_gradient();

  /**
   * Auxiliary routine to compute normals and curvature
   * @param update_grad specify if gradient has to be commputed as well (true by default)
   */
  void update_geometry(const bool update_grad = true);

  /**
   * Auxiliary routine to initialize the fields related to the mesh
   */
  void create_fields();

  /**
   * Auxiliary routine to resize all fields related to the mesh
   */
  void resize_all_fields();

  /**
   * Compute the estimate of the maximum eigenvalue
   */
  Number get_max_lambda();

  /**
   * Auxiliary routine to check if spurious values are present
   * @param flag specify after which stage we are doing this check, i.e.
                 after MR (value 1) or after convective subsystem (value 0, default)
   */
  void check_data(unsigned flag = 0);

  /**
   * Auxiliary routine to compute volume fraction from conserved variables
   */
  void recompute_alpha1();

  /**
   * Perform the finite volume stage (hyperbolic + capillarity subsystems)
   * @param numerical_flux_hyp numerical operator for convective subsystem
   * @param numerical_flux_cap numerical operator for capillarity subsystem
   */
  void perform_fv_stage(auto& numerical_flux_hyp,
                        auto& numerical_flux_st);

  /**
   * Apply the relaxation
   * @param relaxation_op numerical operator (cell-based scheme) for relaxation subsystem
   */
  void apply_relaxation(auto& relaxation_op);

  /**
   * Execute the postprocessing
   * @param time current time
   */
  void execute_postprocess(const Number time);
};

/************************************************************
******* START WITH THE IMPLEMENTATION OF THE CONSTRUCTOR ****
*************************************************************/

// Implement class constructor
//
template<std::size_t dim>
TwoScaleCapillarity<dim>::TwoScaleCapillarity(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                                              const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                                              const Simulation_Parameters<Number>& sim_param,
                                              const EOS_Parameters<Number>& eos_param,
                                              std::unique_ptr<TestCaseBase<Traits, AuxFields>> tc):
  box(min_corner, max_corner),
  t0(sim_param.t0), Tf(sim_param.Tf), sigma(sim_param.sigma),
  apply_relax(sim_param.apply_relaxation),
  cfl(sim_param.Courant), mod_grad_alpha1_min(sim_param.mod_grad_alpha1_min),
  max_Newton_iters(sim_param.max_Newton_iters),
  MR_param(sim_param.MR_param), MR_regularity(sim_param.MR_regularity),
  EOS_phase1(eos_param.p0_phase1, eos_param.rho0_phase1, eos_param.c0_phase1),
  EOS_phase2(eos_param.p0_phase2, eos_param.rho0_phase2, eos_param.c0_phase2),
  test_case(std::move(tc)),
  Hyperbolic_flux(create_hyperbolic_flux<Field>(sim_param.num_flux_hyp,
                                                EOS_phase1, EOS_phase2, sigma,
                                                sim_param.lambda, sim_param.atol_Newton, sim_param.rtol_Newton,
                                                max_Newton_iters)),
  SurfaceTension_flux(EOS_phase1, EOS_phase2, sigma,
                      sim_param.lambda, sim_param.atol_Newton, sim_param.rtol_Newton,
                      max_Newton_iters),
  Relaxation_operator(EOS_phase1, EOS_phase2, sigma,
                      sim_param.lambda, sim_param.atol_Newton, sim_param.rtol_Newton,
                      max_Newton_iters),
  path(sim_param.save_dir),
  gradient(samurai::make_gradient_order2<Field_Scalar>()),
  divergence(samurai::make_divergence_order2<Field_Vect>())
  {
    #ifdef SAMURAI_WITH_MPI
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if(rank == 0) {
        std::cout << "Initializing variables " << std::endl;
        std::cout << std::endl;
      }
    #else
      std::cout << "Initializing variables " << std::endl;
      std::cout << std::endl;
    #endif

    // Attach the fields to the mesh
    create_fields();

    // Initialize the fields
    if(sim_param.restart_file.empty()) {
      mesh = {box, sim_param.min_level, sim_param.max_level, {{false, true}}};

      resize_all_fields();
      SolverContext<Traits, AuxFields> ctx{mesh, conserved_variables,
                                           EOS_phase1, EOS_phase2,
                                           alpha1, grad_alpha1, normal, H,
                                           gradient, divergence,
                                           aux_fields};
      ctx.params["sigma"] = sigma;
      ctx.params["alpha_residual"] = sim_param.alpha_residual;
      ctx.params["mod_grad_alpha1_min"] = mod_grad_alpha1_min;
      test_case->setup(ctx);
      test_case->init_fn();
    }
    else {
      samurai::load(sim_param.restart_file, mesh, conserved_variables,
                                                  alpha1, grad_alpha1, normal, H,
                                                  aux_fields.vel,
                                                  aux_fields.p1, aux_fields.p2, aux_fields.p);
      // TO DO: Likely periodic bcs will not work
    }

    // Apply boundary conditions
    test_case->bc_fn();
  }

// Auxiliary routine to create the fields
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::create_fields() {
  conserved_variables = samurai::make_vector_field<Number, Field::n_comp>("conserved", mesh);

  conserved_variables_tmp = samurai::make_vector_field<Number, Field::n_comp>("conserved_tmp", mesh);

  aux_fields.vel = samurai::make_vector_field<Number, dim>("vel", mesh);

  aux_fields.p1 = samurai::make_scalar_field<Number>("p1", mesh);
  aux_fields.p2 = samurai::make_scalar_field<Number>("p2", mesh);
  aux_fields.p  = samurai::make_scalar_field<Number>("p", mesh);

  alpha1      = samurai::make_scalar_field<Number>("alpha1", mesh);
  grad_alpha1 = samurai::make_vector_field<Number, dim>("grad_alpha1", mesh);
  normal      = samurai::make_vector_field<Number, dim>("normal", mesh);
  H           = samurai::make_scalar_field<Number>("H", mesh);

  dalpha1 = samurai::make_scalar_field<Number>("dalpha1", mesh);

  to_be_relaxed     = samurai::make_scalar_field<std::size_t>("to_be_relaxed", mesh);
  Newton_iterations = samurai::make_scalar_field<std::size_t>("Newton_iterations", mesh);
}

// Resize the fields since now mesh has been created
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::resize_all_fields() {
  conserved_variables.resize();
  conserved_variables_tmp.resize();
  alpha1.resize();
  grad_alpha1.resize();
  normal.resize();
  H.resize();
  aux_fields.vel.resize();
  aux_fields.p1.resize();
  aux_fields.p2.resize();
  aux_fields.p.resize();
  dalpha1.resize();
  to_be_relaxed.resize();
  Newton_iterations.resize();
}

/************************************************************
******* FOCUS NOW ON THE AUXILIARY FUNCTIONS ****************
*************************************************************/

// Auxiliary routine to compute the gradient of phase 1 volume fraction
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::update_gradient() {
  samurai::update_ghost_mr(alpha1);
  grad_alpha1.fill(static_cast<Number>(0.0));
  gradient.apply(grad_alpha1, alpha1);
}

// Auxiliary routine to compute normals and curvature
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::update_geometry(const bool update_grad) {
  if(update_grad) {
    update_gradient();
  }

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              //const auto mod_grad_alpha1 = std::sqrt(xt::sum(grad_alpha1[cell]*grad_alpha1[cell])());
                              auto mod2_grad_alpha1_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                mod2_grad_alpha1_loc += grad_alpha1[cell][d]*grad_alpha1[cell][d];
                              }
                              const auto mod_grad_alpha1_loc = std::sqrt(mod2_grad_alpha1_loc);

                              if(mod_grad_alpha1_loc > mod_grad_alpha1_min) {
                                normal[cell] = grad_alpha1[cell]/mod_grad_alpha1_loc;
                              }
                              else {
                                for(std::size_t d = 0; d < dim; ++d) {
                                  normal[cell][d] = static_cast<Number>(nan(""));
                                }
                              }
                            }
                        );
  samurai::update_ghost_mr(normal);
  H = -divergence(normal);
}

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
typename TwoScaleCapillarity<dim>::Number
TwoScaleCapillarity<dim>::get_max_lambda() {
  auto local_res = static_cast<Number>(0.0);

  std::array<Number, dim> vel_loc;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch some variables used multiple times in order to exploit possible vectorization
                              const auto& local_conserved_variables = conserved_variables[cell];

                              const auto m1_loc = local_conserved_variables(M1_INDEX);
                              const auto m2_loc = local_conserved_variables(M2_INDEX);

                              // Compute the velocity along both horizontal and vertical direction
                              const auto rho_loc     = m1_loc + m2_loc;
                              const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel_loc[d] = local_conserved_variables(RHO_U_INDEX + d)*inv_rho_loc;
                              }

                              // Compute frozen speed of sound
                              const auto alpha1_loc       = alpha1[cell];
                              const auto rho1_loc         = m1_loc/alpha1_loc; // TODO: Add a check in case of zero volume fraction
                              const auto alpha2_loc       = static_cast<Number>(1.0) - alpha1_loc;
                              const auto rho2_loc         = m2_loc/alpha2_loc; // TODO: Add a check in case of zero volume fraction
                              const auto rhoc_squared_loc = m1_loc*EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)
                                                          + m2_loc*EOS_phase2.c_value(rho2_loc)*EOS_phase2.c_value(rho2_loc);
                              const auto c_loc            = std::sqrt(rhoc_squared_loc*inv_rho_loc);

                              // Add term due to surface tension
                              auto mod2_grad_alpha1_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                mod2_grad_alpha1_loc += grad_alpha1[cell][d]*grad_alpha1[cell][d];
                              }
                              const auto mod_grad_alpha1_loc = std::sqrt(mod2_grad_alpha1_loc);

                              const auto r = sigma*mod_grad_alpha1_loc/(rho_loc*c_loc*c_loc);

                              // Update eigenvalue estimate
                              for(std::size_t d = 0; d < dim; ++d) {
                                local_res = std::max(local_res,
                                                     std::abs(vel_loc[d]) + c_loc*(static_cast<Number>(1.0) +
                                                                                   static_cast<Number>(0.125)*r));
                              }
                            }
                        );

  #ifdef SAMURAI_WITH_MPI
    const double local_res_d = static_cast<double>(local_res);
    double global_res;
    MPI_Allreduce(&local_res_d, &global_res, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return static_cast<Number>(global_res);
  #else
    return local_res;
  #endif
}

// Auxiliary routine to check if negative values arise
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::check_data(unsigned flag) {
  std::string op;
  if(flag == 0) {
    op = "after hyperbolic operator (i.e. at the beginning of the relaxation)";
  }
  else {
    op = "after mesh adaptation";
  }

  auto check_positive_field = [&](const Number val, const auto& cell,
                                  const std::string& name,
                                  const Number low_tol = static_cast<Number>(0.0))
                                  {
                                    if(val < low_tol) {
                                      std::cerr << cell << std::endl;
                                      std::cerr << "Negative " + name + op << std::endl;
                                      save("_diverged", conserved_variables, alpha1);
                                      exit(1);
                                    }
                                    else if(std::isnan(val)) {
                                      std::cerr << cell << std::endl;
                                      std::cerr << "NaN " + name + op << std::endl;
                                      save("_diverged", conserved_variables, alpha1);
                                      exit(1);
                                    }
                                  };

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch local state
                              const auto& local_conserved_variables = conserved_variables[cell];

                              // Sanity check for alpha1
                              const auto alpha1_loc = alpha1[cell];
                              if(alpha1_loc < static_cast<Number>(0.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative volume fraction of phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1);
                                exit(1);
                              }
                              else if(alpha1_loc > static_cast<Number>(1.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Exceeding volume fraction of phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1);
                                exit(1);
                              }
                              else if(std::isnan(alpha1_loc)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN volume fraction of phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1);
                                exit(1);
                              }

                              // Sanity check for m1
                              check_positive_field(local_conserved_variables(M1_INDEX), cell,
                                                   "mass phase 1 ");

                              // Sanity check for m2
                              check_positive_field(local_conserved_variables(M2_INDEX), cell,
                                                   "mass phase 2 ");
                            }
                        );
}

// Auxiliary function to compute volume fraction from conserved variables
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::recompute_alpha1() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              const auto& local_conserved_variables = conserved_variables[cell];

                              alpha1[cell] = local_conserved_variables(RHO_ALPHA1_INDEX)/
                                             (local_conserved_variables(M1_INDEX) +
                                              local_conserved_variables(M2_INDEX));
                            }
                        );
}

/************************************************************
******* FOCUS NOW ON THE FINITE VOLUME ROUTINE **************
*************************************************************/

// Perform the finite volume stage (hyperbolic + capillarity subsystems)
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::perform_fv_stage(auto& numerical_flux_hyp,
                                                auto& numerical_flux_st) {
  // Convective operator
  #ifdef RELAX_RECONSTRUCTION
    samurai::update_ghost_mr(H);
  #endif
  try {
    conserved_variables_tmp = conserved_variables
                            - dt*numerical_flux_hyp(conserved_variables);
    samurai::swap(conserved_variables, conserved_variables_tmp);
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << std::endl;
    save("_diverged", conserved_variables, alpha1);
    exit(1);
  }

  // Recompute geometrical quantities
  recompute_alpha1();
  #ifdef DEBUG
    check_data();
  #endif
  update_gradient();

  // Capillarity contribution
  conserved_variables_tmp = conserved_variables
                          - dt*numerical_flux_st(grad_alpha1);
  samurai::swap(conserved_variables, conserved_variables_tmp);
}

/************************************************************
******* FOCUS NOW ON THE RELAXATION FUNCTIONS ***************
*************************************************************/

// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::apply_relaxation(auto& relaxation_op) {
  // Initialize the variables
  Newton_iterations.fill(0);
  dalpha1.fill(std::numeric_limits<Number>::infinity());

  // Loop of Newton method. Conceptually, a loop over cells followed by a Newton loop
  // over each cell would (could?) be more logic, but this would lead to issues to call 'update_geometry'
  bool global_relaxation_applied;
  for(std::size_t Newton_iter = 1; Newton_iter <= max_Newton_iters; ++Newton_iter) {
    Relaxation_operator.set_relaxation_applied(false);

    try {
      conserved_variables_tmp = relaxation_op(conserved_variables);
      samurai::swap(conserved_variables, conserved_variables_tmp);
    }
    catch(const std::exception& e) {
      std::cerr << e.what() << '\n';
      save("_diverged",
           conserved_variables,
           alpha1, dalpha1, grad_alpha1, normal, H,
           to_be_relaxed, Newton_iterations);
      exit(1);
    }

    // Recompute geometric quantities (curvature potentially changed in the Newton loop)
    update_geometry();

    // Check if we converged: reduce in case of MPI
    const bool local_relaxation_applied = Relaxation_operator.get_relaxation_applied();
    #ifdef SAMURAI_WITH_MPI
      mpi::communicator world;
      boost::mpi::all_reduce(world, local_relaxation_applied, global_relaxation_applied, std::logical_or<bool>());
    #else
      global_relaxation_applied = local_relaxation_applied;
    #endif
    // Converged: no cell requested further relaxation
    if(!global_relaxation_applied) {
      break;
    }
  }

  // Divergence check outside the loop
  if(global_relaxation_applied) {
    std::cerr << "Netwon method not converged in the post-hyperbolic relaxation" << std::endl;
    save("_diverged",
         conserved_variables,
         alpha1, dalpha1, grad_alpha1, normal, H,
         to_be_relaxed, Newton_iterations);
    exit(1);
  }
}

/************************************************************
******* FOCUS NOW ON THE POSTPROCESSING FUNCTIONS ***********
*************************************************************/

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void TwoScaleCapillarity<dim>::save(const std::string& suffix,
                                    const Variables&... fields) {
  if(!fs::exists(path)) {
    fs::create_directory(path);
  }

  samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, fields...);
  if(!(suffix.find("diverged") != std::string::npos)) {
    samurai::dump(path, fmt::format("{}{}", filename, "_restart"), mesh, fields...);
  }
}

/************************************************************************
**** IMPLEMENT THE FUNCTION THAT EFFECTIVELY SOLVES THE PROBLEM *********
*************************************************************************/

// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::run(const std::string& num_flux_hyp,
                                   const std::size_t nfiles) {
  // Default output arguments
  path = fs::current_path();
  filename = "liquid_column_no_mass_transfer";
  filename += "_" + num_flux_hyp;
  #ifdef ORDER_2
    filename += "_order2";
    #ifdef RELAX_RECONSTRUCTION
      filename += "_relaxed_reconstruction";
    #endif
  #else
    filename += "_order1";
  #endif

  const auto dt_save = Tf/static_cast<Number>(nfiles);

  // Auxiliary variables to save current state in case of second order
  #ifdef ORDER_2
    auto conserved_variables_old = samurai::make_vector_field<Number, Field::n_comp>("conserved_old", mesh);
  #endif

  // Create the flux variables
  #ifdef RELAX_RECONSTRUCTION
    auto numerical_flux_hyp = std::visit([this](auto& f)
                                               {
                                                 return f.make_flux(H);
                                               },
                                         Hyperbolic_flux);
  #else
    auto numerical_flux_hyp = std::visit([](auto& f)
                                           {
                                             return f.make_flux();
                                           },
                                         Hyperbolic_flux);
  #endif
  auto numerical_flux_st = SurfaceTension_flux.make_flux_capillarity();
  auto relaxation_op = Relaxation_operator.make_Newton_step_relaxation(H, dalpha1, alpha1,
                                                                       to_be_relaxed, Newton_iterations);

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_" + Utilities::unsigned_to_string(0) : "";
  save(suffix_init, conserved_variables,
                    alpha1, grad_alpha1, normal, H,
                    aux_fields.vel,
                    aux_fields.p1, aux_fields.p2, aux_fields.p);

  // Save mesh size (so as to compute time step)
  const auto dx = static_cast<Number>(mesh.cell_length(mesh.max_level()));
  using mesh_id_t = typename mesh_type::mesh_id_t;
  unsigned n_elements;
  #ifdef SAMURAI_WITH_MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const auto n_elements_per_subdomain = mesh[mesh_id_t::cells].nb_cells();
    MPI_Allreduce(&n_elements_per_subdomain, &n_elements, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0) {
      std::cout << "Number of initial elements = " <<  n_elements << std::endl;
      std::cout << std::endl;
    }
  #else
    n_elements = mesh[mesh_id_t::cells].nb_cells();
    std::cout << "Number of initial elements = " <<  n_elements << std::endl;
    std::cout << std::endl;
  #endif

  // Declare operators for MR
  /*auto prediction_fn = [&](auto& new_field, const auto& old_field) {
    return make_field_operator_function<TwoScaleCapillarity_prediction_op>(new_field, old_field);
  };*/
  auto MRadaptation = samurai::make_MRAdapt(/*prediction_fn,*/ conserved_variables);
  auto mra_config   = samurai::mra_config();
  mra_config.epsilon(MR_param);
  mra_config.regularity(MR_regularity);

  // Start the loop
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  auto t            = static_cast<Number>(t0);
  while(t != Tf) {
    // Apply mesh adaptation
    MRadaptation(mra_config);
    alpha1.resize();
    recompute_alpha1();
    #ifdef DEBUG
      check_data(1);
    #endif

    // Compute the time step
    grad_alpha1.resize();
    normal.resize();
    H.resize();
    update_gradient();
    dt = std::min(Tf - t, cfl*dx/get_max_lambda());
    t += dt;

    #ifdef SAMURAI_WITH_MPI
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if(rank == 0) {
        std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;
      }
    #else
      std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;
    #endif

    // Save current state in case of order 2
    #ifdef ORDER_2
      conserved_variables_old.resize();
      conserved_variables_old = conserved_variables;
    #endif

    // Solve the hyperbolic + capillarity subsytems
    conserved_variables_tmp.resize();
    #ifdef RELAX_RECONSTRUCTION
      update_geometry(false);
    #endif
    perform_fv_stage(numerical_flux_hyp, numerical_flux_st);

    // Apply relaxation
    if(apply_relax) {
      // Apply relaxation if desired, which will modify alpha1 and, consequently, for what
      // concerns next time step, rho_alpha1 (as well as grad_alpha1)
      dalpha1.resize();
      to_be_relaxed.resize();
      Newton_iterations.resize();
      update_geometry(false);
      apply_relaxation(relaxation_op);
    }

    /*--- Consider the second stage for the second order ---*/
    #ifdef ORDER_2
      // Solve the hyperbolic + capillarity subsytems
      perform_fv_stage(numerical_flux_hyp, numerical_flux_st);

      // Complete evaluation before applying relaxation
      conserved_variables_tmp = static_cast<Number>(0.5)*
                                (conserved_variables_old + conserved_variables);
      samurai::swap(conserved_variables, conserved_variables_tmp);

      // Apply the relaxation
      if(apply_relax) {
        recompute_alpha1();
        update_geometry();
        // Apply relaxation if desired, which will modify alpha1 and, consequently, for what
        // concerns next time step, rho_alpha1
        apply_relaxation(relaxation_op);
      }
      else {
        #ifdef RELAX_RECONSTRUCTION
          recompute_alpha1();
          update_geometry();
        #endif
      }
    #endif

    // Save the results
    if(t >= static_cast<Number>(nsave + 1)*dt_save || t == Tf) {
      // Resize all the fields not resized yet
      aux_fields.vel.resize();
      aux_fields.p1.resize();
      aux_fields.p2.resize();
      aux_fields.p.resize();

      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                                {
                                  // Pre-fetch local state
                                  const auto& local_conserved_variables = conserved_variables[cell];

                                  const auto m1_loc = local_conserved_variables(M1_INDEX);
                                  const auto m2_loc = local_conserved_variables(M2_INDEX);

                                  // Compute velocity
                                  const auto rho_loc     = m1_loc + m2_loc;
                                  const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;
                                  auto vel_loc           = aux_fields.vel[cell];
                                  for(std::size_t d = 0; d < dim; ++d) {
                                    vel_loc[d] = local_conserved_variables(RHO_U_INDEX + d)*inv_rho_loc;
                                  }

                                  // Compute pressure phase 1
                                  const auto alpha1_loc = local_conserved_variables(RHO_ALPHA1_INDEX)*inv_rho_loc;
                                  const auto rho1_loc   = m1_loc/alpha1_loc; // TODO: Add a check in case of zero volume fraction
                                  const auto p1_loc     = EOS_phase1.pres_value(rho1_loc);
                                  aux_fields.p1[cell]   = p1_loc;

                                  // Compute pressure phase 2
                                  const auto alpha2_loc = static_cast<Number>(1.0) - alpha1_loc;
                                  const auto rho2_loc   = m2_loc/alpha2_loc; // TODO: Add a check in case of zero volume fraction
                                  const auto p2_loc     = EOS_phase2.pres_value(rho2_loc);
                                  aux_fields.p2[cell]   = p2_loc;

                                  // Compute mixture pressure
                                  aux_fields.p[cell] = alpha1_loc*p1_loc
                                                     + alpha2_loc*p2_loc;
                                }
                            );

      // Perform the saving
      const std::string suffix = (nfiles != 1) ? "_ite_" + Utilities::unsigned_to_string(++nsave) : "";
      save(suffix, conserved_variables,
                   alpha1, grad_alpha1, normal, H,
                   aux_fields.vel,
                   aux_fields.p1, aux_fields.p2, aux_fields.p,
                   Newton_iterations);
    }
  }
}
