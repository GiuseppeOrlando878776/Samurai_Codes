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

/*--- Add header file for the multiresolution ---*/
#include <samurai/mr/adapt.hpp>
//#include "prediction.hpp"

/*--- Add header with auxiliary structs ---*/
#include "containers.hpp"

/*--- Add user implemented boundary condition ---*/
#include "user_bc.hpp"

/*--- Include the headers with the numerical fluxes ---*/
#include "Hyperbolic_flux.hpp"
#include "SurfaceTension_flux.hpp"
#include "Relaxation_operator.hpp"

/*--- Add header with auxiliary data structures for post-processing ---*/
#include "postprocessing.hpp"

/*--- Specify the use of this namespace where we just store the indices ---*/
using namespace EquationData;

/*--- Define preprocessor to check whether to control data or not ---*/
#define DEBUG

/**
 * This is the class for the simulation for the two-scale capillarity model
 */
template<std::size_t dim>
class TwoScaleCapillarity {
public:
  using Config    = samurai::MRConfig<dim, 2, 1, 0>;
  using mesh_type = samurai::MRMesh<Config>;
  using Field     = samurai::VectorField<mesh_type,
                                         double,
                                         EquationData::NVARS,
                                         false>;
  using Number    = samurai::Flux<Field>::Number; // Define the shortcut for the arithmetic type

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
   */
  TwoScaleCapillarity(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                      const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                      const Simulation_Parameters<Number>& sim_param,
                      const EOS_Parameters<Number>& eos_param);

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

  using Field_Scalar = samurai::ScalarField<mesh_type, Number>;
  using Field_Vect   = samurai::VectorField<mesh_type, Number, dim, false>;

  const Number t0; /*!< Initial time of the simulation */
  const Number Tf; /*!< Final time of the simulation */

  const Number sigma; /*!< Surface tension coefficient */

  bool apply_relax; /*!< Choose whether to apply or not the relaxation */

  const bool   mass_transfer; /*!< Choose wheter to apply or not the mass transfer */
  const Number alpha_d_max;   /*!< Maximum threshold of small-scale volume fraction */
  const Number alpha_l_min;   /*!< Minimum large-scale volume fraction to identify the mixture region */
  const Number alpha_l_max;   /*!< Maximum large-scale volume fraction to identify the mixture region */

  Number cfl; /*!< Courant number of the simulation so as to compute the time step */
  Number dt; /*!< Time step */

  const Number mod_grad_alpha_l_min; /*!< Minimum threshold for which not computing anymore the unit normal */

  const std::size_t max_Newton_iters; /*!< Maximum number of Newton iterations */

  double MR_param;      /*!< Multiresolution parameter */
  double MR_regularity; /*!< Multiresolution regularity */

  LinearizedBarotropicEOS<Number> EOS_phase_liq,
                                  EOS_phase_gas; // The two variables which take care of the
                                                 // barotropic EOS to compute the speed of sound

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
  Field_Scalar alpha_l,
               dalpha_l,
               p_liq,
               p_g,
               p;

  Field_Vect normal,
             grad_alpha_l;

  Field_Scalar alpha_d,
               alpha_l_bar,
               Sigma_d,
               H,
               H_bar,
               div_vel,
               H_filter;

  Field_Vect grad_alpha_d,
             vel,
             normal_bar,
             grad_alpha_l_bar;

  Field_Scalar Mach;

  samurai::ScalarField<mesh_type, std::size_t> to_be_relaxed;
  samurai::ScalarField<mesh_type, std::size_t> Newton_iterations;

  using gradient_type = decltype(samurai::make_gradient_order2<decltype(alpha_l)>());
  gradient_type gradient;

  using divergence_type = decltype(samurai::make_divergence_order2<decltype(normal)>());
  divergence_type divergence;

  std::optional<PostprocessWriter<Number>> postprocess_writer; /*!< Auxiliary output for post-processing */

  // Setup a filter to 'smooth' a field
  bool apply_filter; /*!< Choose whether to apply or not the filtering */

  Utilities::cell_based_scheme<Field_Scalar> filter;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  /**
   * Auxiliary routine to compute gradient of large-scale volume fraction
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
   * Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)
   * @param x0 x-center of liquid column
   * @param y0 y-center of liquid column
   * @param U0 "gas" component of horizontal velocity
   * @param U1 "liquid" component of horizontal velocity
   * @param V0 vertical velocity
   * @param R radius of the liquid column
   * @param eps_over_R initial interface thickness (w.r.t the radius)
   * @param alpha_residual initial 'residual' volume fraction
   */
  void init_variables(const Number x0, const Number y0,
                      const Number U0, const Number U1,
                      const Number V0,
                      const Number R, const Number eps_over_R,
                      const Number alpha_residual);

  /**
   * Auxiliary routine for the boundary conditions
   * @param U0 "gas" component of iniital horizontal velocity
   * @param V0 vertical velocity
   * @param alpha_residual initial 'residual' volume fraction
   */
  void apply_bcs(const Number U0,
                 const Number V0,
                 const Number alpha_residual);

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
   * Auxiliary routine to compute large-scale volume fraction from conserved variables
   */
  void recompute_alpha_l();

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
                                              const EOS_Parameters<Number>& eos_param):
  box(min_corner, max_corner),
  t0(sim_param.t0), Tf(sim_param.Tf), sigma(sim_param.sigma),
  apply_relax(sim_param.apply_relaxation),
  mass_transfer(sim_param.mass_transfer),
  alpha_d_max(sim_param.alpha_d_max),
  alpha_l_min(sim_param.alpha_l_min), alpha_l_max(sim_param.alpha_l_max),
  cfl(sim_param.Courant),
  mod_grad_alpha_l_min(sim_param.mod_grad_alpha_l_min),
  max_Newton_iters(sim_param.max_Newton_iters),
  MR_param(sim_param.MR_param), MR_regularity(sim_param.MR_regularity),
  EOS_phase_liq(eos_param.p0_phase1, eos_param.rho0_phase1, eos_param.c0_phase1),
  EOS_phase_gas(eos_param.p0_phase2, eos_param.rho0_phase2, eos_param.c0_phase2),
  Hyperbolic_flux(create_hyperbolic_flux<Field>(sim_param.num_flux_hyp,
                                                EOS_phase_liq, EOS_phase_gas, sigma,
                                                sim_param.lambda, sim_param.atol_Newton, sim_param.rtol_Newton,
                                                max_Newton_iters)),
  SurfaceTension_flux(EOS_phase_liq, EOS_phase_gas, sigma,
                      sim_param.lambda, sim_param.atol_Newton, sim_param.rtol_Newton,
                      max_Newton_iters),
  Relaxation_operator(EOS_phase_liq, EOS_phase_gas, sigma,
                      sim_param.Hmax, sim_param.kappa,
                      alpha_d_max, alpha_l_min, alpha_l_max,
                      sim_param.lambda, sim_param.atol_Newton, sim_param.rtol_Newton,
                      max_Newton_iters),
  path(sim_param.save_dir),
  gradient(samurai::make_gradient_order2<decltype(alpha_l)>()),
  divergence(samurai::make_divergence_order2<decltype(normal)>()),
  apply_filter(sim_param.apply_filter),
  filter(samurai::make_cell_based_scheme<Utilities::cfg_filter<Field_Scalar>>())
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

    // Complete creation of filter
    filter.stencil() = {{-1, 1}, {0, 1}, {1, 1},
                        {-1, 0}, {0, 0}, {1, 0},
                        {-1, -1}, {0, -1}, {1, -1}};

    filter.coefficients_func() = [](samurai::StencilCoeffs<Utilities::cfg_filter<Field_Scalar>>& coeffs, Number)
                                    {
                                      coeffs[0] = static_cast<Number>(1.0/16.0);
                                      coeffs[1] = static_cast<Number>(1.0/8.0);
                                      coeffs[2] = static_cast<Number>(1.0/16.0);
                                      coeffs[3] = static_cast<Number>(1.0/8.0);
                                      coeffs[4] = static_cast<Number>(1.0/4.0);
                                      coeffs[5] = static_cast<Number>(1.0/8.0);
                                      coeffs[6] = static_cast<Number>(1.0/16.0);
                                      coeffs[7] = static_cast<Number>(1.0/8.0);
                                      coeffs[8] = static_cast<Number>(1.0/16.0);
                                    };

    // Attach the fields to the mesh
    create_fields();

    // Initialize the fields
    if(sim_param.restart_file.empty()) {
      mesh = {box, sim_param.min_level, sim_param.max_level, {{false, true}}};

      init_variables(sim_param.x0, sim_param.y0,
                     sim_param.U0, sim_param.U1,
                     sim_param.V0,
                     sim_param.R, sim_param.eps_over_R,
                     sim_param.alpha_residual);
    }
    else {
      samurai::load(sim_param.restart_file, mesh, conserved_variables,
                                                  alpha_l, grad_alpha_l, normal, H,
                                                  p_liq, p_g, p,
                                                  alpha_d, Sigma_d, grad_alpha_d,
                                                  vel, div_vel,
                                                  alpha_l_bar, grad_alpha_l_bar, H_bar,
                                                  Mach);
      // TO DO: Likely periodic bcs will not work
    }

    // Apply boundary conditions
    apply_bcs(sim_param.U0, sim_param.V0, sim_param.alpha_residual);
  }

// Auxiliary routine to create the fields
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::create_fields() {
  conserved_variables = samurai::make_vector_field<Number, Field::n_comp>("conserved", mesh);

  conserved_variables_tmp = samurai::make_vector_field<Number, Field::n_comp>("conserved_tmp", mesh);

  alpha_l      = samurai::make_scalar_field<Number>("alpha_l", mesh);
  grad_alpha_l = samurai::make_vector_field<Number, dim>("grad_alpha_l", mesh);
  normal       = samurai::make_vector_field<Number, dim>("normal", mesh);
  H            = samurai::make_scalar_field<Number>("H", mesh);

  dalpha_l = samurai::make_scalar_field<Number>("dalpha_l", mesh);

  p_liq = samurai::make_scalar_field<Number>("p_liq", mesh);
  p_g   = samurai::make_scalar_field<Number>("p_g", mesh);
  p     = samurai::make_scalar_field<Number>("p", mesh);

  alpha_d      = samurai::make_scalar_field<Number>("alpha_d", mesh);
  grad_alpha_d = samurai::make_vector_field<Number, dim>("grad_alpha_d", mesh);
  Sigma_d      = samurai::make_scalar_field<Number>("Sigma_d", mesh);
  vel          = samurai::make_vector_field<Number, dim>("vel", mesh);
  div_vel      = samurai::make_scalar_field<Number>("div_vel", mesh);

  alpha_l_bar      = samurai::make_scalar_field<Number>("alpha_l_bar", mesh);
  grad_alpha_l_bar = samurai::make_vector_field<Number, dim>("grad_alpha_l_bar", mesh);
  normal_bar       = samurai::make_vector_field<Number, dim>("normal_bar", mesh);
  H_bar            = samurai::make_scalar_field<Number>("H_bar", mesh);
  H_filter         = samurai::make_scalar_field<Number>("H_unfiltered", mesh);

  Mach = samurai::make_scalar_field<Number>("Mach", mesh);

  to_be_relaxed     = samurai::make_scalar_field<std::size_t>("to_be_relaxed", mesh);
  Newton_iterations = samurai::make_scalar_field<std::size_t>("Newton_iterations", mesh);
}

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::init_variables(const Number x0, const Number y0,
                                              const Number U0, const Number U1,
                                              const Number V0,
                                              const Number R, const Number eps_over_R,
                                              const Number alpha_residual) {
  // Resize the fields since now mesh has been created
  conserved_variables.resize();
  conserved_variables_tmp.resize();
  alpha_l.resize();
  grad_alpha_l.resize();
  normal.resize();
  H.resize();
  dalpha_l.resize();
  p_liq.resize();
  p_g.resize();
  p.resize();
  alpha_d.resize();
  grad_alpha_d.resize();
  Sigma_d.resize();
  vel.resize();
  div_vel.resize();
  alpha_l_bar.resize();
  grad_alpha_l_bar.resize();
  normal_bar.resize();
  H_bar.resize();
  H_filter.resize();
  Mach.resize();
  to_be_relaxed.resize();
  Newton_iterations.resize();

  // Declare some constant parameters associated with the initial state
  const auto eps_R = eps_over_R*R;

  // Initialize the large-scale volume fraction to define the liquid column with a loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Set large-scale volume fraction
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);
                              const auto y      = static_cast<Number>(center[1]);
                              const auto r      = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
                              const auto w      = (r >= R && r < R + eps_R) ?
                                                  std::exp(static_cast<Number>(2.0)*
                                                           (r - R)*(r - R)/(eps_R*eps_R)*
                                                           ((r - R)*(r - R)/(eps_R*eps_R) - static_cast<Number>(3.0))/
                                                           (((r - R)*(r - R)/(eps_R*eps_R) - static_cast<Number>(1.0))*
                                                            ((r - R)*(r - R)/(eps_R*eps_R) - static_cast<Number>(1.0)))) :
                                                  ((r < R) ? static_cast<Number>(1.0) :
                                                             static_cast<Number>(0.0));

                              alpha_l[cell] = std::min(std::max(alpha_residual, w),
                                                       static_cast<Number>(1.0) - alpha_residual);
                            }
                        );

  // Compute the geometrical quantities
  update_geometry();

  // Loop over a cell to complete the remaining variables
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Set small-scale variables
                              alpha_d[cell]                          = static_cast<Number>(0.0);
                              conserved_variables[cell](RHO_Z_INDEX) = static_cast<Number>(0.0);
                              const auto rho_liq_ref                 = EOS_phase_liq.get_rho0();
                              Sigma_d[cell]                          = conserved_variables[cell](RHO_Z_INDEX)/std::cbrt(rho_liq_ref*rho_liq_ref);
                              conserved_variables[cell](Md_INDEX)    = alpha_d[cell]*rho_liq_ref;

                              // Recompute geometric locations to set partial masses
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);
                              const auto y      = static_cast<Number>(center[1]);
                              const auto r      = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                              // Set mass large-scale liquid phase
                              if(r >= R + eps_R) {
                                p_liq[cell] = EOS_phase_liq.get_p0();
                              }
                              else {
                                p_liq[cell] = EOS_phase_gas.get_p0();
                                if(r >= R && r < R + eps_R && !std::isnan(H[cell])) {
                                  p_liq[cell] += sigma*H[cell];
                                }
                                else {
                                  p_liq[cell] += sigma/R;
                                }
                              }
                              const auto rho_liq_loc = EOS_phase_liq.rho_value(p_liq[cell]);

                              conserved_variables[cell](Ml_INDEX) = alpha_l[cell]*rho_liq_loc;

                              // Set mass gas phase
                              p_g[cell]            = EOS_phase_gas.get_p0();
                              const auto rho_g_loc = EOS_phase_gas.rho_value(p_g[cell]);

                              const auto alpha_liq_loc = alpha_l[cell] + alpha_d[cell];
                              const auto alpha_g_loc   = static_cast<Number>(1.0) - alpha_liq_loc;
                              conserved_variables[cell](Mg_INDEX) = alpha_g_loc*rho_g_loc;

                              // Save mixture pressure for post-processing
                              p[cell] = alpha_liq_loc*p_liq[cell]
                                      + alpha_g_loc*p_g[cell]
                                      - static_cast<Number>(2.0/3.0)*sigma*Sigma_d[cell];

                              // Set conserved variable associated with large-scale volume fraction
                              const auto rho_loc = conserved_variables[cell](Ml_INDEX)
                                                 + conserved_variables[cell](Mg_INDEX)
                                                 + conserved_variables[cell](Md_INDEX);

                              conserved_variables[cell](RHO_ALPHA_l_INDEX) = rho_loc*alpha_l[cell];

                              // Set momentum
                              conserved_variables[cell](RHO_U_INDEX)     = conserved_variables[cell](Ml_INDEX)*U1
                                                                         + conserved_variables[cell](Mg_INDEX)*U0;
                              conserved_variables[cell](RHO_U_INDEX + 1) = rho_loc*V0;

                              // Save velocity for post-processing
                              auto norm2_vel_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel[cell][d] = conserved_variables[cell](RHO_U_INDEX + d)/rho_loc;
                                norm2_vel_loc += vel[cell][d]*vel[cell][d];
                              }

                              // Save 'bar' volume fraction for post-processing
                              alpha_l_bar[cell] = alpha_l[cell]/(static_cast<Number>(1.0) - alpha_d[cell]);

                              // Save Mach number for post-processing
                              const auto Y_g_loc   = conserved_variables[cell](Mg_INDEX)/rho_loc;
                              const auto c_liq_loc = EOS_phase_liq.c_value(rho_liq_loc);
                              const auto c_g_loc   = EOS_phase_gas.c_value(rho_g_loc);
                              const auto cf_loc    = std::sqrt((static_cast<Number>(1.0) - Y_g_loc)*c_liq_loc*c_liq_loc +
                                                               Y_g_loc*c_g_loc*c_g_loc -
                                                               static_cast<Number>(2.0/9.0)*sigma*Sigma_d[cell]/rho_loc);
                              Mach[cell]           = std::sqrt(norm2_vel_loc)/cf_loc;
                            }
                        );

  // Set useful small-scale related fields
  samurai::update_ghost_mr(alpha_d);
  grad_alpha_d.fill(static_cast<Number>(0.0));
  gradient.apply(grad_alpha_d, alpha_d);

  samurai::update_ghost_mr(vel);
  div_vel.fill(static_cast<Number>(0.0));
  divergence.apply(div_vel, vel);

  // Set auxiliary gradient alpha_l_bar volume fraction
  samurai::update_ghost_mr(alpha_l_bar);
  grad_alpha_l_bar.fill(static_cast<Number>(0.0));
  gradient.apply(grad_alpha_l_bar, alpha_l_bar);
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              const auto& grad_alpha_l_bar_loc = grad_alpha_l_bar[cell];
                              auto mod2_grad_alpha_l_bar_loc   = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                mod2_grad_alpha_l_bar_loc += grad_alpha_l_bar_loc[d]*grad_alpha_l_bar_loc[d];
                              }
                              const auto mod_grad_alpha_l_bar_loc = std::sqrt(mod2_grad_alpha_l_bar_loc);

                              if(mod_grad_alpha_l_bar_loc > mod_grad_alpha_l_min) {
                                normal_bar[cell] = grad_alpha_l_bar_loc/mod_grad_alpha_l_bar_loc;
                              }
                              else {
                                for(std::size_t d = 0; d < dim; ++d) {
                                  normal_bar[cell][d] = static_cast<Number>(nan(""));
                                }
                              }
                            }
                        );
  samurai::update_ghost_mr(normal_bar);
  H_bar = -divergence(normal_bar);
}

// Auxiliary routine to impose the boundary conditions
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::apply_bcs(const Number U0,
                                         const Number V0,
                                         const Number alpha_residual) {
  const samurai::DirectionVector<dim> left = {-1, 0};
  samurai::make_bc<Default>(conserved_variables,
                            Inlet(conserved_variables, U0, V0, alpha_residual,
                                  static_cast<Number>(0.0),
                                  static_cast<Number>(0.0)))->on(left);

  const samurai::DirectionVector<dim> right = {1, 0};
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables,
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0))->on(right);
}

/************************************************************
******* FOCUS NOW ON THE AUXILIARY FUNCTIONS ****************
*************************************************************/

// Auxiliary routine to compute the gradient of large-scale volume fraction
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::update_gradient() {
  samurai::update_ghost_mr(alpha_l);
  grad_alpha_l.fill(static_cast<Number>(0.0));
  gradient.apply(grad_alpha_l, alpha_l);
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
                              const auto& grad_alpha_l_loc = grad_alpha_l[cell];
                              auto mod2_grad_alpha_l_loc   = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                mod2_grad_alpha_l_loc += grad_alpha_l_loc[d]*grad_alpha_l_loc[d];
                              }
                              const auto mod_grad_alpha_l_loc = std::sqrt(mod2_grad_alpha_l_loc);

                              if(mod_grad_alpha_l_loc > mod_grad_alpha_l_min) {
                                normal[cell] = grad_alpha_l_loc/mod_grad_alpha_l_loc;
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

  // Apply the filter to the curvature
  if(apply_filter) {
    filter.apply(H_filter, H);
    samurai::swap(H_filter, H);
  }
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

                              const auto m_l_loc = local_conserved_variables(Ml_INDEX);
                              const auto m_g_loc = local_conserved_variables(Mg_INDEX);
                              const auto m_d_loc = local_conserved_variables(Md_INDEX);

                              const auto alpha_l_loc = alpha_l[cell];

                              // Compute the velocity along all the directions
                              const auto m_liq_loc   = m_l_loc + m_d_loc;
                              const auto rho_loc     = m_liq_loc + m_g_loc;
                              const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel_loc[d] = local_conserved_variables(RHO_U_INDEX + d)*inv_rho_loc;
                              }

                              // Compute frozen speed of sound
                              const auto alpha_d_loc   = alpha_l_loc*m_d_loc/m_l_loc; // TODO: Add a check in case of zero volume fraction
                              const auto alpha_liq_loc = alpha_l_loc + alpha_d_loc;
                              const auto rho_liq_loc   = m_liq_loc/alpha_liq_loc; // TODO: Add a check in case of zero volume fraction
                              const auto alpha_g_loc   = static_cast<Number>(1.0) - alpha_liq_loc;
                              const auto rho_g_loc     = m_g_loc/alpha_g_loc; // TODO: Add a check in case of zero volume fraction
                              const auto Y_g_loc       = m_g_loc*inv_rho_loc;
                              const auto Sigma_d_loc   = local_conserved_variables(RHO_Z_INDEX)/std::cbrt(rho_liq_loc*rho_liq_loc);
                              const auto c_liq_loc     = EOS_phase_liq.c_value(rho_liq_loc);
                              const auto c_g_loc       = EOS_phase_gas.c_value(rho_g_loc);
                              const auto cf_loc        = std::sqrt((static_cast<Number>(1.0) - Y_g_loc)*c_liq_loc*c_liq_loc +
                                                                   Y_g_loc*c_g_loc*c_g_loc -
                                                                   static_cast<Number>(2.0/9.0)*sigma*Sigma_d_loc*inv_rho_loc);

                              // Add term due to surface tension
                              const auto& grad_alpha_l_loc = grad_alpha_l[cell];
                              auto mod2_grad_alpha_l_loc   = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                mod2_grad_alpha_l_loc += grad_alpha_l_loc[d]*grad_alpha_l_loc[d];
                              }
                              const auto mod_grad_alpha_l_loc = std::sqrt(mod2_grad_alpha_l_loc);

                              const auto r = sigma*mod_grad_alpha_l_loc/(rho_loc*cf_loc*cf_loc);

                              // Update eigenvalue estimate
                              for(std::size_t d = 0; d < dim; ++d) {
                                local_res = std::max(local_res,
                                                     std::abs(vel_loc[d]) + cf_loc*std::sqrt(static_cast<Number>(1.0) + r));
                              }
                            }
                        );

  return Utilities::mpi_reduce_max(local_res);
}

// Auxiliary function to check if spurious values are present
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
                                      save("_diverged", conserved_variables, alpha_l);
                                      exit(1);
                                    }
                                    else if(std::isnan(val)) {
                                      std::cerr << cell << std::endl;
                                      std::cerr << "NaN " + name + op << std::endl;
                                      save("_diverged", conserved_variables, alpha_l);
                                      exit(1);
                                    }
                                  };

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch local state
                              const auto& local_conserved_variables = conserved_variables[cell];

                              // Sanity check for alpha_l
                              const auto alpha_l_loc = alpha_l[cell];
                              if(alpha_l_loc < static_cast<Number>(0.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative volume fraction large-scale liquid " + op << std::endl;
                                save("_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }
                              else if(alpha_l_loc > static_cast<Number>(1.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Exceeding volume fraction large-scale liquid " + op << std::endl;
                                save("_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }
                              else if(std::isnan(alpha_l_loc)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN volume fraction large-scale liquid " + op << std::endl;
                                save("_diverged", conserved_variables, alpha_l);
                                exit(1);
                              }

                              // Sanity check for m_l
                              check_positive_field(local_conserved_variables(Ml_INDEX), cell,
                                                   "mass large-scale liquid ");

                              // Sanity check for m_g
                              check_positive_field(local_conserved_variables(Mg_INDEX), cell,
                                                   "mass gas phase ");

                              // Sanity check for m_d
                              check_positive_field(local_conserved_variables(Md_INDEX), cell,
                                                   "mass small-scale liquid ");

                              // Sanity check for z (the transported variable related to small-scale IAD)
                              check_positive_field(local_conserved_variables(RHO_Z_INDEX), cell,
                                                   "interface area small-scale liquid ");
                            }
                        );
}

// Auxiliary function to compute large-scale volume fraction from conserved variables
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::recompute_alpha_l() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              const auto& local_conserved_variables = conserved_variables[cell];

                              alpha_l[cell] = local_conserved_variables(RHO_ALPHA_l_INDEX)/
                                              (local_conserved_variables(Ml_INDEX) +
                                               local_conserved_variables(Mg_INDEX) +
                                               local_conserved_variables(Md_INDEX));
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
    save("_diverged", conserved_variables, alpha_l);
    exit(1);
  }

  // Recompute geometrical quantities
  recompute_alpha_l();
  #ifdef DEBUG
    check_data();
  #endif
  update_gradient();

  // Capillarity contribution
  conserved_variables_tmp = conserved_variables
                          - dt*numerical_flux_st(grad_alpha_l);
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
  dalpha_l.fill(std::numeric_limits<Number>::infinity());
  Relaxation_operator.set_mass_transfer_NR(mass_transfer); // In principle we might think to disable it after a certain
                                                           // number of iterations (as in Arthur's code), not done here.

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
      std::cerr << e.what() << std::endl;
      save("_diverged",
           conserved_variables,
           alpha_l, dalpha_l, grad_alpha_l, normal, H,
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

  // Newton cycle diverged
  if(global_relaxation_applied) {
    std::cerr << "Newton method not converged in the post-hyperbolic relaxation" << std::endl;
    save("_diverged",
         conserved_variables,
         alpha_l, dalpha_l, grad_alpha_l, normal, H,
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

// Execute postprocessing
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::execute_postprocess(const Number time) {
  // Auxiliary struct for relevant integral quantities
  IntegralQuantities<Number> local_q;

  // Recompute auxiliary fields and perform calculations
  alpha_l_bar.resize();
  alpha_d.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Save alpha_d and alpha_l_bar
                              const auto& local_conserved_variables = conserved_variables[cell];

                              const auto alpha_l_loc = alpha_l[cell];
                              const auto alpha_d_loc = alpha_l_loc*local_conserved_variables(Md_INDEX)/local_conserved_variables(Ml_INDEX);
                              alpha_d[cell]          = alpha_d_loc;
                              alpha_l_bar[cell]      = alpha_l_loc/(static_cast<Number>(1.0) - alpha_d_loc);
                            }
                        );
  samurai::update_ghost_mr(alpha_l_bar, alpha_d);
  grad_alpha_l_bar.resize();
  grad_alpha_l_bar.fill(static_cast<Number>(0.0));
  gradient.apply(grad_alpha_l_bar, alpha_l_bar);
  grad_alpha_d.resize();
  grad_alpha_d.fill(static_cast<Number>(0.0));
  gradient.apply(grad_alpha_d, alpha_d);

  Sigma_d.resize();
  p_liq.resize();
  p_g.resize();
  p.resize();
  vel.resize();
  Mach.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch some variables used multiple times in order to exploit possible vectorization
                              const auto& local_conserved_variables = conserved_variables[cell];

                              const auto m_l_loc = local_conserved_variables(Ml_INDEX);
                              const auto m_g_loc = local_conserved_variables(Mg_INDEX);
                              const auto m_d_loc = local_conserved_variables(Md_INDEX);

                              const auto alpha_l_loc = alpha_l[cell];
                              const auto alpha_d_loc = alpha_d[cell];

                              const auto& grad_alpha_l_loc     = grad_alpha_l[cell];
                              const auto& grad_alpha_d_loc     = grad_alpha_d[cell];
                              const auto& grad_alpha_l_bar_loc = grad_alpha_l_bar[cell];

                              // Compue H_lig
                              if(alpha_l_loc > alpha_l_min && alpha_l_loc < alpha_l_max &&
                                 alpha_d_loc < alpha_d_max) {
                                local_q.H_lig = std::max(H[cell], local_q.H_lig);
                              }

                              // Compute pressures
                              const auto m_liq_loc     = m_l_loc + m_d_loc;
                              const auto alpha_liq_loc = alpha_l_loc + alpha_d_loc;
                              const auto rho_liq_loc   = m_liq_loc/alpha_liq_loc; // TODO: Add a check in case of zero volume fraction
                              const auto p_liq_loc     = EOS_phase_liq.pres_value(rho_liq_loc);
                              p_liq[cell]              = p_liq_loc;

                              const auto alpha_g_loc = static_cast<Number>(1.0) - alpha_liq_loc;
                              const auto rho_g_loc   = m_g_loc/alpha_g_loc; // TODO: Add a check in case of zero volume fraction
                              const auto p_g_loc     = EOS_phase_gas.pres_value(rho_g_loc);
                              p_g[cell]              = p_g_loc;

                              const auto Sigma_d_loc = local_conserved_variables(RHO_Z_INDEX)/std::cbrt(rho_liq_loc*rho_liq_loc);
                              Sigma_d[cell]          = Sigma_d_loc;

                              p[cell] = alpha_liq_loc*p_liq_loc
                                      + alpha_g_loc*p_g_loc
                                      - static_cast<Number>(2.0/3.0)*sigma*Sigma_d_loc;

                              // Compute geometric Euclidean norms
                              auto mod2_grad_alpha_l_loc     = static_cast<Number>(0.0);
                              auto mod2_grad_alpha_d_loc     = static_cast<Number>(0.0);
                              auto mod2_grad_alpha_liq_loc   = static_cast<Number>(0.0);
                              auto mod2_grad_alpha_l_bar_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                mod2_grad_alpha_l_loc     += grad_alpha_l_loc[d]*grad_alpha_l_loc[d];
                                mod2_grad_alpha_d_loc     += grad_alpha_d_loc[d]*grad_alpha_d_loc[d];
                                mod2_grad_alpha_liq_loc   += (grad_alpha_l_loc[d] + grad_alpha_d_loc[d])*
                                                             (grad_alpha_l_loc[d] + grad_alpha_d_loc[d]);
                                mod2_grad_alpha_l_bar_loc += grad_alpha_l_bar_loc[d]*grad_alpha_l_bar_loc[d];
                              }
                              const auto mod_grad_alpha_l_loc     = std::sqrt(mod2_grad_alpha_l_loc);
                              const auto mod_grad_alpha_d_loc     = std::sqrt(mod2_grad_alpha_d_loc);
                              const auto mod_grad_alpha_liq_loc   = std::sqrt(mod2_grad_alpha_liq_loc);
                              const auto mod_grad_alpha_l_bar_loc = std::sqrt(mod2_grad_alpha_l_bar_loc);

                              // Compute the total energy (Hamiltonian)
                              const auto rho_loc     = m_liq_loc + m_g_loc;
                              const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;
                              auto norm2_vel_loc     = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                const auto vel_d_loc = local_conserved_variables(RHO_U_INDEX + d)*inv_rho_loc;
                                vel[cell][d] = vel_d_loc;
                                norm2_vel_loc += vel_d_loc*vel_d_loc;
                              }
                              const auto e_liq_loc = m_liq_loc*EOS_phase_liq.e_value(rho_liq_loc);
                              const auto e_g_loc   = m_g_loc*EOS_phase_gas.e_value(rho_g_loc);

                              const auto Etot_loc = static_cast<Number>(0.5)*rho_loc*norm2_vel_loc
                                                  + e_liq_loc + e_g_loc
                                                  + sigma*(mod_grad_alpha_l_loc + Sigma_d_loc);

                              // Save Mach number for post-processing
                              const auto Y_g_loc   = m_g_loc*inv_rho_loc;
                              const auto c_liq_loc = EOS_phase_liq.c_value(rho_liq_loc);
                              const auto c_g_loc   = EOS_phase_gas.c_value(rho_g_loc);
                              const auto cf_loc    = std::sqrt((static_cast<Number>(1.0) - Y_g_loc)*c_liq_loc*c_liq_loc +
                                                                Y_g_loc*c_g_loc*c_g_loc -
                                                                static_cast<Number>(2.0/9.0)*sigma*Sigma_d_loc*inv_rho_loc);
                              Mach[cell]           = std::sqrt(norm2_vel_loc)/cf_loc;

                              // Compute the integral quantities
                              auto cell_volume = static_cast<Number>(cell.length);
                              for(std::size_t d = 1; d < dim; ++d) {
                                cell_volume *= static_cast<Number>(cell.length);
                              }

                              local_q.m_l_int += m_l_loc*cell_volume;
                              local_q.m_d_int += m_d_loc*cell_volume;
                              local_q.alpha_l_int += alpha_l_loc*cell_volume;
                              local_q.grad_alpha_l_int += mod_grad_alpha_l_loc*cell_volume;
                              local_q.Sigma_d_int += Sigma_d_loc*cell_volume;
                              local_q.grad_alpha_d_int += mod_grad_alpha_d_loc*cell_volume;
                              local_q.alpha_d_int += alpha_d_loc*cell_volume;
                              local_q.grad_alpha_liq_int += mod_grad_alpha_liq_loc*cell_volume;
                              local_q.alpha_l_bar_int += alpha_l_bar[cell]*cell_volume;
                              local_q.grad_alpha_l_bar_int += mod_grad_alpha_l_bar_loc*cell_volume;
                              local_q.Etot_int += Etot_loc*cell_volume;
                            }
                        );

  // Save the data
  postprocess_writer->write(time, local_q);
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
  filename = "liquid_column";
  filename += "_" + num_flux_hyp;

  #ifdef ORDER_2
    filename += "_order2";
    #ifdef RELAX_RECONSTRUCTION
      filename += "_relaxed_reconstruction";
    #endif
  #else
    filename += "_order1";
  #endif

  if(mass_transfer) {
    filename += "_mass_transfer";
  }
  else {
    filename += "_no_mass_transfer";
  }

  const auto dt_save = Tf/static_cast<Number>(nfiles);

  // Auxiliary variables to save current state in case of second order
  #ifdef ORDER_2
    auto conserved_variables_old = samurai::make_vector_field<Number, Field::n_comp>("conserved_old", mesh);
  #endif

  // Create the flux variables
  #ifdef RELAX_RECONSTRUCTION
    auto numerical_flux_hyp = std::visit([this](auto& f)
                                               {
                                                 return f.make_two_scale_capillarity(H);
                                               },
                                         Hyperbolic_flux);
  #else
    auto numerical_flux_hyp = std::visit([](auto& f)
                                           {
                                              return f.make_two_scale_capillarity();
                                           },
                                         Hyperbolic_flux);
  #endif
  auto numerical_flux_st = SurfaceTension_flux.make_two_scale_capillarity();
  auto relaxation_op = Relaxation_operator.make_Newton_step_relaxation(H, dalpha_l, alpha_l,
                                                                       to_be_relaxed, Newton_iterations,
                                                                       grad_alpha_l);

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_" + Utilities::unsigned_to_string(0) : "";
  save(suffix_init, conserved_variables,
                    alpha_l, grad_alpha_l, normal, H,
                    p_liq, p_g, p,
                    alpha_d, Sigma_d, grad_alpha_d,
                    vel, div_vel,
                    alpha_l_bar, grad_alpha_l_bar, H_bar,
                    Mach);
  postprocess_writer.emplace(path);
  auto t = static_cast<Number>(t0);
  execute_postprocess(t);

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
  while(t != Tf) {
    // Apply mesh adaptation
    MRadaptation(mra_config);
    alpha_l.resize();
    recompute_alpha_l();
    #ifdef DEBUG
      check_data(1);
    #endif

    // Compute the time step
    grad_alpha_l.resize();
    normal.resize();
    H.resize();
    H_filter.resize();
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
      // Apply relaxation if desired, which will modify alpha_l and, consequently, for what
      // concerns next time step, rho_alpha_l (as well as grad_alpha_l).
      dalpha_l.resize();
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

      // Apply relaxation
      if(apply_relax) {
        recompute_alpha_l();
        update_geometry();
        // Apply relaxation if desired, which will modify alpha_l and, consequently, for what
        // concerns next time step, rho_alpha_l (as well as grad_alpha_l).
        apply_relaxation(relaxation_op);
      }
      else {
        #ifdef RELAX_RECONSTRUCTION
          recompute_alpha_l();
          update_geometry();
        #endif
      }
    #endif

    // Postprocess data
    #ifndef RELAX_RECONSTRUCTION
      if(!apply_relax) {
        recompute_alpha_l();
        update_geometry();
      }
    #endif
    execute_postprocess(t);

    // Save the results
    if(t >= static_cast<Number>(nsave + 1)*dt_save || t == Tf) {
      // Resize the fields not resized yet
      normal_bar.resize();
      div_vel.resize();
      H_bar.resize();
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                                {
                                  // Compute |\grad\bar{\alpha}| and the related normal
                                  const auto& grad_alpha_l_bar_loc = grad_alpha_l_bar[cell];
                                  auto mod2_grad_alpha_l_bar_loc   = static_cast<Number>(0.0);
                                  for(std::size_t d = 0; d < dim; ++d) {
                                    mod2_grad_alpha_l_bar_loc += grad_alpha_l_bar_loc[d]*grad_alpha_l_bar_loc[d];
                                  }
                                  const auto mod_grad_alpha_l_bar_loc = std::sqrt(mod2_grad_alpha_l_bar_loc);

                                  if(mod_grad_alpha_l_bar_loc > mod_grad_alpha_l_min) {
                                    normal_bar[cell] = grad_alpha_l_bar_loc/mod_grad_alpha_l_bar_loc;
                                  }
                                  else {
                                    for(std::size_t d = 0; d < dim; ++d) {
                                      normal_bar[cell][d] = static_cast<Number>(nan(""));
                                    }
                                  }
                                }
                            );
      samurai::update_ghost_mr(vel, normal_bar);
      div_vel.fill(static_cast<Number>(0.0));
      divergence.apply(div_vel, vel);
      H_bar = -divergence(normal_bar);

      // Perform the saving
      const std::string suffix = (nfiles != 1) ? "_ite_" + Utilities::unsigned_to_string(++nsave) : "";
      save(suffix, conserved_variables,
                   alpha_l, grad_alpha_l, normal, H,
                   p_liq, p_g, p,
                   alpha_d, Sigma_d, grad_alpha_d,
                   vel, div_vel,
                   alpha_l_bar, grad_alpha_l_bar, H_bar,
                   Newton_iterations, Mach);
    }
  }
}
