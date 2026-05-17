// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
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

  const bool   mass_transfer;  /*!< Choose wheter to apply or not the mass transfer */
  const Number Hmax;           /*!< Threshold length scale */
  const Number kappa;          /*!< Parameter related to the radius of small-scale droplets */
  const Number alpha1d_max;    /*!< Maximum threshold of small-scale volume fraction */
  const Number alpha1_bar_min; /*!< Minimum effective volume fraction to identify the mixture region */
  const Number alpha1_bar_max; /*!< Maximum effective volume fraction to identify the mixture region */

  Number cfl; /*!< Courant number of the simulation so as to compute the time step */
  Number dt; /*!< Time step */

  const Number mod_grad_alpha1_bar_min; /*!< Minimum threshold for which not computing anymore the unit normal */

  const Number      lambda;           /*!< Parameter for bound-preserving strategy */
  const Number      atol_Newton;      /*!< Absolute tolerance Newton method relaxation */
  const Number      rtol_Newton;      /*!< Relative tolerance Newton method relaxation */
  const std::size_t max_Newton_iters; /*!< Maximum number of Newton iterations */

  double MR_param;      /*!< Multiresolution parameter */
  double MR_regularity; /*!< Multiresolution regularity */

  LinearizedBarotropicEOS<Number> EOS_phase1,
                                  EOS_phase2; // The two variables which take care of the
                                              // barotropic EOS to compute the speed of sound

  HyperbolicFlux<Field> Hyperbolic_flux; /*!< Auxiliary variable to compute the contribution associated with hyperbolic operator */
  samurai::SurfaceTensionFlux<Field, Field_Vect> SurfaceTension_flux; /*!< Auxiliary variable to compute the contribution associated with surface tension */

  fs::path    path;     /*!< Auxiliary variable to store the output directory */
  std::string filename; /*!< Auxiliary variable to store the name of output */

  Field conserved_variables; /*!< The variable which stores the conserved variables,
                                  namely the varialbes for which we solve a PDE system */
  Field conserved_variables_tmp; /*!< Auxiliary field since we are solving a time-dependent PDE */

  /*--- Now we declare a bunch of fields which depend from the state, but it is useful
        to have it so as to avoid recomputation ---*/
  Field_Scalar alpha1_bar,
               dalpha1_bar,
               p1,
               p2,
               p_bar;

  Field_Vect normal,
             grad_alpha1_bar;

  Field_Scalar alpha1_d,
               Dt_alpha1_d,
               CV_alpha1_d,
               alpha1,
               H_bar,
               div_vel;

  Field_Vect grad_alpha1_d,
             vel,
             grad_alpha1;

  samurai::ScalarField<mesh_type, std::size_t> to_be_relaxed;
  samurai::ScalarField<mesh_type, std::size_t> Newton_iterations;

  using gradient_type = decltype(samurai::make_gradient_order2<decltype(alpha1_bar)>());
  gradient_type gradient;

  using divergence_type = decltype(samurai::make_divergence_order2<decltype(normal)>());
  divergence_type divergence;

  std::optional<PostprocessWriter<Number>> postprocess_writer; /*!< Auxiliary output for post-processing */

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
  void create_fields()

  /**
   * Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)
   * @param x0 x-center of liquid column
   * @param y0 y-center of liquid column
   * @param U0 "phase 2" component of horizontal velocity
   * @param U1 "phase 1" component of horizontal velocity
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
   * @param U0 "phase 2" component of horizontal velocity
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
  void recompute_alpha1_bar();

  /**
   * Perform the finite volume stage (hyperbolic + capillarity subsystems)
   * @param numerical_flux_hyp numerical operator for convective subsystem
   * @param numerical_flux_cap numerical operator for capillarity subsystem
   */
  void perform_fv_stage(auto& numerical_flux_hyp,
                        auto& numerical_flux_st);

  /**
   * Apply the relaxation
   */
  void apply_relaxation();

  void perform_Newton_step_relaxation(auto local_conserved_variables,
                                      const Number H_bar_loc,
                                      Number& dalpha1_bar_loc,
                                      Number& alpha1_bar_loc,
                                      std::size_t& to_be_relaxed_loc,
                                      std::size_t& Newton_iterations_loc,
                                      bool& local_relaxation_applied,
                                      const auto& grad_alpha1_bar_loc,
                                      const bool mass_transfer_NR);

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
  mass_transfer(sim_param.mass_transfer), Hmax(sim_param.Hmax),
  kappa(sim_param.kappa), alpha1d_max(sim_param.alpha1d_max),
  alpha1_bar_min(sim_param.alpha1_bar_min), alpha1_bar_max(sim_param.alpha1_bar_max),
  cfl(sim_param.Courant), mod_grad_alpha1_bar_min(sim_param.mod_grad_alpha1_bar_min),
  lambda(sim_param.lambda), atol_Newton(sim_param.atol_Newton),
  rtol_Newton(sim_param.rtol_Newton), max_Newton_iters(sim_param.max_Newton_iters),
  MR_param(sim_param.MR_param), MR_regularity(sim_param.MR_regularity),
  EOS_phase1(eos_param.p0_phase1, eos_param.rho0_phase1, eos_param.c0_phase1),
  EOS_phase2(eos_param.p0_phase2, eos_param.rho0_phase2, eos_param.c0_phase2),
  Hyperbolic_flux(create_hyperbolic_flux<Field>(sim_param.num_flux_hyp,
                                                EOS_phase1, EOS_phase2, sigma,
                                                lambda, atol_Newton, rtol_Newton, max_Newton_iters)),
  SurfaceTension_flux(EOS_phase1, EOS_phase2, sigma,
                      lambda, atol_Newton, rtol_Newton, max_Newton_iters),
  path(sim_param.save_dir),
  gradient(samurai::make_gradient_order2<decltype(alpha1_bar)>()),
  divergence(samurai::make_divergence_order2<decltype(normal)>())
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

      init_variables(sim_param.x0, sim_param.y0,
                     sim_param.U0, sim_param.U1,
                     sim_param.V0,
                     sim_param.R, sim_param.eps_over_R,
                     sim_param.alpha_residual);
    }
    else {
      samurai::load(sim_param.restart_file, mesh, conserved_variables,
                                                  alpha1_bar, grad_alpha1_bar, normal, H_bar,
                                                  p1, p2, p_bar,
                                                  grad_alpha1_d, vel, div_vel, Dt_alpha1_d, CV_alpha1_d,
                                                  alpha1, grad_alpha1,
                                                  Newton_iterations);
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

  alpha1_bar      = samurai::make_scalar_field<Number>("alpha1_bar", mesh);
  grad_alpha1_bar = samurai::make_vector_field<Number, dim>("grad_alpha1_bar", mesh);
  normal          = samurai::make_vector_field<Number, dim>("normal", mesh);
  H_bar           = samurai::make_scalar_field<Number>("H_bar", mesh);

  dalpha1_bar = samurai::make_scalar_field<Number>("dalpha1_bar", mesh);

  p1    = samurai::make_scalar_field<Number>("p1", mesh);
  p2    = samurai::make_scalar_field<Number>("p2", mesh);
  p_bar = samurai::make_scalar_field<Number>("p_bar", mesh);

  alpha1_d      = samurai::make_scalar_field<Number>("alpha1_d", mesh);
  grad_alpha1_d = samurai::make_vector_field<Number, dim>("grad_alpha1_d", mesh);
  vel           = samurai::make_vector_field<Number, dim>("vel", mesh);
  div_vel       = samurai::make_scalar_field<Number>("div_vel", mesh);
  Dt_alpha1_d   = samurai::make_scalar_field<Number>("Dt_alpha1_d", mesh);
  CV_alpha1_d   = samurai::make_scalar_field<Number>("CV_alpha1_d", mesh);
  alpha1        = samurai::make_scalar_field<Number>("alpha1", mesh);
  grad_alpha1   = samurai::make_vector_field<Number, dim>("grad_alpha1", mesh);

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
  alpha1_bar.resize();
  grad_alpha1_bar.resize();
  normal.resize();
  H_bar.resize();
  dalpha1_bar.resize();
  p1.resize();
  p2.resize();
  p_bar.resize();
  alpha1_d.resize();
  grad_alpha1_d.resize();
  vel.resize();
  div_vel.resize();
  Dt_alpha1_d.resize();
  CV_alpha1_d.resize();
  alpha1.resize();
  grad_alpha1.resize();
  to_be_relaxed.resize();
  Newton_iterations.resize();

  // Declare some constant parameters associated with the initial state
  const auto eps_R = eps_over_R*R;

  // Initialize some fields to define the liquid column with a loop over all cells
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

                              alpha1_bar[cell] = std::min(std::max(alpha_residual, w), 1.0 - alpha_residual);
                            }
                        );

  // Compute the geometrical quantities
  update_geometry();

  // Loop over a cell to complete the remaining variables
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Set small-scale variables
                              conserved_variables[cell](ALPHA1_D_INDEX) = static_cast<Number>(0.0);
                              alpha1_d[cell]                            = conserved_variables[cell](ALPHA1_D_INDEX);
                              conserved_variables[cell](SIGMA_D_INDEX)  = static_cast<Number>(0.0);
                              conserved_variables[cell](M1_D_INDEX)     = conserved_variables[cell](ALPHA1_D_INDEX)*EOS_phase1.get_rho0();

                              // Recompute geometric locations to set partial masses
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);
                              const auto y      = static_cast<Number>(center[1]);
                              const auto r      = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                              // Set mass large-scale phase 1
                              if(r >= R + eps_R) {
                                p1[cell] = EOS_phase1.get_p0();
                              }
                              else {
                                p1[cell] = EOS_phase2.get_p0();
                                if(r >= R && r < R + eps_R && !std::isnan(H_bar[cell])) {
                                  p1[cell] += sigma*H_bar[cell];
                                }
                                else {
                                  p1[cell] += sigma/R;
                                }
                              }
                              const auto rho1 = EOS_phase1.rho_value(p1[cell]);

                              alpha1[cell] = alpha1_bar[cell]*
                                             (static_cast<Number>(1.0) - conserved_variables[cell](ALPHA1_D_INDEX));
                              conserved_variables[cell](M1_INDEX) = alpha1[cell]*rho1;

                              // Set mass phase 2
                              p2[cell] = EOS_phase2.get_p0();
                              const auto rho2 = EOS_phase2.rho_value(p2[cell]);

                              const auto alpha2 = static_cast<Number>(1.0)
                                                - alpha1[cell]
                                                - conserved_variables[cell](ALPHA1_D_INDEX);
                              conserved_variables[cell](M2_INDEX) = alpha2*rho2;

                              // Save mixture pressure for post-processing
                              p_bar[cell] = alpha1_bar[cell]*p1[cell]
                                          + (static_cast<Number>(1.0) - alpha1_bar[cell])*p2[cell];

                              // Set conserved variable associated with large-scale volume fraction
                              const auto rho = conserved_variables[cell](M1_INDEX)
                                             + conserved_variables[cell](M2_INDEX)
                                             + conserved_variables[cell](M1_D_INDEX);

                              conserved_variables[cell](RHO_ALPHA1_BAR_INDEX) = rho*alpha1_bar[cell];

                              // Set momentum
                              conserved_variables[cell](RHO_U_INDEX)     = conserved_variables[cell](M1_INDEX)*U1
                                                                         + conserved_variables[cell](M2_INDEX)*U0;
                              conserved_variables[cell](RHO_U_INDEX + 1) = rho*V0;

                              // Save velocity for post-processing
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel[cell][d] = conserved_variables[cell](RHO_U_INDEX + d)/rho;
                              }
                            }
                        );

  // Set useful small-scale related fields
  samurai::update_ghost_mr(alpha1_d);
  grad_alpha1_d.fill(static_cast<Number>(0.0));
  gradient.apply(grad_alpha1_d, alpha1_d);

  samurai::update_ghost_mr(vel);
  div_vel.fill(static_cast<Number>(0.0));
  divergence.apply(div_vel, vel);

  // Set auxiliary gradient large-scale volume fraction
  samurai::update_ghost_mr(alpha1);
  grad_alpha1.fill(static_cast<Number>(0.0));
  gradient.apply(grad_alpha1, alpha1);
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
                                  EOS_phase1.get_rho0(),
                                  static_cast<Number>(0.0)))->on(left);
  /*samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables,
                                          alpha_residual*EOS_phase1.get_rho0(),
                                          (static_cast<Number>(1.0) - alpha_residual)*EOS_phase2.get_rho0(),
                                          static_cast<Number>(0.0), static_cast<Number>(0.0), static_cast<Number>(0.0),
                                          (alpha_residual*EOS_phase1.get_rho0() +
                                           (static_cast<Number>(1.0) - alpha_residual)*EOS_phase2.get_rho0())*
                                          alpha_residual,
                                          (alpha_residual*EOS_phase1.get_rho0() +
                                           (static_cast<Number>(1.0) - alpha_residual)*EOS_phase2.get_rho0())*U0,
                                          (alpha_residual*EOS_phase1.get_rho0() +
                                           (static_cast<Number>(1.0) - alpha_residual)*EOS_phase2.get_rho0())*V0)->on(left);*/

  const samurai::DirectionVector<dim> right = {1, 0};
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables,
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0))->on(right);

  /*const samurai::DirectionVector<dim> top = {0, 1};
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables,
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0))->on(top);

  const samurai::DirectionVector<dim> bottom = {0, -1};
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables,
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0))->on(bottom);*/
}

/************************************************************
******* FOCUS NOW ON THE AUXILIARY FUNCTIONS ****************
*************************************************************/

// Auxiliary routine to compute the gradient of large-scale volume fraction
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::update_gradient() {
  samurai::update_ghost_mr(alpha1_bar);
  grad_alpha1_bar.fill(static_cast<Number>(0.0));
  gradient.apply(grad_alpha1_bar, alpha1_bar);
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
                              //const auto mod_grad_alpha1_bar_loc = std::sqrt(xt::sum(grad_alpha1_bar[cell]*grad_alpha1_bar[cell])());
                              auto mod2_grad_alpha1_bar_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                mod2_grad_alpha1_bar_loc += grad_alpha1_bar[cell][d]*grad_alpha1_bar[cell][d];
                              }
                              const auto mod_grad_alpha1_bar_loc = std::sqrt(mod2_grad_alpha1_bar_loc);

                              if(mod_grad_alpha1_bar_loc > mod_grad_alpha1_bar_min) {
                                normal[cell] = grad_alpha1_bar[cell]/mod_grad_alpha1_bar_loc;
                              }
                              else {
                                for(std::size_t d = 0; d < dim; ++d) {
                                  normal[cell][d] = static_cast<Number>(nan(""));
                                }
                              }
                            }
                        );
  samurai::update_ghost_mr(normal);
  H_bar = -divergence(normal);
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

                              const auto m1_loc       = local_conserved_variables(M1_INDEX);
                              const auto m2_loc       = local_conserved_variables(M2_INDEX);
                              const auto m1_d_loc     = local_conserved_variables(M1_D_INDEX);
                              const auto alpha1_d_loc = local_conserved_variables(ALPHA1_D_INDEX);

                              // Compute the velocity along all the directions
                              const auto rho_loc     = m1_loc + m2_loc + m1_d_loc;
                              const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel_loc[d] = local_conserved_variables(RHO_U_INDEX + d)*inv_rho_loc;
                              }

                              // Compute frozen speed of sound
                              const auto alpha1_loc       = alpha1_bar[cell]*
                                                            (static_cast<Number>(1.0) - alpha1_d_loc);
                              const auto rho1_loc         = m1_loc/alpha1_loc;
                                                            // TODO: Add a check in case of zero volume fraction
                              const auto alpha2_loc       = static_cast<Number>(1.0) - alpha1_loc - alpha1_d_loc;
                              const auto rho2_loc         = m2_loc/alpha2_loc;
                                                            // TODO: Add a check in case of zero volume fraction
                              const auto c1_loc           = EOS_phase1.c_value(rho1_loc);
                              const auto c2_loc           = EOS_phase2.c_value(rho2_loc);
                              const auto rhoc_squared_loc = m1_loc*c1_loc*c1_loc
                                                          + m2_loc*c2_loc*c2_loc;
                              const auto c_loc            = std::sqrt(rhoc_squared_loc*inv_rho_loc)/
                                                            (static_cast<Number>(1.0) - alpha1_d_loc);

                              // Add term due to surface tension
                              auto mod2_grad_alpha1_bar_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                mod2_grad_alpha1_bar_loc += grad_alpha1_bar[cell][d]*grad_alpha1_bar[cell][d];
                              }
                              const auto mod_grad_alpha1_bar_loc = std::sqrt(mod2_grad_alpha1_bar_loc);

                              const auto r = sigma*mod_grad_alpha1_bar_loc/(rho_loc*c_loc*c_loc);

                              // Update eigenvalue estimate
                              for(std::size_t d = 0; d < dim; ++d) {
                                local_res = std::max(local_res,
                                                     std::abs(vel_loc[d]) + c_loc*(static_cast<Number>(1.0) +
                                                                                   static_cast<Number>(0.125)*r));
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
                                      save("_diverged", conserved_variables, alpha1_bar);
                                      exit(1);
                                    }
                                    else if(std::isnan(val)) {
                                      std::cerr << cell << std::endl;
                                      std::cerr << "NaN " + name + op << std::endl;
                                      save("_diverged", conserved_variables, alpha1_bar);
                                      exit(1);
                                    }
                                  };

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch local state
                              const auto& local_conserved_variables = conserved_variables[cell];

                              // Sanity check for alpha1_bar
                              const auto alpha1_bar_loc = alpha1_bar[cell];
                              if(alpha1_bar_loc < static_cast<Number>(0.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative large-scale volume fraction of phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }
                              else if(alpha1_bar_loc > static_cast<Number>(1.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Exceeding large-scale volume fraction of phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }
                              else if(std::isnan(alpha1_bar_loc)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN large-scale volume fraction of phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }

                              // Sanity check for m1
                              check_positive_field(local_conserved_variables(M1_INDEX), cell,
                                                   "mass large-scale phase 1 ");

                              // Sanity check for m2
                              check_positive_field(local_conserved_variables(M2_INDEX), cell,
                                                   "mass phase 2 ");

                              // Sanity check for m1_d
                              check_positive_field(local_conserved_variables(M1_D_INDEX), cell,
                                                   "mass small-scale phase 1 ");

                              // Sanity check for alpha1_d
                              const auto alpha1_d_loc = local_conserved_variables(ALPHA1_D_INDEX);
                              if(alpha1_d_loc > static_cast<Number>(1.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Exceeding value of small-scale volume fraction " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }
                              else if(alpha1_d_loc < static_cast<Number>(0.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative small-scale volume fraction " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }
                              else if(std::isnan(alpha1_d_loc)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN small-scale volume fraction " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }

                              // Sanity check for Sigma_d
                              check_positive_field(local_conserved_variables(SIGMA_D_INDEX), cell,
                                                   "small-scale interfacial area ");
                            }
                        );
}

// Auxiliary function to compute large-scale volume fraction from conserved variables
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::recompute_alpha1_bar() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              const auto& local_conserved_variables = conserved_variables[cell];

                              alpha1_bar[cell] = local_conserved_variables(RHO_ALPHA1_BAR_INDEX)/
                                                 (local_conserved_variables(M1_INDEX) +
                                                  local_conserved_variables(M2_INDEX) +
                                                  local_conserved_variables(M1_D_INDEX));
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
    samurai::update_ghost_mr(H_bar);
  #endif
  try {
    conserved_variables_tmp = conserved_variables
                            - dt*numerical_flux_hyp(conserved_variables);
    samurai::swap(conserved_variables, conserved_variables_tmp);
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << std::endl;
    save("_diverged", conserved_variables, alpha1_bar);
    exit(1);
  }

  // Recompute geometrical quantities
  recompute_alpha1_bar();
  #ifdef DEBUG
    check_data();
  #endif
  update_gradient();

  // Capillarity contribution
  conserved_variables_tmp = conserved_variables
                          - dt*numerical_flux_st(grad_alpha1_bar);
  samurai::swap(conserved_variables, conserved_variables_tmp);
}

/************************************************************
******* FOCUS NOW ON THE RELAXATION FUNCTIONS ***************
*************************************************************/

// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::apply_relaxation() {
  // Initialize the variables
  samurai::times::timers.start("apply_relaxation");

  std::size_t Newton_iter = 0;
  Newton_iterations.fill(0);
  dalpha1_bar.fill(std::numeric_limits<Number>::infinity());
  bool global_relaxation_applied = true;
  bool mass_transfer_NR          = mass_transfer; // In principle we might think to disable it after a certain
                                                  // number of iterations (as in Arthur's code), not done here.

  samurai::times::timers.stop("apply_relaxation");

  // Loop of Newton method. Conceptually, a loop over cells followed by a Newton loop
  // over each cell would (could?) be more logic, but this would lead to issues to call 'update_geometry'
  while(global_relaxation_applied == true) {
    samurai::times::timers.start("apply_relaxation");

    bool local_relaxation_applied = false;
    Newton_iter++;

    // Loop over all cells.
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             try {
                               perform_Newton_step_relaxation(conserved_variables[cell],
                                                              H_bar[cell], dalpha1_bar[cell], alpha1_bar[cell],
                                                              to_be_relaxed[cell], Newton_iterations[cell],
                                                              local_relaxation_applied,
                                                              grad_alpha1_bar[cell], mass_transfer_NR);
                             }
                             catch(const std::exception& e) {
                               std::cerr << e.what() << std::endl;
                               save("_diverged",
                                    conserved_variables,
                                    alpha1_bar, dalpha1_bar, grad_alpha1_bar, normal, H_bar,
                                    to_be_relaxed, Newton_iterations);
                               exit(1);
                             }
                           });

    #ifdef SAMURAI_WITH_MPI
      mpi::communicator world;
      boost::mpi::all_reduce(world, local_relaxation_applied, global_relaxation_applied, std::logical_or<bool>());
    #else
      global_relaxation_applied = local_relaxation_applied;
    #endif

    // Newton cycle diverged
    if(Newton_iter > max_Newton_iters && global_relaxation_applied == true) {
      std::cerr << "Netwon method not converged in the post-hyperbolic relaxation" << std::endl;
      save("_diverged",
           conserved_variables,
           alpha1_bar, dalpha1_bar, grad_alpha1_bar, normal, H_bar,
           to_be_relaxed, Newton_iterations);
      exit(1);
    }

    samurai::times::timers.stop("apply_relaxation");

    // Recompute geometric quantities (curvature potentially changed in the Newton loop)
    update_geometry();
  }
}

// Implement a single step of the relaxation procedure (valid for a general EOS)
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::perform_Newton_step_relaxation(auto local_conserved_variables,
                                                              const Number H_bar_loc,
                                                              Number& dalpha1_bar_loc,
                                                              Number& alpha1_bar_loc,
                                                              std::size_t& to_be_relaxed_loc,
                                                              std::size_t& Newton_iterations_loc,
                                                              bool& local_relaxation_applied,
                                                              const auto& grad_alpha1_bar_loc,
                                                              const bool mass_transfer_NR) {
  to_be_relaxed_loc = 0;

  if(!std::isnan(H_bar_loc)) {
    // Pre-fetch some variables used multiple times in order to exploit possible vectorization
    const auto m1_loc       = local_conserved_variables(M1_INDEX);
    const auto m2_loc       = local_conserved_variables(M2_INDEX);
    const auto m1_d_loc     = local_conserved_variables(M1_D_INDEX);
    const auto alpha1_d_loc = local_conserved_variables(ALPHA1_D_INDEX);

    // Update auxiliary values affected by the nonlinear function for which we seek a zero
    const auto alpha1_loc = alpha1_bar_loc*(static_cast<Number>(1.0) - alpha1_d_loc);
    const auto rho1_loc   = m1_loc/alpha1_loc; // TODO: Add a check in case of zero volume fraction
    const auto p1_loc     = EOS_phase1.pres_value(rho1_loc);

    const auto alpha2_loc = static_cast<Number>(1.0) - alpha1_loc - alpha1_d_loc;
    const auto rho2_loc   = m2_loc/alpha2_loc; // TODO: Add a check in case of zero volume fraction
    const auto p2_loc     = EOS_phase2.pres_value(rho2_loc);

    const auto rho1d_loc     = (m1_d_loc > static_cast<Number>(0.0) &&
                               alpha1_d_loc > static_cast<Number>(0.0)) ?
                               m1_d_loc/alpha1_d_loc : EOS_phase1.get_rho0();
    const auto inv_rho1d_loc = static_cast<Number>(1.0)/rho1d_loc;

    // Prepare for mass transfer if desired
    const auto rho_loc = m1_loc + m2_loc + m1_d_loc;

    // Compute first ordrer integral reminder "specific enthalpy"
    auto alpha2_bar_loc  = static_cast<Number>(1.0) - alpha1_bar_loc;
    const auto p_bar_loc = alpha1_bar_loc*p1_loc
                         + alpha2_bar_loc*p2_loc;
    const auto p2_minus_p1_times_theta = rho1_loc/alpha2_bar_loc*
                                         (EOS_phase1.e_value(rho1d_loc) - EOS_phase1.e_value(rho1_loc) +
                                          p_bar_loc*inv_rho1d_loc - p1_loc/rho1_loc) -
                                         (p2_loc - p1_loc);
    Number H_lim = std::min(H_bar_loc, Hmax);
    const auto fac_Ru = sigma*(static_cast<Number>(3.0)*H_lim/(kappa*rho1d_loc))*
                              (rho1_loc/alpha2_bar_loc)
                      - sigma*H_lim/(static_cast<Number>(1.0) - alpha1_d_loc)
                      + p2_minus_p1_times_theta;
    if(mass_transfer_NR) {
      if(fac_Ru > static_cast<Number>(0.0) &&
         alpha1_bar_loc > alpha1_bar_min && alpha1_bar_loc < alpha1_bar_max &&
         -grad_alpha1_bar_loc[0]*local_conserved_variables(RHO_U_INDEX)
         -grad_alpha1_bar_loc[1]*local_conserved_variables(RHO_U_INDEX + 1) > static_cast<Number>(0.0) &&
         alpha1_d_loc < alpha1d_max) {
        ;
      }
      else {
        H_lim = H_bar_loc;
      }
    }
    else {
      H_lim = H_bar_loc;
    }

    const auto dH = H_bar_loc - H_lim;

    // Compute the nonlinear function for which we seek the zero (basically the Laplace law)
    const auto F = (static_cast<Number>(1.0) - alpha1_d_loc)*(p1_loc - p2_loc)
                 - sigma*H_lim;

    // Perform the relaxation only where really needed
    if(std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase1.get_p0(), sigma*std::abs(H_lim)) &&
       std::abs(dalpha1_bar_loc) > atol_Newton) {
      to_be_relaxed_loc = 1;
      Newton_iterations_loc++;
      local_relaxation_applied = true;

      // Compute the derivative w.r.t large scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
      const auto c1_loc = EOS_phase1.c_value(rho1_loc);
      const auto c2_loc = EOS_phase2.c_value(rho2_loc);

      const auto dF_dalpha1_bar = -m1_loc/(alpha1_bar_loc*alpha1_bar_loc)*
                                   c1_loc*c1_loc
                                  -m2_loc/(alpha2_bar_loc*alpha2_bar_loc)*
                                   c2_loc*c2_loc;

      // Compute the pseudo time step starting as initial guess from the ideal unmodified Newton method
      auto dtau_ov_epsilon = std::numeric_limits<Number>::infinity();

      // Bound-preserving condition for m1, velocity and small-scale volume fraction
      if(dH > static_cast<Number>(0.0)) {
        // Bound-preserving condition for m1
        dtau_ov_epsilon = lambda*(alpha1_loc*alpha2_bar_loc)/(sigma*dH);
        #ifdef DEBUG
          if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
            throw std::runtime_error("Negative time step found after relaxation of mass of large-scale phase 1");
          }
        #endif

        // Bound preserving for the velocity
        const auto mom_dot_vel   = (local_conserved_variables(RHO_U_INDEX)*local_conserved_variables(RHO_U_INDEX) +
                                    local_conserved_variables(RHO_U_INDEX + 1)*local_conserved_variables(RHO_U_INDEX + 1))/rho_loc;
        auto dtau_ov_epsilon_tmp = lambda*mom_dot_vel/(dH*fac_Ru*sigma);
        dtau_ov_epsilon          = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
        #ifdef DEBUG
          if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
            throw std::runtime_error("Negative time step found after relaxation of velocity");
          }
        #endif

        // Bound preserving for the small-scale volume fraction
        dtau_ov_epsilon_tmp = lambda*(alpha1d_max - alpha1_d_loc)*alpha2_bar_loc*rho1d_loc/
                                     (rho1_loc*sigma*dH);
        dtau_ov_epsilon     = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
        if(alpha1_d_loc > static_cast<Number>(0.0)) {
          dtau_ov_epsilon_tmp = lambda*alpha1_d_loc*alpha2_bar_loc*rho1d_loc/
                                       (rho1_loc*sigma*dH);

          dtau_ov_epsilon     = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
        }
        #ifdef DEBUG
          if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
            throw std::runtime_error("Negative time step found after relaxation of small-scale volume fraction");
          }
        #endif
      }

      // Bound-preserving condition for large-scale volume fraction
      const auto dF_dalpha1d   = p2_loc - p1_loc
                               + EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)*rho1_loc
                               - EOS_phase2.c_value(rho2_loc)*EOS_phase2.c_value(rho2_loc)*rho2_loc;
      const auto dF_dm1        = EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)/alpha1_bar_loc;
      const auto R             = dF_dalpha1d*inv_rho1d_loc - dF_dm1;
      const auto a             = rho1_loc*sigma*dH*R/
                                 (alpha2_bar_loc*(static_cast<Number>(1.0) - alpha1_d_loc));
      // Upper bound
      auto b                   = (F + lambda*alpha2_bar_loc*dF_dalpha1_bar)/
                                 (static_cast<Number>(1.0) - alpha1_d_loc);
      auto D                   = b*b - static_cast<Number>(4.0)*a*(-lambda*alpha2_bar_loc);
      auto dtau_ov_epsilon_tmp = std::numeric_limits<Number>::infinity();
      if(D > static_cast<Number>(0.0) &&
         (a > static_cast<Number>(0.0) ||
         (a < static_cast<Number>(0.0) && b > static_cast<Number>(0.0)))) {
        dtau_ov_epsilon_tmp = static_cast<Number>(0.5)*(-b + std::sqrt(D))/a;
      }
      if(a == static_cast<Number>(0.0) &&
         b > static_cast<Number>(0.0)) {
        dtau_ov_epsilon_tmp = lambda*alpha2_bar_loc/b;
      }
      dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
      // Lower bound
      dtau_ov_epsilon_tmp = std::numeric_limits<Number>::infinity();
      b                   = (F - lambda*alpha1_bar_loc*dF_dalpha1_bar)/
                            (static_cast<Number>(1.0) - alpha1_d_loc);
      D                   = b*b - static_cast<Number>(4.0)*a*(lambda*alpha1_bar_loc);
      if(D > static_cast<Number>(0.0) &&
         (a < static_cast<Number>(0.0) ||
         (a > static_cast<Number>(0.0) && b < static_cast<Number>(0.0)))) {
        dtau_ov_epsilon_tmp = static_cast<Number>(0.5)*(-b - std::sqrt(D))/a;
      }
      if(a == static_cast<Number>(0.0) &&
         b < static_cast<Number>(0.0)) {
        dtau_ov_epsilon_tmp = -lambda*alpha1_bar_loc/b;
      }
      dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
      #ifdef DEBUG
        if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
          throw std::runtime_error("Negative time step found after relaxation of large-scale volume fraction");
        }
      #endif

      // Compute the effective variation of the variables
      if(std::isinf(dtau_ov_epsilon)) {
        // If we are in this branch we do not have mass transfer
        // and we do not have other restrictions on the bounds of large scale volume fraction
        dalpha1_bar_loc = -F/dF_dalpha1_bar;

        /*if(dalpha1_bar_loc > static_cast<Number>(0.0)) {
          dalpha1_bar_loc = std::min(dalpha1_bar_loc, lambda*alpha2_bar_loc);
        }
        else if(dalpha1_bar_loc < static_cast<Number>(0.0)) {
          dalpha1_bar_loc = std::max(dalpha1_bar_loc, -lambda*alpha1_bar_loc);
        }*/
      }
      else {
        const auto dm1 = -dtau_ov_epsilon/alpha2_bar_loc*
                          (m1_loc/(alpha1_bar_loc*(static_cast<Number>(1.0) - alpha1_d_loc)))*
                          sigma*dH;

        #ifdef DEBUG
          if(dm1 > static_cast<Number>(0.0)) {
            throw std::runtime_error("Negative sign of mass transfer inside Newton step");
          }
        #endif
        local_conserved_variables(M1_INDEX) += dm1;
        #ifdef DEBUG
          // I should never get here. Added only for the sake of safety!!
          if(local_conserved_variables(M1_INDEX) < static_cast<Number>(0.0)) {
            throw std::runtime_error("Negative mass of large-scale phase 1 inside Newton step");
          }
        #endif

        local_conserved_variables(M1_D_INDEX) -= dm1;
        #ifdef DEBUG
          // I should never get here. Added only for the sake of safety!!
          if(local_conserved_variables(M1_D_INDEX) < static_cast<Number>(0.0)) {
            throw std::runtime_error("Negative mass of small-scale phase 1 inside Newton step");
          }
        #endif

        #ifdef DEBUG
          if(alpha1_d_loc - dm1*inv_rho1d_loc > static_cast<Number>(1.0)) {
            throw std::runtime_error("Exceeding value for small-scale volume fraction inside Newton step");
          }
        #endif
        local_conserved_variables(ALPHA1_D_INDEX) -= dm1*inv_rho1d_loc;

        local_conserved_variables(SIGMA_D_INDEX) -= dm1*static_cast<Number>(3.0)*Hmax/(kappa*rho1d_loc);

        const auto mom_squared = local_conserved_variables(RHO_U_INDEX)*local_conserved_variables(RHO_U_INDEX)
                               + local_conserved_variables(RHO_U_INDEX + 1)*local_conserved_variables(RHO_U_INDEX + 1);
        const auto drho_fac_Ru = dtau_ov_epsilon*
                                 sigma*dH*fac_Ru*rho_loc/mom_squared; /*--- u/u^{2} = rho*u/(rho*(u^{2})) = (rho/(rho*u)^{2})*(rho*u) ---*/

        for(std::size_t d = 0; d < Field::dim; ++d) {
          local_conserved_variables(RHO_U_INDEX + d) -= drho_fac_Ru*local_conserved_variables(RHO_U_INDEX + d);
        }

        const auto num_dalpha1_bar = dtau_ov_epsilon/(static_cast<Number>(1.0) - alpha1_d_loc);
        const auto den_dalpha1_bar = static_cast<Number>(1.0) - num_dalpha1_bar*dF_dalpha1_bar;
        dalpha1_bar_loc            = (num_dalpha1_bar/den_dalpha1_bar)*(F - dm1*R);
      }

      #ifdef DEBUG
        if(alpha1_bar_loc + dalpha1_bar_loc < static_cast<Number>(0.0) ||
           alpha1_bar_loc + dalpha1_bar_loc > static_cast<Number>(1.0)) {
          // I should never get here. Added only for the sake of safety!!
          throw std::runtime_error("Bounds exceeding value for large-scale volume fraction inside Newton step");
        }
      #endif
      alpha1_bar_loc += dalpha1_bar_loc;
    }

    // Update "conservative counter part" of large-scale volume fraction.
    // Do it outside because this can change either because of mass-transfer or
    // alpha1_bar.
    local_conserved_variables(RHO_ALPHA1_BAR_INDEX) = rho_loc*alpha1_bar_loc;
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

  alpha1.resize();
  alpha1_d.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Save liquid large- and small-scale variables
                              const auto alpha1_d_loc = conserved_variables[cell](ALPHA1_D_INDEX);
                              alpha1[cell]            = alpha1_bar[cell]*(static_cast<Number>(1.0) - alpha1_d_loc);
                              alpha1_d[cell]          = alpha1_d_loc;
                            }
                        );
  samurai::update_ghost_mr(alpha1);
  grad_alpha1.resize();
  grad_alpha1.fill(static_cast<Number>(0.0));
  gradient.apply(grad_alpha1, alpha1);
  samurai::update_ghost_mr(alpha1_d);
  grad_alpha1_d.resize();
  grad_alpha1_d.fill(static_cast<Number>(0.0));
  gradient.apply(grad_alpha1_d, alpha1_d);

  p1.resize();
  p2.resize();
  p_bar.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch some variables used multiple times in order to exploit possible vectorization
                              const auto& local_conserved_variables = conserved_variables[cell];

                              const auto m1_loc         = local_conserved_variables(M1_INDEX);
                              const auto m2_loc         = local_conserved_variables(M2_INDEX);
                              const auto m1_d_loc       = local_conserved_variables(M1_D_INDEX);
                              const auto alpha1_d_loc   = local_conserved_variables(ALPHA1_D_INDEX);
                              const auto alpha1_loc     = alpha1[cell];
                              const auto alpha1_bar_loc = alpha1_bar[cell];
                              const auto alpha2_bar_loc = static_cast<Number>(1.0) - alpha1_bar_loc;
                              const auto H_bar_loc      = H_bar[cell];

                              // Compue H_lig
                              const auto rho1_loc  = m1_loc/alpha1_loc;
                                                     // TODO: Add a check in case of zero volume fraction
                              const auto rho1d_loc = (alpha1_d_loc > static_cast<Number>(0.0)) ?
                                                     m1_d_loc/alpha1_d_loc : EOS_phase1.get_rho0();
                              const auto p1_loc    = EOS_phase1.pres_value(rho1_loc);
                              p1[cell]             = p1_loc;
                              const auto rho2_loc  = m2_loc/(static_cast<Number>(1.0) - alpha1_loc - alpha1_d_loc);
                                                     // TODO: Add a check in case of zero volume fraction
                              const auto p2_loc    = EOS_phase2.pres_value(rho2_loc);
                              p2[cell]             = p2_loc;
                              const auto p_bar_loc = alpha1_bar_loc*p1_loc
                                                   + alpha2_bar_loc*p2_loc;
                              p_bar[cell]          = p_bar_loc;
                              const auto H_lim_loc = std::min(H_bar_loc, Hmax);
                              const auto p2_minus_p1_times_theta = rho1_loc/alpha2_bar_loc*
                                                                   (EOS_phase1.e_value(rho1d_loc) - EOS_phase1.e_value(rho1_loc) +
                                                                    p_bar_loc/rho1d_loc - p1_loc/rho1_loc) -
                                                                   (p2_loc - p1_loc);
                              const auto fac_Ru = sigma*(static_cast<Number>(3.0)*H_lim_loc/(kappa*rho1d_loc))*
                                                        (rho1_loc/alpha2_bar_loc)
                                                - sigma*H_lim_loc/(static_cast<Number>(1.0) - alpha1_d_loc)
                                                + p2_minus_p1_times_theta;
                              if(fac_Ru > static_cast<Number>(0.0) &&
                                 alpha1_bar_loc > alpha1_bar_min && alpha1_bar_loc < alpha1_bar_max &&
                                 -grad_alpha1_bar[cell][0]*local_conserved_variables(RHO_U_INDEX)
                                 -grad_alpha1_bar[cell][1]*local_conserved_variables(RHO_U_INDEX + 1) > static_cast<Number>(0.0) &&
                                alpha1_d_loc < alpha1d_max) {
                                local_q.H_lig = std::max(H_bar_loc, local_q.H_lig);
                              }

                              // Compute geometric Euclidean norms
                              auto mod2_grad_alpha1_bar_loc = static_cast<Number>(0.0);
                              auto mod2_grad_alpha1_d_loc   = static_cast<Number>(0.0);
                              auto mod2_grad_alpha1_loc     = static_cast<Number>(0.0);
                              auto mod2_grad_alpha1_tot_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                mod2_grad_alpha1_bar_loc += grad_alpha1_bar[cell][d]*grad_alpha1_bar[cell][d];
                                mod2_grad_alpha1_d_loc   += grad_alpha1_d[cell][d]*grad_alpha1_d[cell][d];
                                mod2_grad_alpha1_loc     += grad_alpha1[cell][d]*grad_alpha1[cell][d];
                                mod2_grad_alpha1_tot_loc += (grad_alpha1[cell][d] + grad_alpha1_d[cell][d])*
                                                            (grad_alpha1[cell][d] + grad_alpha1_d[cell][d]);
                              }

                              // Compute the integral quantities
                              auto cell_volume = static_cast<Number>(cell.length);
                              for(std::size_t d = 1; d < dim; ++d) {
                                cell_volume *= static_cast<Number>(cell.length);
                              }

                              local_q.m1_int += m1_loc*cell_volume;
                              local_q.m1_d_int += m1_d_loc*cell_volume;
                              local_q.alpha1_bar_int += alpha1_bar_loc*cell_volume;
                              local_q.grad_alpha1_bar_int += std::sqrt(mod2_grad_alpha1_bar_loc)*cell_volume;
                              local_q.Sigma_d_int += local_conserved_variables(SIGMA_D_INDEX)*cell_volume;
                              local_q.alpha1_d_int += alpha1_d_loc*cell_volume;
                              local_q.grad_alpha1_d_int += std::sqrt(mod2_grad_alpha1_d_loc)*cell_volume;
                              local_q.grad_alpha1_int += std::sqrt(mod2_grad_alpha1_loc)*cell_volume;
                              local_q.grad_alpha1_tot_int += std::sqrt(mod2_grad_alpha1_tot_loc)*cell_volume;
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
                                                 return f.make_two_scale_capillarity(H_bar);
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

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_" + Utilities::unsigned_to_string(0) : "";
  save(suffix_init, conserved_variables,
                    alpha1_bar, grad_alpha1_bar, normal, H_bar,
                    p1, p2, p_bar,
                    grad_alpha1_d, vel, div_vel, alpha1, grad_alpha1);
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
    alpha1_bar.resize();
    recompute_alpha1_bar();
    #ifdef DEBUG
      check_data(1);
    #endif

    // Compute the time step
    grad_alpha1_bar.resize();
    normal.resize();
    H_bar.resize();
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
      // Apply relaxation if desired, which will modify alpha1_bar and, consequently, for what
      // concerns next time step, rho_alpha1_bar (as well as grad_alpha1_bar).
      dalpha1_bar.resize();
      to_be_relaxed.resize();
      Newton_iterations.resize();
      update_geometry(false);
      apply_relaxation();
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
        recompute_alpha1_bar();
        update_geometry();
        // Apply relaxation if desired, which will modify alpha1_bar and, consequently, for what
        // concerns next time step, rho_alpha1_bar (as well as grad_alpha1_bar).
        apply_relaxation();
      }
      else {
        #ifdef RELAX_RECONSTRUCTION
          recompute_alpha1_bar();
          update_geometry();
        #endif
      }
    #endif

    // Postprocess data
    #ifndef RELAX_RECONSTRUCTION
      if(!apply_relax) {
        recompute_alpha1_bar();
        update_geometry();
      }
    #endif
    execute_postprocess(t);

    // Save the results
    if(t >= static_cast<Number>(nsave + 1)*dt_save || t == Tf) {
      // Resize all the fields not resized yet
      vel.resize();
      div_vel.resize();
      Dt_alpha1_d.resize();
      CV_alpha1_d.resize();

      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                                {
                                  // Pre-fetch local state
                                  const auto& local_conserved_variables = conserved_variables[cell];

                                  // Compute velocity
                                  const auto rho_loc     = local_conserved_variables(M1_INDEX)
                                                         + local_conserved_variables(M2_INDEX)
                                                         + local_conserved_variables(M1_D_INDEX);
                                  const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;
                                  auto vel_loc           = vel[cell];
                                  for(std::size_t d = 0; d < dim; ++d) {
                                    vel_loc[d] = local_conserved_variables(RHO_U_INDEX + d)*inv_rho_loc;
                                  }

                                  // Compute auxiliary variables
                                  const auto alpha1_d_loc       = local_conserved_variables(ALPHA1_D_INDEX);
                                  const auto& grad_alpha1_d_loc = grad_alpha1_d[cell];
                                  #ifdef ORDER_2
                                    Dt_alpha1_d[cell] = (alpha1_d_loc - conserved_variables_old[cell](ALPHA1_D_INDEX))/dt
                                                      + vel_loc[0]*grad_alpha1_d_loc[0]
                                                      + vel_loc[1]*grad_alpha1_d_loc[1];
                                  #else
                                    Dt_alpha1_d[cell] = (alpha1_d_loc - conserved_variables_tmp[cell](ALPHA1_D_INDEX))/dt
                                                      + vel_loc[0]*grad_alpha1_d_loc[0]
                                                      + vel_loc[1]*grad_alpha1_d_loc[1];
                                  #endif
                                }
                            );

      samurai::update_ghost_mr(vel);
      div_vel.fill(static_cast<Number>(0.0));
      divergence.apply(div_vel, vel);

      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                                {
                                  CV_alpha1_d[cell] = Dt_alpha1_d[cell]
                                                    + conserved_variables[cell][ALPHA1_D_INDEX]*div_vel[cell];
                                }
                            );

      // Perform the saving
      const std::string suffix = (nfiles != 1) ? "_ite_" + Utilities::unsigned_to_string(++nsave) : "";
      save(suffix, conserved_variables,
                   alpha1_bar, grad_alpha1_bar, normal, H_bar,
                   p1, p2, p_bar,
                   grad_alpha1_d, vel, div_vel, Dt_alpha1_d, CV_alpha1_d,
                   alpha1, grad_alpha1,
                   Newton_iterations);
    }
  }
}
