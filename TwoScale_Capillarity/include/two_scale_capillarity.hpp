// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2025
//
#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/io/hdf5.hpp>

#include <filesystem>
namespace fs = std::filesystem;

/*--- Add header file for the multiresolution ---*/
#include <samurai/mr/adapt.hpp>
#include "prediction.hpp"

/*--- Add header with auxiliary structs ---*/
#include "containers.hpp"

/*--- Add user implemented boundary condition ---*/
#include "user_bc.hpp"

/*--- Include the headers with the numerical fluxes ---*/
//#define RUSANOV_FLUX
#define GODUNOV_FLUX
//#define HLLC_FLUX

#ifdef RUSANOV_FLUX
  #include "Rusanov_flux.hpp"
#elifdef GODUNOV_FLUX
  #include "Exact_Godunov_flux.hpp"
#elifdef HLLC_FLUX
  #include "HLLC_flux.hpp"
#endif
#include "SurfaceTension_flux.hpp"

// Specify the use of this namespace where we just store the indices
// and, in this case, some parameters related to EOS
using namespace EquationData;

/*--- Define preprocessor to check whether to control data or not ---*/
#define VERBOSE

/** This is the class for the simulation for the two-scale capillarity model
 */
template<std::size_t dim>
class TwoScaleCapillarity {
public:
  using Config = samurai::MRConfig<dim, 2, 2, 0>;
  using Field  = samurai::VectorField<samurai::MRMesh<Config>, double, EquationData::NVARS, false>;
  using Number = samurai::Flux<Field>::Number; /*--- Define the shortcut for the arithmetic type ---*/

  TwoScaleCapillarity() = default; /*--- Default constructor. This will do nothing
                                         and basically will never be used ---*/

  TwoScaleCapillarity(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                      const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                      const Simulation_Paramaters<Number>& sim_param,
                      const EOS_Parameters<Number>& eos_param); /*--- Class constructor with the arguments related
                                                                      to the grid, to the physics, and to the relaxation. ---*/

  void run(const unsigned nfiles = 10); /*--- Function which actually executes the temporal loop ---*/

  template<class... Variables>
  void save(const std::string& suffix,
            const Variables&... fields); /*--- Routine to save the results ---*/

private:
  /*--- Now we declare some relevant variables ---*/
  const samurai::Box<double, dim> box;

  samurai::MRMesh<Config> mesh; /*--- Variable to store the mesh ---*/

  using Field_Scalar       = samurai::ScalarField<decltype(mesh), Number>;
  using Field_Vect         = samurai::VectorField<decltype(mesh), Number, dim, false>;
  using Field_ScalarVector = samurai::VectorField<decltype(mesh), Number, 1, false>;

  const Number t0; /*--- Initial time of the simulation ---*/
  const Number Tf; /*--- Final time of the simulation ---*/

  const Number sigma; /*--- Surface tension coefficient ---*/

  bool apply_relax; /*--- Choose whether to apply or not the relaxation ---*/

  const bool   mass_transfer;  /*--- Choose wheter to apply or not the mass transfer ---*/
  const Number Hmax;           /*--- Threshold length scale ---*/
  const Number kappa;          /*--- Parameter related to the radius of small-scale droplets ---*/
  const Number alpha1d_max;    /*--- Maximum threshold of small-scale volume fraction ---*/
  const Number alpha1_bar_min; /*--- Minimum effective volume fraction to identify the mixture region ---*/
  const Number alpha1_bar_max; /*--- Maximum effective volume fraction to identify the mixture region ---*/

  Number cfl; /*--- Courant number of the simulation so as to compute the time step ---*/

  const Number mod_grad_alpha1_bar_min; /*--- Minimum threshold for which not computing anymore the unit normal ---*/

  const Number      lambda;           /*--- Parameter for bound preserving strategy ---*/
  const Number      atol_Newton;      /*--- Absolute tolerance Newton method relaxation ---*/
  const Number      rtol_Newton;      /*--- Relative tolerance Newton method relaxation ---*/
  const std::size_t max_Newton_iters; /*--- Maximum number of Newton iterations ---*/

  double MR_param;      /*--- Multiresolution parameter ---*/
  double MR_regularity; /*--- Multiresolution regularity ---*/

  LinearizedBarotropicEOS<Number> EOS_phase1,
                                  EOS_phase2; /*--- The two variables which take care of the
                                                    barotropic EOS to compute the speed of sound ---*/

  #ifdef RUSANOV_FLUX
    samurai::RusanovFlux<Field> Rusanov_flux; /*--- Auxiliary variable to compute the flux for the hyperbolic operator ---*/
  #elifdef GODUNOV_FLUX
    samurai::GodunovFlux<Field> Godunov_flux; /*--- Auxiliary variable to compute the flux for the hyperbolic operator ---*/
  #elifdef HLLC_FLUX
    samurai::HLLCFlux<Field> HLLC_flux; /*--- Auxiliary variable to compute the flux for the hyperbolic operator ---*/
  #endif
  samurai::SurfaceTensionFlux<Field> SurfaceTension_flux; /*--- Auxiliary variable to compute the contribution associated to surface tension ---*/

  fs::path    path;     /*--- Auxiliary variable to store the output directory ---*/
  std::string filename; /*--- Auxiliary variable to store the name of output ---*/

  Field conserved_variables; /*--- The variable which stores the conserved variables,
                                   namely the varialbes for which we solve a PDE system ---*/

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
               alpha1;

  Field_Vect grad_alpha1_d,
             vel,
             grad_alpha1;

  Field_ScalarVector H_bar,
                     div_vel;

  samurai::ScalarField<decltype(mesh), std::size_t> to_be_relaxed;
  samurai::ScalarField<decltype(mesh), std::size_t> Newton_iterations;

  using gradient_type = decltype(samurai::make_gradient_order2<decltype(alpha1_bar)>());
  gradient_type gradient;

  using divergence_type = decltype(samurai::make_divergence_order2<decltype(normal)>());
  divergence_type divergence;

  /*--- Auxiliary output streams for post-processing ---*/
  std::ofstream Hlig;
  std::ofstream m1_integral;
  std::ofstream m1_d_integral;
  std::ofstream alpha1_bar_integral;
  std::ofstream grad_alpha1_bar_integral;
  std::ofstream Sigma_d_integral;
  std::ofstream alpha1_d_integral;
  std::ofstream grad_alpha1_d_integral;
  std::ofstream grad_alpha1_integral;
  std::ofstream grad_alpha1_tot_integral;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void update_geometry(); /*--- Auxiliary routine to compute normals and curvature ---*/

  void create_fields(); /*--- Auxiliary routine to initialize the fields to the mesh ---*/

  void init_variables(const Number x0, const Number y0,
                      const Number U0, const Number U1,
                      const Number V0,
                      const Number R, const Number eps_over_R,
                      const Number alpha_residual); /*--- Routine to initialize the variables
                                                          (both conserved and auxiliary, this is problem dependent) ---*/

  void apply_bcs(const Number U0,
                 const Number V0,
                 const Number alpha_residual); /*--- Auxiliary routine for the boundary conditions ---*/

  Number get_max_lambda(); /*--- Compute the estimate of the maximum eigenvalue ---*/

  void check_data(unsigned flag = 0); /*--- Auxiliary routine to check if spurious values are present ---*/

  void perform_mesh_adaptation(); /*--- Perform the mesh adaptation ---*/

  void apply_relaxation(); /*--- Apply the relaxation ---*/

  template<typename State, typename Gradient>
  void perform_Newton_step_relaxation(State local_conserved_variables,
                                      const Number H_bar_loc,
                                      Number& dalpha1_bar_loc,
                                      Number& alpha1_bar_loc,
                                      std::size_t& to_be_relaxed_loc,
                                      std::size_t& Newton_iterations_loc,
                                      bool& local_relaxation_applied,
                                      const Gradient& grad_alpha1_bar_loc,
                                      const bool mass_transfer_NR);

  void execute_postprocess(const Number time); /*--- Execute the postprocess ---*/
};

//////////////////////////////////////////////////////////////
/*---- START WITH THE IMPLEMENTATION OF THE CONSTRUCTOR ---*/
//////////////////////////////////////////////////////////////

// Implement class constructor
//
template<std::size_t dim>
TwoScaleCapillarity<dim>::TwoScaleCapillarity(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                                              const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                                              const Simulation_Paramaters<Number>& sim_param,
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
  #ifdef RUSANOV_FLUX
    Rusanov_flux(EOS_phase1, EOS_phase2,
                 sigma, mod_grad_alpha1_bar_min,
                 lambda, atol_Newton, rtol_Newton, max_Newton_iters),
  #elifdef GODUNOV_FLUX
    Godunov_flux(EOS_phase1, EOS_phase2,
                 sigma, mod_grad_alpha1_bar_min,
                 lambda, atol_Newton, rtol_Newton, max_Newton_iters,
                 sim_param.atol_Newton_p_star, sim_param.rtol_Newton_p_star,
                 sim_param.tol_Newton_alpha1_d),
  #elifdef HLLC_FLUX
    HLLC_flux(EOS_phase1, EOS_phase2,
              sigma, mod_grad_alpha1_bar_min,
              lambda, atol_Newton, rtol_Newton, max_Newton_iters),
  #endif
  SurfaceTension_flux(EOS_phase1, EOS_phase2,
                      sigma, mod_grad_alpha1_bar_min,
                      lambda, atol_Newton, rtol_Newton, max_Newton_iters),
  gradient(samurai::make_gradient_order2<decltype(alpha1_bar)>()),
  divergence(samurai::make_divergence_order2<decltype(normal)>())
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
      std::cout << "Initializing variables " << std::endl;
      std::cout << std::endl;
    }

    /*--- Attach the fields to the mesh ---*/
    create_fields();

    /*--- Initialize the fields ---*/
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
      /*--- TO DO: Likely periodic bcs will not work ---*/
    }

    /*--- Apply boundary conditions ---*/
    apply_bcs(sim_param.U0, sim_param.V0, sim_param.alpha_residual);
  }

// Auxiliary routine to create the fields
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::create_fields() {
  conserved_variables = samurai::make_vector_field<Number, Field::n_comp>("conserved", mesh);

  alpha1_bar      = samurai::make_scalar_field<Number>("alpha1_bar", mesh);
  grad_alpha1_bar = samurai::make_vector_field<Number, dim>("grad_alpha1_bar", mesh);
  normal          = samurai::make_vector_field<Number, dim>("normal", mesh);
  H_bar           = samurai::make_vector_field<Number, 1>("H_bar", mesh);

  dalpha1_bar     = samurai::make_scalar_field<Number>("dalpha1_bar", mesh);

  p1              = samurai::make_scalar_field<Number>("p1", mesh);
  p2              = samurai::make_scalar_field<Number>("p2", mesh);
  p_bar           = samurai::make_scalar_field<Number>("p_bar", mesh);

  alpha1_d        = samurai::make_scalar_field<Number>("alpha1_d", mesh);
  grad_alpha1_d   = samurai::make_vector_field<Number, dim>("grad_alpha1_d", mesh);
  vel             = samurai::make_vector_field<Number, dim>("vel", mesh);
  div_vel         = samurai::make_vector_field<Number, 1>("div_vel", mesh);
  Dt_alpha1_d     = samurai::make_scalar_field<Number>("Dt_alpha1_d", mesh);
  CV_alpha1_d     = samurai::make_scalar_field<Number>("CV_alpha1_d", mesh);
  alpha1          = samurai::make_scalar_field<Number>("alpha1", mesh);
  grad_alpha1     = samurai::make_vector_field<Number, dim>("grad_alpha1", mesh);

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
  /*--- Resize the fields since now mesh has been created ---*/
  conserved_variables.resize();
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

  /*--- Declare some constant parameters associated to the initial state ---*/
  const auto eps_R = eps_over_R*R;

  /*--- Initialize some fields to define the liquid column with a loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Set large-scale volume fraction
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);
                              const auto y      = static_cast<Number>(center[1]);
                              const auto r      = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
                              const auto w      = (r >= R && r < R + eps_R) ?
                                                  std::max(std::exp(static_cast<Number>(2.0)*
                                                                    (r - R)*(r - R)/(eps_R*eps_R)*
                                                                    ((r - R)*(r - R)/(eps_R*eps_R) - static_cast<Number>(3.0))/
                                                                    (((r - R)*(r - R)/(eps_R*eps_R) - static_cast<Number>(1.0))*
                                                                     ((r - R)*(r - R)/(eps_R*eps_R) - static_cast<Number>(1.0)))),
                                                           static_cast<Number>(0.0)) :
                                                  ((r < R) ? static_cast<Number>(1.0) :
                                                             static_cast<Number>(0.0));

                              alpha1_bar[cell] = std::min(std::max(alpha_residual, w), 1.0 - alpha_residual);
                            }
                        );

  /*--- Compute the geometrical quantities ---*/
  update_geometry();

  /*--- Loop over a cell to complete the remaining variables ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Set small-scale variables
                              conserved_variables[cell][ALPHA1_D_INDEX] = static_cast<Number>(0.0);
                              alpha1_d[cell]                            = conserved_variables[cell][ALPHA1_D_INDEX];
                              conserved_variables[cell][SIGMA_D_INDEX]  = static_cast<Number>(0.0);
                              conserved_variables[cell][M1_D_INDEX]     = conserved_variables[cell][ALPHA1_D_INDEX]*EOS_phase1.get_rho0();

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
                                if(r >= R && r < R + eps_R && !std::isnan(H_bar[cell][0])) {
                                  p1[cell] += sigma*H_bar[cell][0];
                                }
                                else {
                                  p1[cell] += sigma/R;
                                }
                              }
                              const auto rho1 = EOS_phase1.rho_value(p1[cell]);

                              alpha1[cell] = alpha1_bar[cell]*
                                             (static_cast<Number>(1.0) - conserved_variables[cell][ALPHA1_D_INDEX]);
                              conserved_variables[cell][M1_INDEX] = alpha1[cell]*rho1;

                              // Set mass phase 2
                              p2[cell] = EOS_phase2.get_p0();
                              const auto rho2 = EOS_phase2.rho_value(p2[cell]);

                              const auto alpha2 = static_cast<Number>(1.0)
                                                - alpha1[cell]
                                                - conserved_variables[cell][ALPHA1_D_INDEX];
                              conserved_variables[cell][M2_INDEX] = alpha2*rho2;

                              // Save mixture pressure for post-processing
                              p_bar[cell] = alpha1_bar[cell]*p1[cell]
                                          + (static_cast<Number>(1.0) - alpha1_bar[cell])*p2[cell];

                              // Set conserved variable associated to large-scale volume fraction
                              const auto rho = conserved_variables[cell][M1_INDEX]
                                             + conserved_variables[cell][M2_INDEX]
                                             + conserved_variables[cell][M1_D_INDEX];

                              conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = rho*alpha1_bar[cell];

                              // Set momentum
                              conserved_variables[cell][RHO_U_INDEX]     = conserved_variables[cell][M1_INDEX]*U1
                                                                         + conserved_variables[cell][M2_INDEX]*U0;
                              conserved_variables[cell][RHO_U_INDEX + 1] = rho*V0;

                              // Save velocity for post-processing
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel[cell][d] = conserved_variables[cell][RHO_U_INDEX + d]/rho;
                              }
                            }
                        );

  /*--- Set useful small-scale related fields ---*/
  samurai::update_ghost_mr(alpha1_d);
  grad_alpha1_d = gradient(alpha1_d);

  samurai::update_ghost_mr(vel);
  div_vel = divergence(vel);

  /*--- Set auxiliary gradient large-scale volume fraction ---*/
  samurai::update_ghost_mr(alpha1);
  grad_alpha1 = gradient(alpha1);
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

//////////////////////////////////////////////////////////////
/*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/
/////////////////////////////////////////////////////////////

// Auxiliary routine to compute normals and curvature
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::update_geometry() {
  samurai::update_ghost_mr(alpha1_bar);

  grad_alpha1_bar = gradient(alpha1_bar);

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

  vel.resize();

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
                              const auto m1_loc       = conserved_variables[cell][M1_INDEX];
                              const auto m2_loc       = conserved_variables[cell][M2_INDEX];
                              const auto m1_d_loc     = conserved_variables[cell][M1_D_INDEX];
                              const auto alpha1_d_loc = conserved_variables[cell][ALPHA1_D_INDEX];

                              /*--- Compute the velocity along all the directions ---*/
                              const auto rho_loc     = m1_loc + m2_loc + m1_d_loc;
                              const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel[cell][d] = conserved_variables[cell][RHO_U_INDEX + d]*inv_rho_loc;
                              }

                              /*--- Compute frozen speed of sound ---*/
                              const auto alpha1_loc       = alpha1_bar[cell]*
                                                            (static_cast<Number>(1.0) - alpha1_d_loc);
                              const auto rho1_loc         = m1_loc/alpha1_loc;
                                                          /*--- TODO: Add a check in case of zero volume fraction ---*/
                              const auto alpha2_loc       = static_cast<Number>(1.0) - alpha1_loc - alpha1_d_loc;
                              const auto rho2_loc         = m2_loc/alpha2_loc;
                                                            /*--- TODO: Add a check in case of zero volume fraction ---*/
                              const auto rhoc_squared_loc = m1_loc*EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)
                                                          + m2_loc*EOS_phase2.c_value(rho2_loc)*EOS_phase2.c_value(rho2_loc);
                              const auto c_loc            = std::sqrt(rhoc_squared_loc*inv_rho_loc)/
                                                            (static_cast<Number>(1.0) - alpha1_d_loc);

                              /*--- Add term due to surface tension ---*/
                              auto mod2_grad_alpha1_bar_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                mod2_grad_alpha1_bar_loc += grad_alpha1_bar[cell][d]*grad_alpha1_bar[cell][d];
                              }
                              const auto mod_grad_alpha1_bar_loc = std::sqrt(mod2_grad_alpha1_bar_loc);

                              const auto r = sigma*mod_grad_alpha1_bar_loc/(rho_loc*c_loc*c_loc);

                              /*--- Update eigenvalue estimate ---*/
                              for(std::size_t d = 0; d < dim; ++d) {
                                local_res = std::max(local_res,
                                                     std::abs(vel[cell][d]) + c_loc*(static_cast<Number>(1.0) +
                                                                                     static_cast<Number>(0.125)*r));
                              }
                            }
                        );

  double global_res;
  MPI_Allreduce(&local_res, &global_res, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return global_res;
}

// Perform the mesh adaptation strategy.
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::perform_mesh_adaptation() {
  /*auto prediction_fn = [&](auto& new_field, const auto& old_field) {
    return make_field_operator_function<TwoScaleCapillarity_prediction_op>(new_field, old_field);
  };*/

  auto MRadaptation = samurai::make_MRAdapt(/*prediction_fn,*/ conserved_variables);
  auto mra_config   = samurai::mra_config();
  mra_config.epsilon(MR_param);
  mra_config.regularity(MR_regularity);
  MRadaptation(mra_config);

  /*--- Sanity check after mesh adaptation ---*/
  alpha1_bar.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              alpha1_bar[cell] = conserved_variables[cell][RHO_ALPHA1_BAR_INDEX]/
                                                 (conserved_variables[cell][M1_INDEX] +
                                                  conserved_variables[cell][M2_INDEX] +
                                                  conserved_variables[cell][M1_D_INDEX]);
                            }
                        );
  #ifdef VERBOSE
    check_data(1);
  #endif
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
    op = "after mesh adptation";
  }

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
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
                              const auto m1_loc = conserved_variables[cell][M1_INDEX];
                              if(m1_loc < static_cast<Number>(0.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative large-scale mass of phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }
                              else if(std::isnan(m1_loc)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN large-scale mass of phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }

                              // Sanity check for m2
                              const auto m2_loc = conserved_variables[cell][M2_INDEX];
                              if(m2_loc < static_cast<Number>(0.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative mass of phase 2 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }
                              else if(std::isnan(m2_loc)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN mass of phase 2 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }

                              // Sanity check for m1_d
                              const auto m1_d_loc = conserved_variables[cell][M1_D_INDEX];
                              if(m1_d_loc < static_cast<Number>(-1e-15)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative small-scale mass of phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }
                              else if(std::isnan(m1_d_loc)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN small-scale mass of phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }

                              // Sanity check for alpha1_d
                              const auto alpha1_d_loc = conserved_variables[cell][ALPHA1_D_INDEX];
                              if(alpha1_d_loc > static_cast<Number>(1.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Exceeding value of small-scale volume fraction " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }
                              else if(alpha1_d_loc < static_cast<Number>(-1e-15)) {
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
                              const auto Sigma_d_loc = conserved_variables[cell][SIGMA_D_INDEX];
                              if(Sigma_d_loc < static_cast<Number>(-1e-15)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative small-scale interfacial area " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }
                              else if(std::isnan(Sigma_d_loc)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN small-scale interfacial area " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1_bar);
                                exit(1);
                              }
                            }
                        );
}

//////////////////////////////////////////////////////////////
/*---- FOCUS NOW ON THE RELAXATION FUNCTIONS ---*/
/////////////////////////////////////////////////////////////

// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::apply_relaxation() {
  /*--- Initialize the variables ---*/
  samurai::times::timers.start("apply_relaxation");

  std::size_t Newton_iter = 0;
  Newton_iterations.fill(0);
  dalpha1_bar.fill(std::numeric_limits<Number>::infinity());
  bool global_relaxation_applied = true;
  bool mass_transfer_NR          = mass_transfer; // In principle we might think to disable it after a certain
                                                  // number of iterations (as in Arthur's code), not done here.

  samurai::times::timers.stop("apply_relaxation");

  /*--- Loop of Newton method. Conceptually, a loop over cells followed by a Newton loop
        over each cell would (could?) be more logic, but this would lead to issues to call 'update_geometry' ---*/
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
                                                              H_bar[cell][0], dalpha1_bar[cell], alpha1_bar[cell],
                                                              to_be_relaxed[cell], Newton_iterations[cell], local_relaxation_applied,
                                                              grad_alpha1_bar[cell], mass_transfer_NR);
                             }
                             catch(std::exception& e) {
                               std::cerr << e.what() << std::endl;
                               save("_diverged",
                                    conserved_variables,
                                    alpha1_bar, dalpha1_bar, grad_alpha1_bar, normal, H_bar,
                                    to_be_relaxed, Newton_iterations);
                               exit(1);
                             }
                           });

    mpi::communicator world;
    boost::mpi::all_reduce(world, local_relaxation_applied, global_relaxation_applied, std::logical_or<bool>());

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
template<typename State, typename Gradient>
void TwoScaleCapillarity<dim>::perform_Newton_step_relaxation(State local_conserved_variables,
                                                              const Number H_bar_loc,
                                                              Number& dalpha1_bar_loc,
                                                              Number& alpha1_bar_loc,
                                                              std::size_t& to_be_relaxed_loc,
                                                              std::size_t& Newton_iterations_loc,
                                                              bool& local_relaxation_applied,
                                                              const Gradient& grad_alpha1_bar_loc,
                                                              const bool mass_transfer_NR) {
  to_be_relaxed_loc = 0;

  if(!std::isnan(H_bar_loc)) {
    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m1_loc       = local_conserved_variables(M1_INDEX);
    const auto m2_loc       = local_conserved_variables(M2_INDEX);
    const auto m1_d_loc     = local_conserved_variables(M1_D_INDEX);
    const auto alpha1_d_loc = local_conserved_variables(ALPHA1_D_INDEX);

    /*--- Update auxiliary values affected by the nonlinear function for which we seek a zero ---*/
    const auto alpha1_loc = alpha1_bar_loc*(static_cast<Number>(1.0) - alpha1_d_loc);
    const auto rho1_loc   = m1_loc/alpha1_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p1_loc     = EOS_phase1.pres_value(rho1_loc);

    const auto alpha2_loc = static_cast<Number>(1.0) - alpha1_loc - alpha1_d_loc;
    const auto rho2_loc   = m2_loc/alpha2_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p2_loc     = EOS_phase2.pres_value(rho2_loc);

    const auto rho1d_loc     = (m1_d_loc > static_cast<Number>(0.0) &&
                               alpha1_d_loc > static_cast<Number>(0.0)) ?
                               m1_d_loc/alpha1_d_loc : EOS_phase1.get_rho0();
    const auto inv_rho1d_loc = static_cast<Number>(1.0)/rho1d_loc;

    /*--- Prepare for mass transfer if desired ---*/
    const auto rho_loc = m1_loc + m2_loc + m1_d_loc;

    // Compute first ordrer integral reminder "specific enthalpy"
    auto alpha2_bar_loc  = static_cast<Number>(1.0) - alpha1_bar_loc;
    const auto p_bar_loc = alpha1_bar_loc*p1_loc
                         + alpha2_bar_loc*p2_loc;
    Number p2_minus_p1_times_theta;
    try {
      p2_minus_p1_times_theta = rho1_loc/(static_cast<Number>(1.0) - alpha1_bar_loc)*
                                (EOS_phase1.e_value(rho1d_loc) - EOS_phase1.e_value(rho1_loc) +
                                 p_bar_loc*inv_rho1d_loc - p1_loc/rho1_loc) -
                                (p2_loc - p1_loc);
    }
    catch(std::exception& e) {
      std::cerr << e.what() << std::endl;
      throw std::runtime_error("Error when computing the internal energy");
    }
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

    /*--- Compute the nonlinear function for which we seek the zero (basically the Laplace law) ---*/
    const auto F = (static_cast<Number>(1.0) - alpha1_d_loc)*(p1_loc - p2_loc)
                 - sigma*H_lim;

    /*--- Perform the relaxation only where really needed ---*/
    if(std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase1.get_p0(), sigma*std::abs(H_lim)) &&
       std::abs(dalpha1_bar_loc) > atol_Newton) {
      to_be_relaxed_loc = 1;
      Newton_iterations_loc++;
      local_relaxation_applied = true;

      // Compute the derivative w.r.t large scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
      const auto dF_dalpha1_bar = -m1_loc/(alpha1_bar_loc*alpha1_bar_loc)*
                                   EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)
                                  -m2_loc/(alpha2_bar_loc*alpha2_bar_loc)*
                                   EOS_phase2.c_value(rho2_loc)*EOS_phase2.c_value(rho2_loc);

      // Compute the pseudo time step starting as initial guess from the ideal unmodified Newton method
      auto dtau_ov_epsilon = std::numeric_limits<Number>::infinity();

      // Bound preserving condition for m1, velocity and small-scale volume fraction
      if(dH > static_cast<Number>(0.0)) {
        // Bound preserving condition for m1
        dtau_ov_epsilon = lambda*(alpha1_loc*alpha2_bar_loc)/(sigma*dH);
        if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
          throw std::runtime_error("Negative time step found after relaxation of mass of large-scale phase 1");
        }

        // Bound preserving for the velocity
        const auto mom_dot_vel   = (local_conserved_variables(RHO_U_INDEX)*local_conserved_variables(RHO_U_INDEX) +
                                    local_conserved_variables(RHO_U_INDEX + 1)*local_conserved_variables(RHO_U_INDEX + 1))/rho_loc;
        auto dtau_ov_epsilon_tmp = lambda*mom_dot_vel/(dH*fac_Ru*sigma);
        dtau_ov_epsilon          = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
        if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
          throw std::runtime_error("Negative time step found after relaxation of velocity");
        }

        // Bound preserving for the small-scale volume fraction
        dtau_ov_epsilon_tmp = lambda*(alpha1d_max - alpha1_d_loc)*alpha2_bar_loc*rho1d_loc/
                                     (rho1_loc*sigma*dH);
        dtau_ov_epsilon     = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
        if(alpha1_d_loc > static_cast<Number>(0.0)) {
          dtau_ov_epsilon_tmp = lambda*alpha1_d_loc*alpha2_bar_loc*rho1d_loc/
                                       (rho1_loc*sigma*dH);

          dtau_ov_epsilon     = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
        }
        if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
          throw std::runtime_error("Negative time step found after relaxation of small-scale volume fraction");
        }
      }

      // Bound preserving condition for large-scale volume fraction
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
        dtau_ov_epsilon_tmp = 0.5*(-b - std::sqrt(D))/a;
      }
      if(a == static_cast<Number>(0.0) &&
         b < static_cast<Number>(0.0)) {
        dtau_ov_epsilon_tmp = -lambda*alpha1_bar_loc/b;
      }
      dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
      if(dtau_ov_epsilon < static_cast<Number>(0.0)) {
        throw std::runtime_error("Negative time step found after relaxation of large-scale volume fraction");
      }

      // Compute the effective variation of the variables
      if(std::isinf(dtau_ov_epsilon)) {
        // If we are in this branch we do not have mass transfer
        // and we do not have other restrictions on the bounds of large scale volume fraction
        dalpha1_bar_loc = -F/dF_dalpha1_bar;

        /*if(dalpha1_bar_loc > static_cast<Number>(0.0)) {
          dalpha1_bar_loc = std::min(-F/dF_dalpha1_bar, lambda*alpha2_bar_loc);
        }
        else if(dalpha1_bar_loc < static_cast<Number>(0.0)) {
          dalpha1_bar_loc = std::max(-F/dF_dalpha1_bar, -lambda*alpha1_bar_loc);
        }*/
      }
      else {
        const auto dm1 = -dtau_ov_epsilon/alpha2_bar_loc*
                          (m1_loc/(alpha1_bar_loc*(static_cast<Number>(1.0) - alpha1_d_loc)))*
                          sigma*dH;

        const auto num_dalpha1_bar = dtau_ov_epsilon/(static_cast<Number>(1.0) - alpha1_d_loc);
        const auto den_dalpha1_bar = static_cast<Number>(1.0) - num_dalpha1_bar*dF_dalpha1_bar;
        dalpha1_bar_loc            = (num_dalpha1_bar/den_dalpha1_bar)*(F - dm1*R);

        if(dm1 > static_cast<Number>(0.0)) {
          throw std::runtime_error("Negative sign of mass transfer inside Newton step");
        }
        else {
          local_conserved_variables(M1_INDEX) += dm1;
          if(local_conserved_variables(M1_INDEX) < static_cast<Number>(0.0)) {
            throw std::runtime_error("Negative mass of large-scale phase 1 inside Newton step");
          }

          local_conserved_variables(M1_D_INDEX) -= dm1;
          if(local_conserved_variables(M1_D_INDEX) < static_cast<Number>(0.0)) {
            throw std::runtime_error("Negative mass of small-scale phase 1 inside Newton step");
          }
        }

        if(alpha1_d_loc - dm1*inv_rho1d_loc > static_cast<Number>(1.0)) {
          throw std::runtime_error("Exceeding value for small-scale volume fraction inside Newton step");
        }
        else {
          local_conserved_variables(ALPHA1_D_INDEX) -= dm1*inv_rho1d_loc;
        }

        local_conserved_variables(SIGMA_D_INDEX) -= dm1*static_cast<Number>(3.0)*Hmax/(kappa*rho1d_loc);
      }

      if(alpha1_bar_loc + dalpha1_bar_loc < static_cast<Number>(0.0) ||
         alpha1_bar_loc + dalpha1_bar_loc > static_cast<Number>(1.0)) {
        throw std::runtime_error("Bounds exceeding value for large-scale volume fraction inside Newton step");
      }
      else {
        alpha1_bar_loc += dalpha1_bar_loc;
      }

      if(dH > static_cast<Number>(0.0)) {
        auto drho_fac_Ru = static_cast<Number>(0.0);
        const auto mom_squared = local_conserved_variables(RHO_U_INDEX)*local_conserved_variables(RHO_U_INDEX)
                               + local_conserved_variables(RHO_U_INDEX + 1)*local_conserved_variables(RHO_U_INDEX + 1);
        if(mom_squared > static_cast<Number>(0.0)) {
          drho_fac_Ru = dtau_ov_epsilon*
                        sigma*dH*fac_Ru*rho_loc/mom_squared; /*--- u/u^{2} = rho*u/(rho*(u^{2})) = (rho/(rho*u)^{2})*(rho*u) ---*/
        }

        for(std::size_t d = 0; d < Field::dim; ++d) {
          local_conserved_variables(RHO_U_INDEX + d) -= drho_fac_Ru*local_conserved_variables(RHO_U_INDEX + d);
        }
      }
    }

    // Update "conservative counter part" of large-scale volume fraction.
    // Do it outside because this can change either because of relaxation or
    // alpha1_bar.
    local_conserved_variables(RHO_ALPHA1_BAR_INDEX) = rho_loc*alpha1_bar_loc;
  }
}

//////////////////////////////////////////////////////////////
/*---- FOCUS NOW ON THE POST-PROCESSING FUNCTIONS ---*/
/////////////////////////////////////////////////////////////

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void TwoScaleCapillarity<dim>::save(const std::string& suffix,
                                    const Variables&... fields) {
  auto level_ = samurai::make_scalar_field<std::size_t>("level", mesh);

  if(!fs::exists(path)) {
    fs::create_directory(path);
  }

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              level_[cell] = cell.level;
                            }
                        );

  samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, fields..., level_);
  samurai::dump(path, fmt::format("{}{}", filename, "_restart"), mesh, fields..., level_);
}

// Execute postprocessing
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::execute_postprocess(const Number time) {
  /*--- Initialize relevant integral quantities ---*/
  auto local_H_lig               = static_cast<Number>(0.0);
  auto local_m1_int              = static_cast<Number>(0.0);
  auto local_m1_d_int            = static_cast<Number>(0.0);
  auto local_alpha1_bar_int      = static_cast<Number>(0.0);
  auto local_grad_alpha1_bar_int = static_cast<Number>(0.0);
  auto local_Sigma_d_int         = static_cast<Number>(0.0);
  auto local_alpha1_d_int        = static_cast<Number>(0.0);
  auto local_grad_alpha1_d_int   = static_cast<Number>(0.0);
  auto local_grad_alpha1_int     = static_cast<Number>(0.0);
  auto local_grad_alpha1_tot_int = static_cast<Number>(0.0);

  alpha1.resize();
  alpha1_d.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Save liquidi large- and small-scale variables
                              const auto alpha1_d_loc = conserved_variables[cell][ALPHA1_D_INDEX];
                              alpha1[cell]            = alpha1_bar[cell]*(static_cast<Number>(1.0) - alpha1_d_loc);
                              alpha1_d[cell]          = alpha1_d_loc;
                            }
                        );
  samurai::update_ghost_mr(alpha1);
  grad_alpha1.resize();
  grad_alpha1 = gradient(alpha1);
  samurai::update_ghost_mr(alpha1_d);
  grad_alpha1_d.resize();
  grad_alpha1_d = gradient(alpha1_d);

  p1.resize();
  p2.resize();
  p_bar.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Pre-fetch some variables used multiple times in order to exploit possible vectorization
                              const auto m1_loc         = conserved_variables[cell][M1_INDEX];
                              const auto m2_loc         = conserved_variables[cell][M2_INDEX];
                              const auto m1_d_loc       = conserved_variables[cell][M1_D_INDEX];
                              const auto alpha1_d_loc   = conserved_variables[cell][ALPHA1_D_INDEX];
                              const auto alpha1_loc     = alpha1[cell];
                              const auto alpha1_bar_loc = alpha1_bar[cell];
                              const auto alpha2_bar_loc = static_cast<Number>(1.0) - alpha1_bar_loc;
                              const auto H_bar_loc      = H_bar[cell][0];

                              // Compue H_lig
                              const auto rho1_loc  = m1_loc/alpha1_loc;
                                                    /*--- TODO: Add a check in case of zero volume fraction ---*/
                              const auto rho1d_loc = (alpha1_d_loc > static_cast<Number>(0.0)) ?
                                                     m1_d_loc/alpha1_d_loc : EOS_phase1.get_rho0();
                              const auto p1_loc    = EOS_phase1.pres_value(rho1_loc);
                              p1[cell]             = p1_loc;
                              const auto rho2_loc  = m2_loc/(static_cast<Number>(1.0) - alpha1_loc - alpha1_d_loc);
                                                    /*--- TODO: Add a check in case of zero volume fraction ---*/
                              const auto p2_loc    = EOS_phase2.pres_value(rho2_loc);
                              p2[cell]             = p2_loc;
                              const auto p_bar_loc = alpha1_bar[cell]*p1_loc
                                                   + alpha2_bar_loc*p2_loc;
                              p_bar[cell]          = p_bar_loc;
                              const auto H_lim_loc = std::min(H_bar_loc, Hmax);
                              Number p2_minus_p1_times_theta;
                              try {
                                p2_minus_p1_times_theta = rho1_loc/alpha2_bar_loc*
                                                          (EOS_phase1.e_value(rho1d_loc) - EOS_phase1.e_value(rho1_loc) +
                                                           p_bar_loc/rho1d_loc - p1_loc/rho1_loc) -
                                                          (p2_loc - p1_loc);
                              }
                              catch(std::exception& e) {
                                std::cerr << e.what() << std::endl;
                                exit(1);
                              }
                              const auto fac_Ru = sigma*(static_cast<Number>(3.0)*H_lim_loc/(kappa*rho1d_loc))*
                                                        (rho1_loc/alpha2_bar_loc)
                                                - sigma*H_lim_loc/(static_cast<Number>(1.0) - alpha1_d_loc)
                                                + p2_minus_p1_times_theta;
                              if(fac_Ru > static_cast<Number>(0.0) &&
                                 alpha1_bar_loc > alpha1_bar_min && alpha1_bar_loc < alpha1_bar_max &&
                                 -grad_alpha1_bar[cell][0]*conserved_variables[cell][RHO_U_INDEX]
                                 -grad_alpha1_bar[cell][1]*conserved_variables[cell][RHO_U_INDEX + 1] > static_cast<Number>(0.0) &&
                                alpha1_d_loc < alpha1d_max) {
                                local_H_lig = std::max(H_bar_loc, local_H_lig);
                              }

                              // Compute the integral quantities
                              auto cell_volume = static_cast<Number>(cell.length);
                              for(std::size_t d = 1; d < dim; ++d) {
                                cell_volume *= static_cast<Number>(cell.length);
                              }
                              local_m1_int += m1_loc*cell_volume;
                              local_m1_d_int += m1_d_loc*cell_volume;
                              local_alpha1_bar_int += alpha1_bar_loc*cell_volume;
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
                              local_grad_alpha1_bar_int += std::sqrt(mod2_grad_alpha1_bar_loc)*cell_volume;
                              local_Sigma_d_int += conserved_variables[cell][SIGMA_D_INDEX]*cell_volume;
                              local_grad_alpha1_d_int += std::sqrt(mod2_grad_alpha1_d_loc)*cell_volume;
                              local_alpha1_d_int += alpha1_d_loc*cell_volume;
                              local_grad_alpha1_int += std::sqrt(mod2_grad_alpha1_loc)*cell_volume;
                              local_grad_alpha1_tot_int += std::sqrt(mod2_grad_alpha1_tot_loc)*cell_volume;
                            }
                        );

  /*--- Perform MPI collective operations ---*/
  double global_H_lig;
  MPI_Allreduce(&local_H_lig, &global_H_lig, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  double global_m1_int;
  MPI_Allreduce(&local_m1_int, &global_m1_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double global_m1_d_int;
  MPI_Allreduce(&local_m1_d_int, &global_m1_d_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double global_alpha1_bar_int;
  MPI_Allreduce(&local_alpha1_bar_int, &global_alpha1_bar_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double global_grad_alpha1_bar_int;
  MPI_Allreduce(&local_grad_alpha1_bar_int, &global_grad_alpha1_bar_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double global_Sigma_d_int;
  MPI_Allreduce(&local_Sigma_d_int, &global_Sigma_d_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double global_alpha1_d_int;
  MPI_Allreduce(&local_alpha1_d_int, &global_alpha1_d_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double global_grad_alpha1_d_int;
  MPI_Allreduce(&local_grad_alpha1_d_int, &global_grad_alpha1_d_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double global_grad_alpha1_int;
  MPI_Allreduce(&local_grad_alpha1_int, &global_grad_alpha1_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double global_grad_alpha1_tot_int;
  MPI_Allreduce(&local_grad_alpha1_tot_int, &global_grad_alpha1_tot_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /*--- Save the data ---*/
  Hlig                     << std::fixed << std::setprecision(12) << time << '\t' << global_H_lig               << std::endl;
  m1_integral              << std::fixed << std::setprecision(12) << time << '\t' << global_m1_int              << std::endl;
  m1_d_integral            << std::fixed << std::setprecision(12) << time << '\t' << global_m1_d_int            << std::endl;
  alpha1_bar_integral      << std::fixed << std::setprecision(12) << time << '\t' << global_alpha1_bar_int      << std::endl;
  grad_alpha1_bar_integral << std::fixed << std::setprecision(12) << time << '\t' << global_grad_alpha1_bar_int << std::endl;
  Sigma_d_integral         << std::fixed << std::setprecision(12) << time << '\t' << global_Sigma_d_int         << std::endl;
  alpha1_d_integral        << std::fixed << std::setprecision(12) << time << '\t' << global_alpha1_d_int        << std::endl;
  grad_alpha1_d_integral   << std::fixed << std::setprecision(12) << time << '\t' << global_grad_alpha1_d_int   << std::endl;
  grad_alpha1_integral     << std::fixed << std::setprecision(12) << time << '\t' << global_grad_alpha1_int     << std::endl;
  grad_alpha1_tot_integral << std::fixed << std::setprecision(12) << time << '\t' << global_grad_alpha1_tot_int << std::endl;
}

//////////////////////////////////////////////////////////////
/*---- IMPLEMENT THE FUNCTION THAT EFFECTIVELY SOLVES THE PROBLEM ---*/
/////////////////////////////////////////////////////////////

// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::run(const unsigned nfiles) {
  /*--- Default output arguemnts ---*/
  path = fs::current_path();
  filename = "liquid_column";
  #ifdef RUSANOV_FLUX
    filename += "_Rusanov";
  #elifdef GODUNOV_FLUX
    filename += "_Godunov";
  #elifdef HLLC_FLUX
    filename += "_HLLC";
  #endif

  #ifdef ORDER_2
    filename += "_order2";
    #ifdef RELAX_RECONSTRUCTION
      filename += "_relaxed_reconstruction";
    #endif
  #else
    filename += "_order1";
  #endif

  if(mass_transfer)
    filename += "_mass_transfer";
  else
    filename += "_no_mass_transfer";

  const auto dt_save = Tf/static_cast<Number>(nfiles);

  /*--- Auxiliary variables to save updated fields ---*/
  #ifdef ORDER_2
    auto conserved_variables_tmp = samurai::make_vector_field<Number, Field::n_comp>("conserved_tmp", mesh);
    auto conserved_variables_old = samurai::make_vector_field<Number, Field::n_comp>("conserved_old", mesh);
  #endif
  auto conserved_variables_np1 = samurai::make_vector_field<Number, Field::n_comp>("conserved_np1", mesh);

  /*--- Create the flux variable ---*/
  #ifdef RUSANOV_FLUX
    #ifdef ORDER_2
      auto numerical_flux_hyp = Rusanov_flux.make_two_scale_capillarity(H_bar);
    #else
      auto numerical_flux_hyp = Rusanov_flux.make_two_scale_capillarity();
    #endif
  #elifdef GODUNOV_FLUX
    #ifdef ORDER_2
      auto numerical_flux_hyp = Godunov_flux.make_two_scale_capillarity(H_bar);
    #else
      auto numerical_flux_hyp = Godunov_flux.make_two_scale_capillarity();
    #endif
  #elifdef HLLC_FLUX
    #ifdef ORDER_2
      auto numerical_flux_hyp = HLLC_flux.make_two_scale_capillarity(H_bar);
    #else
      auto numerical_flux_hyp = HLLC_flux.make_two_scale_capillarity();
    #endif
  #endif
  auto numerical_flux_st = SurfaceTension_flux.make_two_scale_capillarity(grad_alpha1_bar);

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(suffix_init, conserved_variables,
                    alpha1_bar, grad_alpha1_bar, normal, H_bar,
                    p1, p2, p_bar,
                    grad_alpha1_d, vel, div_vel, alpha1, grad_alpha1);
  Hlig.open("Hlig.dat", std::ofstream::out);
  m1_integral.open("m1_integral.dat", std::ofstream::out);
  m1_d_integral.open("m1_d_integral.dat", std::ofstream::out);
  alpha1_bar_integral.open("alpha1_bar_integral.dat", std::ofstream::out);
  grad_alpha1_bar_integral.open("grad_alpha1_bar_integral.dat", std::ofstream::out);
  Sigma_d_integral.open("Sigma_d_integral.dat", std::ofstream::out);
  alpha1_d_integral.open("alpha1_d_integral.dat", std::ofstream::out);
  grad_alpha1_d_integral.open("grad_alpha1_d_integral.dat", std::ofstream::out);
  grad_alpha1_integral.open("grad_alpha1_integral.dat", std::ofstream::out);
  grad_alpha1_tot_integral.open("grad_alpha1_tot_integral.dat", std::ofstream::out);
  auto t = static_cast<Number>(t0);
  execute_postprocess(t);

  /*--- Set initial time step ---*/
  const auto dx = static_cast<Number>(mesh.cell_length(mesh.max_level()));
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  const auto n_elements_per_subdomain = mesh[mesh_id_t::cells].nb_cells();
  unsigned n_elements;
  MPI_Allreduce(&n_elements_per_subdomain, &n_elements, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "Number of initial elements = " <<  n_elements << std::endl;
    std::cout << std::endl;
  }

  /*--- Start the loop ---*/
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  while(t != Tf) {
    // Apply mesh adaptation
    perform_mesh_adaptation();

    // Compute the time step
    normal.resize();
    H_bar.resize();
    grad_alpha1_bar.resize();
    update_geometry();
    const auto dt = std::min(Tf - t, cfl*dx/get_max_lambda());
    t += dt;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
      std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;
    }

    // Save current state in case of order 2
    #ifdef ORDER_2
      conserved_variables_old.resize();
      conserved_variables_old = conserved_variables;
    #endif

    // Apply the numerical scheme without relaxation
    // Convective operator
    samurai::update_ghost_mr(conserved_variables);
    #ifdef RELAX_RECONSTRUCTION
      samurai::update_ghost_mr(H_bar);
    #endif
    try {
      auto flux_hyp = numerical_flux_hyp(conserved_variables);
      #ifdef ORDER_2
        conserved_variables_tmp.resize();
        conserved_variables_tmp = conserved_variables - dt*flux_hyp;
        std::swap(conserved_variables.array(), conserved_variables_tmp.array());
      #else
        conserved_variables_np1.resize();
        conserved_variables_np1 = conserved_variables - dt*flux_hyp;
        std::swap(conserved_variables.array(), conserved_variables_np1.array());
      #endif
    }
    catch(std::exception& e) {
      std::cerr << e.what() << std::endl;
      save("_diverged", conserved_variables, alpha1_bar);
      exit(1);
    }

    // Update the geometry to recompute volume fraction gradient
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                              {
                                alpha1_bar[cell] = conserved_variables[cell][RHO_ALPHA1_BAR_INDEX]/
                                                   (conserved_variables[cell][M1_INDEX] +
                                                    conserved_variables[cell][M2_INDEX] +
                                                    conserved_variables[cell][M1_D_INDEX]);
                              }
                          );
    #ifdef VERBOSE
      check_data();
    #endif
    update_geometry();

    // Capillarity contribution
    samurai::update_ghost_mr(conserved_variables, grad_alpha1_bar);
    auto flux_st = numerical_flux_st(conserved_variables);
    #ifdef ORDER_2
      conserved_variables_tmp = conserved_variables - dt*flux_st;
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());
    #else
      conserved_variables_np1 = conserved_variables - dt*flux_st;
      std::swap(conserved_variables.array(), conserved_variables_np1.array());
    #endif

    // Apply relaxation
    if(apply_relax) {
      // Apply relaxation if desired, which will modify alpha1_bar and, consequently, for what
      // concerns next time step, rho_alpha1_bar (as well as grad_alpha1_bar).
      dalpha1_bar.resize();
      to_be_relaxed.resize();
      Newton_iterations.resize();
      apply_relaxation();
      #ifdef RELAX_RECONSTRUCTION
        update_geometry();
      #endif
    }

    /*--- Consider the second stage for the second order ---*/
    #ifdef ORDER_2
      // Apply the numerical scheme
      // Convective operator
      samurai::update_ghost_mr(conserved_variables);
      #ifdef RELAX_RECONSTRUCTION
        samurai::update_ghost_mr(H_bar);
      #endif
      try {
        auto flux_hyp = numerical_flux_hyp(conserved_variables);
        conserved_variables_tmp = conserved_variables - dt*flux_hyp;
        std::swap(conserved_variables.array(), conserved_variables_tmp.array());
      }
      catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
        save("_diverged", conserved_variables, alpha1_bar);
        exit(1);
      }

      // Recompute geometrical quantities
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                                {
                                  alpha1_bar[cell] = conserved_variables[cell][RHO_ALPHA1_BAR_INDEX]/
                                                     (conserved_variables[cell][M1_INDEX] +
                                                      conserved_variables[cell][M2_INDEX] +
                                                      conserved_variables[cell][M1_D_INDEX]);
                                }
                            );
      #ifdef VERBOSE
        check_data();
      #endif
      update_geometry();

      // Capillarity contribution
      samurai::update_ghost_mr(conserved_variables, grad_alpha1_bar);
      flux_st = numerical_flux_st(conserved_variables);
      conserved_variables_tmp = conserved_variables - dt*flux_st;
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());

      // Apply relaxation
      if(apply_relax) {
        // Apply relaxation if desired, which will modify alpha1_bar and, consequently, for what
        // concerns next time step, rho_alpha1_bar (as well as grad_alpha1_bar).
        apply_relaxation();
      }

      // Complete evaluation
      conserved_variables_np1.resize();
      conserved_variables_np1 = static_cast<Number>(0.5)*
                                (conserved_variables_old + conserved_variables);
      std::swap(conserved_variables.array(), conserved_variables_np1.array());

      // Recompute volume fraction gradient and curvature for the next time step
      #ifdef RELAX_RECONSTRUCTION
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                                  {
                                    alpha1_bar[cell] = conserved_variables[cell][RHO_ALPHA1_BAR_INDEX]/
                                                       (conserved_variables[cell][M1_INDEX] +
                                                        conserved_variables[cell][M2_INDEX] +
                                                        conserved_variables[cell][M1_D_INDEX]);
                                  }
                              );
        update_geometry();
      #endif
    #endif

    /*--- Postprocess data ---*/
    #ifndef RELAX_RECONSTRUCTION
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                                {
                                  alpha1_bar[cell] = conserved_variables[cell][RHO_ALPHA1_BAR_INDEX]/
                                                     (conserved_variables[cell][M1_INDEX] +
                                                      conserved_variables[cell][M2_INDEX] +
                                                      conserved_variables[cell][M1_D_INDEX]);
                                }
                            );
      update_geometry();
    #endif
    execute_postprocess(t);

    /*--- Save the results ---*/
    if(t >= static_cast<Number>(nsave + 1)*dt_save || t == Tf) {
      // Resize all the fields not resized yet
      div_vel.resize();
      Dt_alpha1_d.resize();
      CV_alpha1_d.resize();

      samurai::update_ghost_mr(vel);
      div_vel = divergence(vel);

      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                                {
                                  const auto alpha1_d_loc = conserved_variables[cell][ALPHA1_D_INDEX];

                                  #ifdef ORDER_2
                                    Dt_alpha1_d[cell] = (alpha1_d_loc - conserved_variables_old[cell][ALPHA1_D_INDEX])/dt
                                                      + vel[cell][0]*grad_alpha1_d[cell][0]
                                                      + vel[cell][1]*grad_alpha1_d[cell][1];
                                  #else
                                    Dt_alpha1_d[cell] = (alpha1_d_loc - conserved_variables_np1[cell][ALPHA1_D_INDEX])/dt
                                                      + vel[cell][0]*grad_alpha1_d[cell][0]
                                                      + vel[cell][1]*grad_alpha1_d[cell][1];
                                  #endif

                                  CV_alpha1_d[cell] = Dt_alpha1_d[cell] + alpha1_d_loc*div_vel[cell][0];
                                }
                            );

      // Perform the saving
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(suffix, conserved_variables,
                   alpha1_bar, grad_alpha1_bar, normal, H_bar,
                   p1, p2, p_bar,
                   grad_alpha1_d, vel, div_vel, Dt_alpha1_d, CV_alpha1_d,
                   alpha1, grad_alpha1,
                   Newton_iterations);
    }
  }

  /*--- Close the files for post-proessing ---*/
  Hlig.close();
  m1_integral.close();
  m1_d_integral.close();
  alpha1_bar_integral.close();
  grad_alpha1_bar_integral.close();
  Sigma_d_integral.close();
  alpha1_d_integral.close();
  grad_alpha1_d_integral.close();
  grad_alpha1_integral.close();
  grad_alpha1_tot_integral.close();
}
