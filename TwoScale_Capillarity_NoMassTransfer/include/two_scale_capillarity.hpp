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
//#define GODUNOV_FLUX
#define HLLC_FLUX

#ifdef RUSANOV_FLUX
  #include "Rusanov_flux.hpp"
#elifdef GODUNOV_FLUX
  #include "Exact_Godunov_flux.hpp"
#elifdef HLLC_FLUX
 #include "HLLC_flux.hpp"
#endif
#include "SurfaceTension_flux.hpp"

/*--- Define preprocessor to check whether to control data or not ---*/
#define VERBOSE

// Specify the use of this namespace where we just store the indices
// and, in this case, some parameters related to EOS
using namespace EquationData;

// This is the class for the simulation of a two-scale model
// with capillarity
//
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
                      const EOS_Parameters<Number>& eos_param); /*--- Class constrcutor with the arguments related
                                                                      to the grid, to the physics and to the relaxation. ---*/

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

  Number cfl; /*--- Courant number of the simulation so as to compute the time step ---*/

  const Number mod_grad_alpha1_min; /*--- Minimum threshold for which not computing anymore the unit normal ---*/

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
    samurai::RusanovFlux<Field> Rusanov_flux; /*--- Auxiliary variable to compute the flux ---*/
  #elifdef GODUNOV_FLUX
    samurai::GodunovFlux<Field> Godunov_flux; /*--- Auxiliary variable to compute the flux ---*/
  #elifdef HLLC_FLUX
    samurai::HLLCFlux<Field> HLLC_flux; /*--- Auxiliary variable to compute the flux ---*/
  #endif
  samurai::SurfaceTensionFlux<Field> SurfaceTension_flux; /*--- Auxiliary variable to compute the contribution associated to surface tension ---*/

  fs::path    path;     /*--- Auxiliary variable to store the output directory ---*/
  std::string filename; /*--- Auxiliary variable to store the name of output ---*/

  Field conserved_variables; /*--- The variable which stores the conserved variables,
                                   namely the varialbes for which we solve a PDE system ---*/

  /*--- Now we declare a bunch of fields which depend from the state, but it is useful
        to have it so as to avoid recomputation ---*/
  Field_Scalar alpha1,
               dalpha1;

  Field_Vect vel,
             normal,
             grad_alpha1;

  Field_ScalarVector H;

  samurai::ScalarField<decltype(mesh), std::size_t> to_be_relaxed;
  samurai::ScalarField<decltype(mesh), std::size_t> Newton_iterations;

  using gradient_type = decltype(samurai::make_gradient_order2<decltype(alpha1)>());
  gradient_type gradient;

  using divergence_type = decltype(samurai::make_divergence_order2<decltype(normal)>());
  divergence_type divergence;

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

  void check_data(unsigned flag = 0); /*--- Auxiliary routine to check if small negative values are present ---*/

  void perform_mesh_adaptation(); /*--- Perform the mesh adaptation ---*/

  void apply_relaxation(); /*--- Apply the relaxation ---*/

  template<typename State>
  void perform_Newton_step_relaxation(State local_conserved_variables,
                                      const Number H_loc,
                                      Number& dalpha1_loc,
                                      Number& alpha1_loc,
                                      std::size_t& to_be_relaxed_loc,
                                      std::size_t& Newton_iterations_loc,
                                      bool& local_relaxation_applied);
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
  cfl(sim_param.Courant), mod_grad_alpha1_min(sim_param.mod_grad_alpha1_min),
  lambda(sim_param.lambda), atol_Newton(sim_param.atol_Newton),
  rtol_Newton(sim_param.rtol_Newton), max_Newton_iters(sim_param.max_Newton_iters),
  MR_param(sim_param.MR_param), MR_regularity(sim_param.MR_regularity),
  EOS_phase1(eos_param.p0_phase1, eos_param.rho0_phase1, eos_param.c0_phase1),
  EOS_phase2(eos_param.p0_phase2, eos_param.rho0_phase2, eos_param.c0_phase2),
  #ifdef RUSANOV_FLUX
    Rusanov_flux(EOS_phase1, EOS_phase2, sigma, mod_grad_alpha1_min,
                 lambda, atol_Newton, rtol_Newton, max_Newton_iters),
  #elifdef GODUNOV_FLUX
    Godunov_flux(EOS_phase1, EOS_phase2, sigma, mod_grad_alpha1_min,
                 lambda, atol_Newton, rtol_Newton, max_Newton_iters,
                 sim_param.atol_Newton_p_star, sim_param.rtol_Newton_p_star),
  #elifdef HLLC_FLUX
    HLLC_flux(EOS_phase1, EOS_phase2, sigma, mod_grad_alpha1_min,
              lambda, atol_Newton, rtol_Newton, max_Newton_iters),
  #endif
  SurfaceTension_flux(EOS_phase1, EOS_phase2, sigma, mod_grad_alpha1_min,
                      sim_param.lambda, sim_param.atol_Newton, sim_param.rtol_Newton,
                      max_Newton_iters),
  gradient(samurai::make_gradient_order2<decltype(alpha1)>()),
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
                                                  alpha1, grad_alpha1, normal, H);
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

  vel = samurai::make_vector_field<Number, dim>("vel", mesh);

  alpha1      = samurai::make_scalar_field<Number>("alpha1", mesh);
  grad_alpha1 = samurai::make_vector_field<Number, dim>("grad_alpha1", mesh);
  normal      = samurai::make_vector_field<Number, dim>("normal", mesh);
  H           = samurai::make_vector_field<Number, 1>("H", mesh);

  dalpha1     = samurai::make_scalar_field<Number>("dalpha1", mesh);

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
  vel.resize();
  alpha1.resize();
  grad_alpha1.resize();
  normal.resize();
  H.resize();
  dalpha1.resize();
  to_be_relaxed.resize();
  Newton_iterations.resize();

  /*--- Declare some constant parameters associated to the initial state ---*/
  const auto eps_R = eps_over_R*R;

  /*--- Initialize some fields to define the bubble with a loop over all cells ---*/
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

                              alpha1[cell] = std::min(std::max(alpha_residual, w),
                                                      static_cast<Number>(1.0) - alpha_residual);
                            }
                        );

  /*--- Compute the geometrical quantities ---*/
  update_geometry();

  /*--- Loop over a cell to complete the remaining variables ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              // Recompute geometric locations to set partial masses
                              const auto center = cell.center();
                              const auto x      = static_cast<Number>(center[0]);
                              const auto y      = static_cast<Number>(center[1]);
                              const auto r      = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                              // Set mass large-scale phase 1
                              Number p1;
                              if(r >= R + eps_R) {
                                p1 = EOS_phase1.get_p0();
                              }
                              else {
                                p1 = EOS_phase2.get_p0();
                                if(r >= R && r < R + eps_R && !std::isnan(H[cell][0])) {
                                  p1 += sigma*H[cell][0];
                                }
                                else {
                                  p1 += sigma/R;
                                }
                              }
                              const auto rho1 = EOS_phase1.rho_value(p1);

                              conserved_variables[cell][M1_INDEX] = alpha1[cell]*rho1;

                              // Set mass phase 2
                              const auto p2   = EOS_phase2.get_p0();
                              const auto rho2 = EOS_phase2.rho_value(p2);

                              conserved_variables[cell][M2_INDEX] = (static_cast<Number>(1.0) - alpha1[cell])*rho2;

                              // Set conserved variable associated to large-scale volume fraction
                              const auto rho = conserved_variables[cell][M1_INDEX]
                                             + conserved_variables[cell][M2_INDEX];

                              conserved_variables[cell][RHO_ALPHA1_INDEX] = rho*alpha1[cell];

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
}

// Auxiliary routine to impose the boundary conditions
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::apply_bcs(const Number U0,
                                         const Number V0,
                                         const Number alpha_residual) {
  const samurai::DirectionVector<dim> left = {-1, 0};
  samurai::make_bc<Default>(conserved_variables,
                            Inlet(conserved_variables, U0, V0, alpha_residual))->on(left);

  const samurai::DirectionVector<dim> right = {1, 0};
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables,
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0),
                                        static_cast<Number>(0.0))->on(right);
}

//////////////////////////////////////////////////////////////
/*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/
/////////////////////////////////////////////////////////////

// Auxiliary routine to compute normals and curvature
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::update_geometry() {
  samurai::update_ghost_mr(alpha1);

  grad_alpha1 = gradient(alpha1);

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

  vel.resize();

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
                              const auto m1_loc = conserved_variables[cell][M1_INDEX];
                              const auto m2_loc = conserved_variables[cell][M2_INDEX];

                              /*--- Compute the velocity along both horizontal and vertical direction ---*/
                              const auto rho_loc     = m1_loc + m2_loc;
                              const auto inv_rho_loc = static_cast<Number>(1.0)/rho_loc;
                              for(std::size_t d = 0; d < dim; ++d) {
                                vel[cell][d] = conserved_variables[cell][RHO_U_INDEX + d]*inv_rho_loc;
                              }

                              /*--- Compute frozen speed of sound ---*/
                              const auto alpha1_loc       = alpha1[cell];
                              const auto rho1_loc         = m1_loc/alpha1_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
                              const auto alpha2_loc       = static_cast<Number>(1.0) - alpha1_loc;
                              const auto rho2_loc         = m2_loc/alpha2_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
                              const auto rhoc_squared_loc = m1_loc*EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)
                                                          + m2_loc*EOS_phase2.c_value(rho2_loc)*EOS_phase2.c_value(rho2_loc);
                              const auto c_loc            = std::sqrt(rhoc_squared_loc*inv_rho_loc);

                              /*--- Add term due to surface tension ---*/
                              auto mod2_grad_alpha1_loc = static_cast<Number>(0.0);
                              for(std::size_t d = 0; d < dim; ++d) {
                                mod2_grad_alpha1_loc += grad_alpha1[cell][d]*grad_alpha1[cell][d];
                              }
                              const auto mod_grad_alpha1_loc = std::sqrt(mod2_grad_alpha1_loc);

                              const auto r = sigma*mod_grad_alpha1_loc/(rho_loc*c_loc*c_loc);

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
  #ifdef VERBOSE
    save("_before_mesh_adaptation", conserved_variables, alpha1);
  #endif

  /*auto prediction_fn = [&](auto& new_field, const auto& old_field) {
    return make_field_operator_function<TwoScaleCapillarity_prediction_op>(new_field, old_field);
  };*/

  auto MRadaptation = samurai::make_MRAdapt(/*prediction_fn,*/ conserved_variables);
  auto mra_config   = samurai::mra_config();
  mra_config.epsilon(MR_param);
  mra_config.regularity(MR_regularity);
  MRadaptation(mra_config);

  /*--- Sanity check after mesh adaptation ---*/
  alpha1.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
                              alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                             (conserved_variables[cell][M1_INDEX] +
                                              conserved_variables[cell][M2_INDEX]);
                            }
                        );
  #ifdef VERBOSE
    check_data(1);
    save("_after_mesh_adaptation", conserved_variables, alpha1);
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
    op = "after mesh adpatation";
  }

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                            {
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
                              const auto m1_loc = conserved_variables[cell][M1_INDEX];
                              if(m1_loc < static_cast<Number>(0.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative mass of phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1);
                                exit(1);
                              }
                              else if(std::isnan(m1_loc)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN mass of phase 1 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1);
                                exit(1);
                              }

                              // Sanity check for m2
                              const auto m2_loc = conserved_variables[cell][M2_INDEX];
                              if(m2_loc < static_cast<Number>(0.0)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "Negative mass of phase 2 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1);
                                exit(1);
                              }
                              else if(std::isnan(m2_loc)) {
                                std::cerr << cell << std::endl;
                                std::cerr << "NaN mass of phase 2 " + op << std::endl;
                                save("_diverged", conserved_variables, alpha1);
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
  dalpha1.fill(std::numeric_limits<Number>::infinity());
  bool global_relaxation_applied = true;

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
                                                                 H[cell][0], dalpha1[cell], alpha1[cell],
                                                                 to_be_relaxed[cell], Newton_iterations[cell], local_relaxation_applied);
                                }
                                catch(std::exception& e) {
                                  std::cerr << e.what() << std::endl;
                                  save("_diverged",
                                       conserved_variables,
                                       alpha1, dalpha1, grad_alpha1, normal, H,
                                       to_be_relaxed, Newton_iterations);
                                  exit(1);
                                }
                              }
                          );

    mpi::communicator world;
    boost::mpi::all_reduce(world, local_relaxation_applied, global_relaxation_applied, std::logical_or<bool>());

    // Newton cycle diverged
    if(Newton_iter > max_Newton_iters && global_relaxation_applied == true) {
      std::cerr << "Netwon method not converged in the post-hyperbolic relaxation" << std::endl;
      save("_diverged",
           conserved_variables,
           alpha1, dalpha1, grad_alpha1, normal, H,
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
template<typename State>
void TwoScaleCapillarity<dim>::perform_Newton_step_relaxation(State local_conserved_variables,
                                                              const Number H_loc,
                                                              Number& dalpha1_loc,
                                                              Number& alpha1_loc,
                                                              std::size_t& to_be_relaxed_loc,
                                                              std::size_t& Newton_iterations_loc,
                                                              bool& local_relaxation_applied) {
  to_be_relaxed_loc = 0;

  if(!std::isnan(H_loc)) {
    /*--- Pre-fetch some variables used multiple times in order to exploit possible vectorization ---*/
    const auto m1_loc = local_conserved_variables(M1_INDEX);
    const auto m2_loc = local_conserved_variables(M2_INDEX);

    /*--- Update auxiliary values affected by the nonlinear function for which we seek a zero ---*/
    const auto rho1_loc = m1_loc/alpha1_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p1_loc   = EOS_phase1.pres_value(rho1_loc);

    const auto alpha2_loc = static_cast<Number>(1.0) - alpha1_loc;
    const auto rho2_loc   = m2_loc/alpha2_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p2_loc     = EOS_phase2.pres_value(rho2_loc);

    /*--- Compute the nonlinear function for which we seek the zero (basically the Laplace law) ---*/
    const auto F = p1_loc - p2_loc - sigma*H_loc;

    /*--- Perform the relaxation only where really needed ---*/
    if(std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase1.get_p0(), sigma*std::abs(H_loc)) &&
       std::abs(dalpha1_loc) > atol_Newton) {
      to_be_relaxed_loc = 1;
      Newton_iterations_loc++;
      local_relaxation_applied = true;

      // Compute the derivative w.r.t large-scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
      const auto dF_dalpha1 = -m1_loc/(alpha1_loc*alpha1_loc)*
                               EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)
                              -m2_loc/(alpha2_loc*alpha2_loc)*
                               EOS_phase2.c_value(rho2_loc)*EOS_phase2.c_value(rho2_loc);

      // Compute the large-scale volume fraction update
      dalpha1_loc = -F/dF_dalpha1;
      if(dalpha1_loc > static_cast<Number>(0.0)) {
        dalpha1_loc = std::min(dalpha1_loc, lambda*alpha2_loc);
      }
      else if(dalpha1_loc < static_cast<Number>(0.0)) {
        dalpha1_loc = std::max(dalpha1_loc, -lambda*alpha1_loc);
      }

      if(alpha1_loc + dalpha1_loc < static_cast<Number>(0.0) ||
         alpha1_loc + dalpha1_loc > static_cast<Number>(1.0)) {
        throw std::runtime_error("Bounds exceeding value for large-scale volume fraction inside Newton step ");
      }
      else {
        alpha1_loc += dalpha1_loc;
      }
    }

    /*--- Update the vector of conserved variables (probably not the optimal choice since I need this update only at the end of the Newton loop,
          but the most coherent one thinking about the transfer of mass) ---*/
    local_conserved_variables(RHO_ALPHA1_INDEX) = (m1_loc + m2_loc)*alpha1_loc;
  }
}

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

// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::run(const unsigned nfiles) {
  /*--- Default output arguemnts ---*/
  path = fs::current_path();
  filename = "liquid_column_no_mass_transfer";
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

  const auto dt_save = Tf/static_cast<Number>(nfiles);

  /*--- Auxiliary variables to save updated fields ---*/
  #ifdef ORDER_2
    auto conserved_variables_old = samurai::make_vector_field<Number, Field::n_comp>("conserved_old", mesh);
    auto conserved_variables_tmp = samurai::make_vector_field<Number, Field::n_comp>("conserved_tmp", mesh);
  #endif
  auto conserved_variables_np1 = samurai::make_vector_field<Number, Field::n_comp>("conserved_np1", mesh);

  /*--- Create the flux variable ---*/
  #ifdef RUSANOV_FLUX
    #ifdef ORDER_2
      auto numerical_flux_hyp = Rusanov_flux.make_flux(H);
    #else
      auto numerical_flux_hyp = Rusanov_flux.make_flux();
    #endif
  #elifdef GODUNOV_FLUX
    #ifdef ORDER_2
      auto numerical_flux_hyp = Godunov_flux.make_flux(H);
    #else
      auto numerical_flux_hyp = Godunov_flux.make_flux();
    #endif
  #elifdef HLLC_FLUX
    #ifdef ORDER_2
      auto numerical_flux_hyp = HLLC_flux.make_flux(H);
    #else
      auto numerical_flux_hyp = HLLC_flux.make_flux();
    #endif
  #endif
  auto numerical_flux_st = SurfaceTension_flux.make_flux_capillarity(grad_alpha1);

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? fmt::format("_ite_0") : "";
  save(suffix_init, conserved_variables,
                    alpha1, grad_alpha1, normal, H);

  /*--- Set mesh size ---*/
  const auto dx = static_cast<Number>(mesh.cell_length(mesh.max_level()));
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  const auto n_elements_per_subdomain = mesh[mesh_id_t::cells].nb_cells();
  unsigned n_elements;
  MPI_Allreduce(&n_elements_per_subdomain, &n_elements, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "Number of elements = " <<  n_elements << std::endl;
    std::cout << std::endl;
  }

  /*--- Start the loop ---*/
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  auto t            = static_cast<Number>(t0);
  while(t != Tf) {
    // Apply mesh adaptation
    perform_mesh_adaptation();

    // Compute the time step
    normal.resize();
    H.resize();
    grad_alpha1.resize();
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
      samurai::update_ghost_mr(H);
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
      save("_diverged",
           conserved_variables, alpha1, grad_alpha1, normal, H);
      exit(1);
    }

    // Check if spurious negative values arise and recompute geometrical quantities
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                              {
                                alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                               (conserved_variables[cell][M1_INDEX] +
                                                conserved_variables[cell][M2_INDEX]);
                              }
                          );
    #ifdef VERBOSE
      check_data();
    #endif
    update_geometry();

    // Capillarity contribution
    samurai::update_ghost_mr(conserved_variables);
    samurai::update_ghost_mr(grad_alpha1);
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
      // Apply relaxation if desired, which will modify alpha1 and, consequently, for what
      // concerns next time step, rho_alpha1 (as well as grad_alpha1)
      dalpha1.resize();
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
        samurai::update_ghost_mr(H);
      #endif
      try {
        auto flux_hyp = numerical_flux_hyp(conserved_variables);
        conserved_variables_tmp = conserved_variables - dt*flux_hyp;
        std::swap(conserved_variables.array(), conserved_variables_tmp.array());
      }
      catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
        save("_diverged",
             conserved_variables, alpha1, grad_alpha1, normal, H);
        exit(1);
      }

      // Check if spurious negative values arise and recompute geometrical quantities
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                                {
                                  alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                                 (conserved_variables[cell][M1_INDEX] +
                                                  conserved_variables[cell][M2_INDEX]);
                                }
                            );
      #ifdef VERBOSE
        check_data();
      #endif
      update_geometry();

      // Capillarity contribution
      samurai::update_ghost_mr(conserved_variables);
      samurai::update_ghost_mr(grad_alpha1);
      flux_st = numerical_flux_st(conserved_variables);
      conserved_variables_tmp = conserved_variables - dt*flux_st;
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());

      // Apply the relaxation
      if(apply_relax) {
        // Apply relaxation if desired, which will modify alpha1 and, consequently, for what
        // concerns next time step, rho_alpha1
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
                                    alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                                   (conserved_variables[cell][M1_INDEX] +
                                                    conserved_variables[cell][M2_INDEX]);
                                  }
                              );
        update_geometry();
      #endif
    #endif

    // Save the results
    if(t >= static_cast<Number>(nsave + 1)*dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(suffix, conserved_variables,
                   alpha1, grad_alpha1, normal, H,
                   Newton_iterations);
    }
  }
}
