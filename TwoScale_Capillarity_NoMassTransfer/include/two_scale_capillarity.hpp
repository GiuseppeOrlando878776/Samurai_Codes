// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>

#include <filesystem>
namespace fs = std::filesystem;

/*--- Add header file for the multiresolution ---*/
#include <samurai/mr/adapt.hpp>

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

  TwoScaleCapillarity() = default; /*--- Default constructor. This will do nothing
                                         and basically will never be used ---*/

  TwoScaleCapillarity(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                      const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                      const Simulation_Paramaters& sim_param,
                      const EOS_Parameters& eos_param); /*--- Class constrcutor with the arguments related
                                                              to the grid, to the physics and to the relaxation. ---*/

  void run(); /*--- Function which actually executes the temporal loop ---*/

  template<class... Variables>
  void save(const fs::path& path,
            const std::string& suffix,
            const Variables&... fields); /*--- Routine to save the results ---*/

private:
  /*--- Now we declare some relevant variables ---*/
  const samurai::Box<double, dim> box;

  samurai::MRMesh<Config> mesh; /*--- Variable to store the mesh ---*/

  using Field        = samurai::Field<decltype(mesh), double, EquationData::NVARS, false>;
  using Field_Scalar = samurai::Field<decltype(mesh), typename Field::value_type, 1, false>;
  using Field_Vect   = samurai::Field<decltype(mesh), typename Field::value_type, dim, false>;

  bool apply_relax; /*--- Choose whether to apply or not the relaxation ---*/

  double Tf;  /*--- Final time of the simulation ---*/
  double cfl; /*--- Courant number of the simulation so as to compute the time step ---*/

  std::size_t nfiles; /*--- Number of files desired for output ---*/

  Field conserved_variables; /*--- The variable which stores the conserved variables,
                                   namely the varialbes for which we solve a PDE system ---*/

  /*--- Now we declare a bunch of fields which depend from the state, but it is useful
        to have it so as to avoid recomputation ---*/
  Field_Scalar alpha1,
               H,
               dalpha1;

  Field_Vect normal,
             grad_alpha1;

  samurai::Field<decltype(mesh), std::size_t, 1, false> to_be_relaxed;
  samurai::Field<decltype(mesh), std::size_t, 1, false> Newton_iterations;

  using gradient_type = decltype(samurai::make_gradient_order2<decltype(alpha1)>());
  gradient_type gradient;

  using divergence_type = decltype(samurai::make_divergence_order2<decltype(normal)>());
  divergence_type divergence;

  const double sigma; /*--- Surface tension coefficient ---*/

  const double mod_grad_alpha1_min; /*--- Minimum threshold for which not computing anymore the unit normal ---*/

  const double      lambda;           /*--- Parameter for bound preserving strategy ---*/
  const double      atol_Newton;      /*--- Absolute tolerance Newton method relaxation ---*/
  const double      rtol_Newton;      /*--- Relative tolerance Newton method relaxation ---*/
  const std::size_t max_Newton_iters; /*--- Maximum number of Newton iterations ---*/

  LinearizedBarotropicEOS<typename Field::value_type> EOS_phase1,
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

  std::string filename; /*--- Auxiliary variable to store the name of output ---*/

  double       MR_param;      /*--- Multiresolution parameter ---*/
  unsigned int MR_regularity; /*--- Multiresolution regularity ---*/

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void update_geometry(); /*--- Auxiliary routine to compute normals and curvature ---*/

  void init_variables(const double R, const double eps_over_R,
                      const double alpha_residual); /*--- Routine to initialize the variables
                                                          (both conserved and auxiliary, this is problem dependent) ---*/

  double get_max_lambda(); /*--- Compute the estimate of the maximum eigenvalue ---*/

  void check_data(unsigned int flag = 0); /*--- Auxiliary routine to check if small negative values are present ---*/

  void perform_mesh_adaptation(); /*--- Perform the mesh adaptation ---*/

  void apply_relaxation(); /*--- Apply the relaxation ---*/

  template<typename State>
  void perform_Newton_step_relaxation(std::unique_ptr<State> local_conserved_variables,
                                      const typename Field::value_type H_loc,
                                      typename Field::value_type& dalpha1_loc,
                                      typename Field::value_type& alpha1_loc,
                                      std::size_t& to_be_relaxed_loc,
                                      std::size_t& Newton_iterations_loc,
                                      bool& local_relaxation_applied);
};

/*---- START WITH THE IMPLEMENTATION OF THE CONSTRUCTOR ---*/

// Implement class constructor
//
template<std::size_t dim>
TwoScaleCapillarity<dim>::TwoScaleCapillarity(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                                              const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                                              const Simulation_Paramaters& sim_param,
                                              const EOS_Parameters& eos_param):
  box(min_corner, max_corner), mesh(box, sim_param.min_level, sim_param.max_level, {{false, true}}),
  apply_relax(sim_param.apply_relaxation), Tf(sim_param.Tf),
  cfl(sim_param.Courant), nfiles(sim_param.nfiles),
  gradient(samurai::make_gradient_order2<decltype(alpha1)>()),
  divergence(samurai::make_divergence_order2<decltype(normal)>()),
  sigma(sim_param.sigma), mod_grad_alpha1_min(sim_param.mod_grad_alpha1_min),
  lambda(sim_param.lambda), atol_Newton(sim_param.atol_Newton),
  rtol_Newton(sim_param.rtol_Newton), max_Newton_iters(sim_param.max_Newton_iters),
  EOS_phase1(eos_param.p0_phase1, eos_param.rho0_phase1, eos_param.c0_phase1),
  EOS_phase2(eos_param.p0_phase2, eos_param.rho0_phase2, eos_param.c0_phase2),
  #ifdef RUSANOV_FLUX
    Rusanov_flux(EOS_phase1, EOS_phase2, sigma, mod_grad_alpha1_min,
                 lambda, atol_Newton, rtol_Newton, max_Newton_iters),
  #elifdef GODUNOV_FLUX
    Godunov_flux(EOS_phase1, EOS_phase2, sigma, mod_grad_alpha1_min,
                 lambda, atol_Newton, rtol_Newton, max_Newton_iters
                 sim_param.atol_Newton_p_star, sim_param.rtol_Newton_p_star),
  #elifdef HLLC_FLUX
    HLLC_flux(EOS_phase1, EOS_phase2, sigma, mod_grad_alpha1_min,
              lambda, atol_Newton, rtol_Newton, max_Newton_iters),
  #endif
  SurfaceTension_flux(EOS_phase1, EOS_phase2, sigma, mod_grad_alpha1_min,
                      sim_param.lambda, sim_param.atol_Newton, sim_param.rtol_Newton,
                      max_Newton_iters),
  MR_param(sim_param.MR_param), MR_regularity(sim_param.MR_regularity)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
      std::cout << "Initializing variables " << std::endl;
      std::cout << std::endl;
    }
    init_variables(sim_param.R, sim_param.eps_over_R, sim_param.alpha_residual);
    to_be_relaxed     = samurai::make_field<std::size_t, 1>("to_be_relaxed", mesh);
    Newton_iterations = samurai::make_field<std::size_t, 1>("Newton_iterations", mesh);
  }

// Auxiliary routine to compute normals and curvature
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::update_geometry() {
  samurai::update_ghost_mr(alpha1);

  grad_alpha1 = gradient(alpha1);

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto mod_grad_alpha1 = std::sqrt(xt::sum(grad_alpha1[cell]*grad_alpha1[cell])());

                           if(mod_grad_alpha1 > mod_grad_alpha1_min) {
                             normal[cell] = grad_alpha1[cell]/mod_grad_alpha1;
                           }
                           else {
                             for(std::size_t d = 0; d < dim; ++d) {
                               normal[cell][d] = nan("");
                             }
                           }
                         });
  samurai::update_ghost_mr(normal);
  H = -divergence(normal);
}

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::init_variables(const double R, const double eps_over_R, const double alpha_residual) {
  /*--- Create conserved and auxiliary fields ---*/
  conserved_variables = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved", mesh);

  alpha1      = samurai::make_field<typename Field::value_type, 1>("alpha1", mesh);
  grad_alpha1 = samurai::make_field<typename Field::value_type, dim>("grad_alpha1", mesh);
  normal      = samurai::make_field<typename Field::value_type, dim>("normal", mesh);
  H           = samurai::make_field<typename Field::value_type, 1>("H", mesh);

  dalpha1     = samurai::make_field<typename Field::value_type, 1>("dalpha1", mesh);

  /*--- Declare some constant parameters associated to the grid and to the
        initial state ---*/
  const double x0    = 1.0;
  const double y0    = 1.0;
  const double eps_R = eps_over_R*R;

  const double U_0   = 6.66;
  const double U_1   = 0.0;
  const double V     = 0.0;

  /*--- Initialize some fields to define the bubble with a loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Set large-scale volume fraction
                           const auto center = cell.center();
                           const double x    = center[0];
                           const double y    = center[1];

                           const double r = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                           const double w = (r >= R && r < R + eps_R) ?
                                            std::max(std::exp(2.0*(r - R)*(r - R)/(eps_R*eps_R)*((r - R)*(r - R)/(eps_R*eps_R) - 3.0)/
                                                              (((r - R)*(r - R)/(eps_R*eps_R) - 1.0)*((r - R)*(r - R)/(eps_R*eps_R) - 1.0))), 0.0) :
                                            ((r < R) ? 1.0 : 0.0);

                           alpha1[cell] = std::min(std::max(alpha_residual, w), 1.0 - alpha_residual);
                         });

  /*--- Compute the geometrical quantities ---*/
  update_geometry();

  /*--- Loop over a cell to complete the remaining variables ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Recompute geometric locations to set partial masses
                           const auto center = cell.center();
                           const double x    = center[0];
                           const double y    = center[1];

                           const double r = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                           // Set mass large-scale phase 1
                           typename Field::value_type p1;
                           if(r >= R + eps_R) {
                             p1 = EOS_phase1.get_p0();
                           }
                           else {
                             p1 = EOS_phase2.get_p0();
                             if(r >= R && r < R + eps_R && !std::isnan(H[cell])) {
                               p1 += sigma*H[cell];
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

                           conserved_variables[cell][M2_INDEX] = (1.0 - alpha1[cell])*rho2;

                           // Set conserved variable associated to large-scale volume fraction
                           const auto rho = conserved_variables[cell][M1_INDEX]
                                          + conserved_variables[cell][M2_INDEX];

                           conserved_variables[cell][RHO_ALPHA1_INDEX] = rho*alpha1[cell];

                           // Set momentum
                           conserved_variables[cell][RHO_U_INDEX]     = conserved_variables[cell][M1_INDEX]*U_1
                                                                      + conserved_variables[cell][M2_INDEX]*U_0;
                           conserved_variables[cell][RHO_U_INDEX + 1] = rho*V;
                         });

  /*--- Apply bcs ---*/
  const samurai::DirectionVector<dim> left  = {-1, 0};
  const samurai::DirectionVector<dim> right = {1, 0};
  samurai::make_bc<Default>(conserved_variables,
                            Inlet(conserved_variables, U_0, 0.0, alpha_residual))->on(left);
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables, 0.0, 0.0, 0.0, 0.0, 0.0)->on(right);
}

/*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double TwoScaleCapillarity<dim>::get_max_lambda() {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           #ifndef RELAX_RECONSTRUCTION
                             alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                            (conserved_variables[cell][M1_INDEX] + conserved_variables[cell][M2_INDEX]);
                           #endif

                           /*--- Compute the velocity along both horizontal and vertical direction ---*/
                           const auto rho   = conserved_variables[cell][M1_INDEX]
                                            + conserved_variables[cell][M2_INDEX];
                           const auto vel_x = conserved_variables[cell][RHO_U_INDEX]/rho;
                           const auto vel_y = conserved_variables[cell][RHO_U_INDEX + 1]/rho;

                           /*--- Compute frozen speed of sound ---*/
                           const auto rho1      = conserved_variables[cell][M1_INDEX]/alpha1[cell]; /*--- TODO: Add a check in case of zero volume fraction ---*/
                           const auto rho2      = conserved_variables[cell][M2_INDEX]/(1.0 - alpha1[cell]); /*--- TODO: Add a check in case of zero volume fraction ---*/
                           const auto c_squared = conserved_variables[cell][M1_INDEX]*EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)
                                                + conserved_variables[cell][M2_INDEX]*EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2);
                           const auto c         = std::sqrt(c_squared/rho);

                           /*--- Add term due to surface tension ---*/
                           const double r = sigma*std::sqrt(xt::sum(grad_alpha1[cell]*grad_alpha1[cell])())/(rho*c*c);

                           /*--- Update eigenvalue estimate ---*/
                           res = std::max(std::max(std::abs(vel_x) + c*(1.0 + 0.125*r),
                                                   std::abs(vel_y) + c*(1.0 + 0.125*r)),
                                          res);
                         });

  return res;
}

// Perform the mesh adaptation strategy.
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::perform_mesh_adaptation() {
  #ifdef VERBOSE
    save(fs::current_path(), "_before_mesh_adaptation", conserved_variables, alpha1);
  #endif

  samurai::update_ghost_mr(grad_alpha1);
  auto MRadaptation = samurai::make_MRAdapt(grad_alpha1);
  MRadaptation(MR_param, MR_regularity, conserved_variables);

  /*--- Sanity check after mesh adaptation ---*/
  alpha1.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                            alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                           (conserved_variables[cell][M1_INDEX] + conserved_variables[cell][M2_INDEX]);
                         });
  #ifdef VERBOSE
    check_data(1);
    save(fs::current_path(), "_after_mesh_adaptation", conserved_variables, alpha1);
  #endif
}

// Auxiliary routine to check if negative values arise
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::check_data(unsigned int flag) {
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
                           if(alpha1[cell] < 0.0) {
                             std::cerr << cell << std::endl;
                             std::cerr << "Negative volume fraction of phase 1 " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                           else if(alpha1[cell] > 1.0) {
                             std::cerr << cell << std::endl;
                             std::cerr << "Exceeding volume fraction of phase 1 " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                           else if(std::isnan(alpha1[cell])) {
                             std::cerr << cell << std::endl;
                             std::cerr << "NaN volume fraction of phase 1 " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }

                           // Sanity check for m1
                           if(conserved_variables[cell][M1_INDEX] < 0.0) {
                             std::cerr << cell << std::endl;
                             std::cerr << "Negative mass of phase 1 " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                           else if(std::isnan(conserved_variables[cell][M1_INDEX])) {
                             std::cerr << cell << std::endl;
                             std::cerr << "NaN mass of phase 1 " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }

                           // Sanity check for m2
                           if(conserved_variables[cell][M2_INDEX] < 0.0) {
                             std::cerr << cell << std::endl;
                             std::cerr << "Negative mass of phase 2 " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                           else if(std::isnan(conserved_variables[cell][M2_INDEX])) {
                             std::cerr << cell << std::endl;
                             std::cerr << "NaN mass of phase 2 " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                         });
}

// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::apply_relaxation() {
  /*--- Initialize the variables ---*/
  samurai::times::timers.start("apply_relaxation");

  std::size_t Newton_iter = 0;
  to_be_relaxed.fill(0);
  Newton_iterations.fill(0);
  dalpha1.fill(std::numeric_limits<typename Field::value_type>::infinity());
  bool relaxation_applied = true;

  samurai::times::timers.stop("apply_relaxation");

  /*--- Loop of Newton method. Conceptually, a loop over cells followed by a Newton loop
        over each cell would (could?) be more logic, but this would lead to issues to call 'update_geometry' ---*/
  while(relaxation_applied == true) {
    samurai::times::timers.start("apply_relaxation");

    relaxation_applied = false;
    Newton_iter++;

    // Loop over all cells.
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             try {
                               perform_Newton_step_relaxation(std::make_unique<decltype(conserved_variables[cell])>(conserved_variables[cell]),
                                                              H[cell], dalpha1[cell], alpha1[cell],
                                                              to_be_relaxed[cell], Newton_iterations[cell], relaxation_applied);
                             }
                             catch(std::exception& e) {
                               std::cerr << e.what() << std::endl;
                               save(fs::current_path(), "_diverged",
                                    conserved_variables, alpha1, dalpha1, grad_alpha1, normal, H,
                                    to_be_relaxed, Newton_iterations);
                               exit(1);
                             }
                           });

    // Newton cycle diverged
    if(Newton_iter > max_Newton_iters && relaxation_applied == true) {
      std::cerr << "Netwon method not converged in the post-hyperbolic relaxation" << std::endl;
      save(fs::current_path(), "_diverged",
           conserved_variables, alpha1, dalpha1, grad_alpha1, normal, H,
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
void TwoScaleCapillarity<dim>::perform_Newton_step_relaxation(std::unique_ptr<State> local_conserved_variables,
                                                              const typename Field::value_type H_loc,
                                                              typename Field::value_type& dalpha1_loc,
                                                              typename Field::value_type& alpha1_loc,
                                                              std::size_t& to_be_relaxed_loc,
                                                              std::size_t& Newton_iterations_loc,
                                                              bool& local_relaxation_applied) {
  to_be_relaxed_loc = 0;
  if(!std::isnan(H_loc)) {
    /*--- Update auxiliary values affected by the nonlinear function for which we seek a zero ---*/
    const auto rho1_loc = (*local_conserved_variables)(M1_INDEX)/alpha1_loc; /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p1_loc   = EOS_phase1.pres_value(rho1_loc);

    const auto rho2_loc = (*local_conserved_variables)(M2_INDEX)/(1.0 - alpha1_loc); /*--- TODO: Add a check in case of zero volume fraction ---*/
    const auto p2_loc   = EOS_phase2.pres_value(rho2_loc);

    /*--- Compute the nonlinear function for which we seek the zero (basically the Laplace law) ---*/
    const auto F = p1_loc - p2_loc - sigma*H_loc;

    /*--- Perform the relaxation only where really needed ---*/
    if(std::abs(F) > atol_Newton + rtol_Newton*std::min(EOS_phase1.get_p0(), sigma*std::abs(H_loc)) &&
       std::abs(dalpha1_loc) > atol_Newton) {
      to_be_relaxed_loc = 1;
      Newton_iterations_loc++;
      local_relaxation_applied = true;

      // Compute the derivative w.r.t large-scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
      const auto dF_dalpha1 = -(*local_conserved_variables)(M1_INDEX)/(alpha1_loc*alpha1_loc)*
                               EOS_phase1.c_value(rho1_loc)*EOS_phase1.c_value(rho1_loc)
                              -(*local_conserved_variables)(M2_INDEX)/((1.0 - alpha1_loc)*(1.0 - alpha1_loc))*
                               EOS_phase2.c_value(rho2_loc)*EOS_phase2.c_value(rho2_loc);

      // Compute the large-scale volume fraction update
      dalpha1_loc = -F/dF_dalpha1;
      if(dalpha1_loc > 0.0) {
        dalpha1_loc = std::min(dalpha1_loc, lambda*(1.0 - alpha1_loc));
      }
      else if(dalpha1_loc < 0.0) {
        dalpha1_loc = std::max(dalpha1_loc, -lambda*alpha1_loc);
      }

      if(alpha1_loc + dalpha1_loc < 0.0 || alpha1_loc + dalpha1_loc > 1.0) {
        throw std::runtime_error("Bounds exceeding value for large-scale volume fraction inside Newton step ");
      }
      else {
        alpha1_loc += dalpha1_loc;
      }
    }

    /*--- Update the vector of conserved variables (probably not the optimal choice since I need this update only at the end of the Newton loop,
          but the most coherent one thinking about the transfer of mass) ---*/
    const auto rho_loc = (*local_conserved_variables)(M1_INDEX)
                       + (*local_conserved_variables)(M2_INDEX);
    (*local_conserved_variables)(RHO_ALPHA1_INDEX) = rho_loc*alpha1_loc;
  }
}

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void TwoScaleCapillarity<dim>::save(const fs::path& path,
                                    const std::string& suffix,
                                    const Variables&... fields) {
  auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

  if(!fs::exists(path)) {
    fs::create_directory(path);
  }

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           level_[cell] = cell.level;
                         });

  samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, fields..., level_);
}

// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::run() {
  /*--- Default output arguemnts ---*/
  fs::path path = fs::current_path();
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

  const double dt_save = Tf/static_cast<double>(nfiles);

  /*--- Auxiliary variables to save updated fields ---*/
  #ifdef ORDER_2
    auto conserved_variables_old = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved_old", mesh);
    auto conserved_variables_tmp = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved_tmp", mesh);
  #endif
  auto conserved_variables_np1 = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved_np1", mesh);

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
  save(path, suffix_init, conserved_variables, alpha1, grad_alpha1, normal, H);

  /*--- Set mesh size ---*/
  const double dx = mesh.cell_length(mesh.max_level());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  const auto n_elements_per_subdomain = mesh[mesh_id_t::cells].nb_cells();
  unsigned int n_elements;
  MPI_Allreduce(&n_elements_per_subdomain, &n_elements, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "Number of elements = " <<  n_elements << std::endl;
    std::cout << std::endl;
  }

  /*--- Start the loop ---*/
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  double t          = 0.0;
  double dt         = std::min(Tf - t, cfl*dx/get_max_lambda());
  while(t != Tf) {
    t += dt;
    if(t > Tf) {
      dt += Tf - t;
      t = Tf;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
      std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;
    }

    // Apply mesh adaptation
    perform_mesh_adaptation();

    // Save current state in case of order 2
    #ifdef ORDER_2
      conserved_variables_old.resize();
      conserved_variables_old = conserved_variables;
    #endif

    // Apply the numerical scheme without relaxation
    // Convective operator
    samurai::update_ghost_mr(conserved_variables);
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
      save(fs::current_path(), "_diverged",
           conserved_variables, alpha1, grad_alpha1, normal, H);
      exit(1);
    }

    // Check if spurious negative values arise and recompute geometrical quantities
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                              alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                             (conserved_variables[cell][M1_INDEX] + conserved_variables[cell][M2_INDEX]);
                           });
    #ifdef VERBOSE
      check_data();
    #endif
    normal.resize();
    H.resize();
    grad_alpha1.resize();
    update_geometry();

    // Capillarity contribution
    samurai::update_ghost_mr(conserved_variables);
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
      try {
        auto flux_hyp = numerical_flux_hyp(conserved_variables);
        conserved_variables_tmp = conserved_variables - dt*flux_hyp;
        std::swap(conserved_variables.array(), conserved_variables_tmp.array());
      }
      catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
        save(fs::current_path(), "_diverged",
             conserved_variables, alpha1, grad_alpha1, normal, H);
        exit(1);
      }

      // Check if spurious negative values arise and recompute geometrical quantities
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                                alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                               (conserved_variables[cell][M1_INDEX] + conserved_variables[cell][M2_INDEX]);
                             });
      #ifdef VERBOSE
        check_data();
      #endif
      update_geometry();

      // Capillarity contribution
      samurai::update_ghost_mr(conserved_variables);
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
      conserved_variables_np1 = 0.5*(conserved_variables_old + conserved_variables);
      std::swap(conserved_variables.array(), conserved_variables_np1.array());

      // Recompute volume fraction gradient and curvature for the next time step
      #ifdef RELAX_RECONSTRUCTION
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                 alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                                (conserved_variables[cell][M1_INDEX] + conserved_variables[cell][M2_INDEX]);
                               });
        update_geometry();
      #endif
    #endif

    // Compute updated time step
    dt = std::min(Tf - t, cfl*dx/get_max_lambda());

    // Save the results
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, suffix, conserved_variables, alpha1, grad_alpha1, normal, H, Newton_iterations);
    }
  }
}
