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

/*--- Include the headers with the numerical fluxes ---*/
//#define RUSANOV_FLUX
#define GODUNOV_FLUX

#ifdef RUSANOV_FLUX
  #include "Rusanov_flux.hpp"
#elifdef GODUNOV_FLUX
  #include "Exact_Godunov_flux.hpp"
#endif
#include "SurfaceTension_flux.hpp"

// Specify the use of this namespace where we just store the indices
// and, in this case, some parameters related to EOS
using namespace EquationData;

/*--- Define preprocessor to check whether to control data or not ---*/
#define VERBOSE

/** This is the class for the simulation for the static bubble
 */
template<std::size_t dim>
class StaticBubble {
public:
  using Config = samurai::MRConfig<dim, 2, 2, 0>;

  StaticBubble() = default; /*--- Default constructor. This will do nothing
                                  and basically will never be used ---*/

  StaticBubble(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
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

  samurai::MRMesh<Config> mesh; // Variable to store the mesh

  using Field        = samurai::Field<decltype(mesh), double, EquationData::NVARS, false>;
  using Field_Scalar = samurai::Field<decltype(mesh), typename Field::value_type, 1, false>;
  using Field_Vect   = samurai::Field<decltype(mesh), typename Field::value_type, dim, false>;

  bool apply_relax; /*--- Choose whether to apply or not the relaxation ---*/

  double Tf;  /*--- Final time of the simulation ---*/
  double cfl; /*--- Courant number of the simulation so as to compute the time step ---*/

  std::size_t nfiles; /*--- Number of files desired for output ---*/

  double MR_param;
  unsigned int MR_regularity; /*--- multiresolution parameters ---*/

  Field conserved_variables; /*--- The variable which stores the conserved variables,
                                   namely the varialbes for which we solve a PDE system ---*/

  /*--- Now we declare a bunch of fields which depend from the state, but it is useful
        to have it so as to avoid recomputation ---*/
  Field_Scalar alpha1_bar,
               H_bar,
               dalpha1_bar,
               p1,
               p2,
               p_bar;

  Field_Vect normal,
             grad_alpha1_bar,
             vel;

  using gradient_type = decltype(samurai::make_gradient_order2<decltype(alpha1_bar)>());
  gradient_type gradient;

  using divergence_type = decltype(samurai::make_divergence_order2<decltype(normal)>());
  divergence_type divergence;

  const double R; /*--- Bubble radius ---*/
  const double sigma; /*--- Surface tension coefficient ---*/

  const double mod_grad_alpha1_bar_min; /*--- Minimum threshold for which not computing anymore the unit normal ---*/

  std::size_t max_Newton_iters; /*--- Maximum number of Newton iterations ---*/

  LinearizedBarotropicEOS<typename Field::value_type> EOS_phase1,
                                                      EOS_phase2; /*--- The two variables which take care of the
                                                                        barotropic EOS to compute the speed of sound ---*/
  #ifdef RUSANOV_FLUX
    samurai::RusanovFlux<Field> Rusanov_flux; /*--- Auxiliary variable to compute the flux ---*/
  #elifdef GODUNOV_FLUX
    samurai::GodunovFlux<Field> Godunov_flux; /*--- Auxiliary variable to compute the flux ---*/
  #endif
  samurai::SurfaceTensionFlux<Field> SurfaceTension_flux; /*--- Auxiliary variable to compute the contribution associated to surface tension ---*/

  std::string filename; /*--- Auxiliary variable to store the name of output ---*/

  std::ofstream pressure_data,
                max_velocity;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void update_geometry(); /*--- Auxiliary routine to compute normals and curvature ---*/

  void init_variables(const double x0, const double y0, const double eps_over_R,
                      const double alpha_residual); /*--- Routine to initialize the variables
                                                          (both conserved and auxiliary, this is problem dependent) ---*/

  double get_max_lambda(const double time); /*--- Compute the estimate of the maximum eigenvalue ---*/

  void check_data(unsigned int flag = 0); /*--- Auxiliary routine to check if spurious values are present ---*/

  void perform_mesh_adaptation(); /*--- Perform the mesh adaptation ---*/

  void apply_relaxation(); /*--- Apply the relaxation ---*/

  void execute_postprocess_pressure_jump(const double time); /*--- Execute the postprocess ---*/
};

/*---- START WITH THE IMPLEMENTATION OF THE CONSTRUCTOR ---*/

// Implement class constructor
//
template<std::size_t dim>
StaticBubble<dim>::StaticBubble(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                                const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                                const Simulation_Paramaters& sim_param,
                                const EOS_Parameters& eos_param):
  box(min_corner, max_corner), mesh(box, sim_param.min_level, sim_param.max_level, {{true, true}}),
  apply_relax(sim_param.apply_relaxation), Tf(sim_param.Tf),
  cfl(sim_param.Courant), nfiles(sim_param.nfiles),
  gradient(samurai::make_gradient_order2<decltype(alpha1_bar)>()),
  divergence(samurai::make_divergence_order2<decltype(normal)>()),
  R(sim_param.R), sigma(sim_param.sigma),
  mod_grad_alpha1_bar_min(sim_param.mod_grad_alpha1_bar_min), max_Newton_iters(sim_param.max_Newton_iters),
  EOS_phase1(eos_param.p0_phase1, eos_param.rho0_phase1, eos_param.c0_phase1),
  EOS_phase2(eos_param.p0_phase2, eos_param.rho0_phase2, eos_param.c0_phase2),
  #ifdef RUSANOV_FLUX
    Rusanov_flux(EOS_phase1, EOS_phase2, sigma, sim_param.sigma_relax, mod_grad_alpha1_bar_min,
                 sim_param.lambda, sim_param.tol_Newton, max_Newton_iters),
  #elifdef GODUNOV_FLUX
    Godunov_flux(EOS_phase1, EOS_phase2, sigma, sim_param.sigma_relax, mod_grad_alpha1_bar_min,
                 sim_param.lambda, sim_param.tol_Newton, max_Newton_iters),
  #endif
  SurfaceTension_flux(EOS_phase1, EOS_phase2, sigma, sim_param.sigma_relax, mod_grad_alpha1_bar_min,
                      sim_param.lambda, sim_param.tol_Newton, max_Newton_iters)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
      std::cout << "Initializing variables " << std::endl;
      std::cout << std::endl;
    }
    init_variables(sim_param.xL + 0.5*(sim_param.xR - sim_param.xL),
                   sim_param.yL + 0.5*(sim_param.yR - sim_param.yL),
                   sim_param.eps_over_R, sim_param.alpha_residual);
  }

// Auxiliary routine to compute normals and curvature
//
template<std::size_t dim>
void StaticBubble<dim>::update_geometry() {
  samurai::update_ghost_mr(alpha1_bar);

  grad_alpha1_bar = gradient(alpha1_bar);

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto mod_grad_alpha1_bar = std::sqrt(xt::sum(grad_alpha1_bar[cell]*grad_alpha1_bar[cell])());

                           if(mod_grad_alpha1_bar > mod_grad_alpha1_bar_min) {
                             normal[cell] = grad_alpha1_bar[cell]/mod_grad_alpha1_bar;
                           }
                           else {
                             for(std::size_t d = 0; d < dim; ++d) {
                               normal[cell][d] = nan("");
                             }
                           }
                         });
  samurai::update_ghost_mr(normal);
  H_bar = -divergence(normal);
}

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void StaticBubble<dim>::init_variables(const double x0, const double y0, const double eps_over_R,
                                       const double alpha_residual) {
  /*--- Create conserved and auxiliary fields ---*/
  conserved_variables = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved", mesh);

  alpha1_bar      = samurai::make_field<typename Field::value_type, 1>("alpha1_bar", mesh);
  grad_alpha1_bar = samurai::make_field<typename Field::value_type, dim>("grad_alpha1_bar", mesh);
  normal          = samurai::make_field<typename Field::value_type, dim>("normal", mesh);
  H_bar           = samurai::make_field<typename Field::value_type, 1>("H_bar", mesh);

  dalpha1_bar     = samurai::make_field<typename Field::value_type, 1>("dalpha1_bar", mesh);

  p1              = samurai::make_field<typename Field::value_type, 1>("p1", mesh);
  p2              = samurai::make_field<typename Field::value_type, 1>("p2", mesh);
  p_bar           = samurai::make_field<typename Field::value_type, 1>("p_bar", mesh);

  vel             = samurai::make_field<typename Field::value_type, dim>("vel", mesh);

  /*--- Declare some constant parameters associated to the grid and to the
        initial state ---*/
  const double eps_R = eps_over_R*R;

  const double U_0 = 0.0;
  const double U_1 = 0.0;
  const double V   = 0.0;

  /*--- Initialize some fields to define the bubble with a loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Set large-scale volume fraction
                           const auto center = cell.center();
                           const double x    = center[0];
                           const double y    = center[1];

                           const double r = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                           const double w = (r >= R - 0.5*eps_R && r <= R + 0.5*eps_R) ?
                                            0.5*(1.0 + std::tanh(-8.0*((r - R + 0.5*eps_R)/eps_R - 0.5))) :
                                            ((r < R - 0.5*eps_R) ? 1.0 : 0.0);

                           alpha1_bar[cell] = std::min(std::max(alpha_residual, w), 1.0 - alpha_residual);
                         });

  update_geometry();

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Set small-scale variables
                           conserved_variables[cell][ALPHA1_D_INDEX] = 0.0;
                           conserved_variables[cell][SIGMA_D_INDEX]  = 0.0;
                           conserved_variables[cell][M1_D_INDEX]     = conserved_variables[cell][ALPHA1_D_INDEX]*EOS_phase1.get_rho0();

                           // Recompute geometric locations to set partial masses
                           const auto center = cell.center();
                           const double x    = center[0];
                           const double y    = center[1];

                           const double r = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                           // Set mass large-scale phase 1
                           p1[cell] = EOS_phase2.get_p0();
                           if(r < R + eps_R) {
                             if(r >= R && r < R + eps_R && !std::isnan(H_bar[cell])) {
                               p1[cell] += sigma*H_bar[cell];
                             }
                             else {
                               p1[cell] += sigma/R;
                             }
                           }
                           const auto rho1 = EOS_phase1.rho_value(p1[cell]);

                           conserved_variables[cell][M1_INDEX] = alpha1_bar[cell]*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX])*rho1;

                           // Set mass phase 2
                           p2[cell]        = EOS_phase2.get_p0();
                           const auto rho2 = EOS_phase2.rho_value(p2[cell]);

                           conserved_variables[cell][M2_INDEX] = (1.0 - alpha1_bar[cell])*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX])*rho2;

                           // Set conserved variable associated to large-scale volume fraction
                           const auto rho = conserved_variables[cell][M1_INDEX]
                                          + conserved_variables[cell][M2_INDEX]
                                          + conserved_variables[cell][M1_D_INDEX];

                           conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = rho*alpha1_bar[cell];

                           // Set momentum
                           conserved_variables[cell][RHO_U_INDEX]     = conserved_variables[cell][M1_INDEX]*U_1 + conserved_variables[cell][M2_INDEX]*U_0;
                           conserved_variables[cell][RHO_U_INDEX + 1] = rho*V;

                           // Set mixture pressure for output
                           p_bar[cell] = alpha1_bar[cell]*p1[cell] + (1.0 - alpha1_bar[cell])*p2[cell];

                           // Set velocity for output
                           vel[cell][0] = conserved_variables[cell][RHO_U_INDEX]/rho;
                           vel[cell][1] = conserved_variables[cell][RHO_U_INDEX + 1]/rho;
                         });

  /*--- Compute the geometrical quantities ---*/
  update_geometry();

  /*--- Consider Dirichlet bcs ---*/
  //samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

/*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double StaticBubble<dim>::get_max_lambda(const double time) {
  double res = 0.0;
  typename Field::value_type max_abs_u = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           /*--- Compute the velocity along both horizontal and vertical direction ---*/
                           const auto rho = conserved_variables[cell][M1_INDEX]
                                          + conserved_variables[cell][M2_INDEX]
                                          + conserved_variables[cell][M1_D_INDEX];
                           vel[cell][0]   = conserved_variables[cell][RHO_U_INDEX]/rho;
                           vel[cell][1]   = conserved_variables[cell][RHO_U_INDEX + 1]/rho;

                           /*--- Compute frozen speed of sound ---*/
                           const auto alpha1    = alpha1_bar[cell]*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                           const auto rho1      = conserved_variables[cell][M1_INDEX]/alpha1; /*--- TODO: Add a check in case of zero volume fraction ---*/
                           const auto alpha2    = 1.0 - alpha1 - conserved_variables[cell][ALPHA1_D_INDEX];
                           const auto rho2      = conserved_variables[cell][M2_INDEX]/alpha2; /*--- TODO: Add a check in case of zero volume fraction ---*/
                           const auto c_squared = conserved_variables[cell][M1_INDEX]*EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)
                                                + conserved_variables[cell][M2_INDEX]*EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2);
                           const auto c         = std::sqrt(c_squared/rho)/(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);

                           /*--- Add term due to surface tension ---*/
                           const double r = sigma*std::sqrt(xt::sum(grad_alpha1_bar[cell]*grad_alpha1_bar[cell])())/(rho*c*c);

                           /*--- Update eigenvalue estimate ---*/
                           res = std::max(std::max(std::abs(vel[cell][0]) + c*(1.0 + 0.125*r),
                                                   std::abs(vel[cell][1]) + c*(1.0 + 0.125*r)),
                                          res);

                           /*--- Maximum absolute value of the velocity ---*/
                           max_abs_u = std::max(std::sqrt(vel[cell][0]*vel[cell][0] + vel[cell][1]*vel[cell][1]), max_abs_u);
                         });

  /*--- Save in output max velocity ---*/
  max_velocity << std::fixed << std::setprecision(12) << time << '\t' << max_abs_u << std::endl;

  return res;
}

// Perform the mesh adaptation strategy.
//
template<std::size_t dim>
void StaticBubble<dim>::perform_mesh_adaptation() {
  samurai::update_ghost_mr(grad_alpha1_bar);
  auto MRadaptation = samurai::make_MRAdapt(grad_alpha1_bar);
  MRadaptation(MR_param, MR_regularity, conserved_variables);

  /*--- Sanity check (and numerical artefacts to clear data) after mesh adaptation ---*/
  alpha1_bar.resize();
  check_data(1);

  /*--- Recompute geoemtrical quantities ---*/
  normal.resize();
  H_bar.resize();
  grad_alpha1_bar.resize();
  update_geometry();
}

// Auxiliary fuction to check if spurious values are present
//
template<std::size_t dim>
void StaticBubble<dim>::check_data(unsigned int flag) {
  /*--- Re-update effective volume fraction ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                            alpha1_bar[cell] = conserved_variables[cell][RHO_ALPHA1_BAR_INDEX]/
                                               (conserved_variables[cell][M1_INDEX] +
                                                conserved_variables[cell][M2_INDEX] +
                                                conserved_variables[cell][M1_D_INDEX]);
                         });

  /*--- Check data ---*/
  #ifdef VERBOSE
    std::string op;
    if(flag == 0) {
      op = "after hyperbolic opeator (i.e. at the beginning of the relaxation)";
    }
    else {
      op = "after mesh adptation";
    }

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             // Sanity check for alpha1_bar
                             if(alpha1_bar[cell] < 0.0) {
                               std::cerr << cell << std::endl;
                               std::cerr << "Negative large-scale volume fraction of phase 1 " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }
                             else if(alpha1_bar[cell] > 1.0) {
                               std::cerr << cell << std::endl;
                               std::cerr << "Exceeding large-scale volume fraction of phase 1 " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }
                             else if(std::isnan(alpha1_bar[cell])) {
                               std::cerr << cell << std::endl;
                               std::cerr << "NaN large-scale volume fraction of phase 1 " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }

                             // Sanity check for m1
                             if(conserved_variables[cell][M1_INDEX] < 0.0) {
                               std::cerr << cell << std::endl;
                               std::cerr << "Negative large-scale mass of phase 1 " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }
                             else if(std::isnan(conserved_variables[cell][M1_INDEX])) {
                               std::cerr << cell << std::endl;
                               std::cerr << "NaN large-scale mass of phase 1 " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }

                             // Sanity check for m2
                             if(conserved_variables[cell][M2_INDEX] < 0.0) {
                               std::cerr << cell << std::endl;
                               std::cerr << "Negative mass of phase 2 " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }
                             else if(std::isnan(conserved_variables[cell][M2_INDEX])) {
                               std::cerr << cell << std::endl;
                               std::cerr << "NaN large-scale mass of phase 2 " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }

                             // Sanity check for m1_d
                             if(conserved_variables[cell][M1_D_INDEX] < 0.0) {
                               std::cerr << cell << std::endl;
                               std::cerr << "Negative small-scale mass of phase 1 " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }
                             else if(std::isnan(conserved_variables[cell][M1_D_INDEX])) {
                               std::cerr << cell << std::endl;
                               std::cerr << "NaN small-scale mass of phase 1 " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }

                             // Sanity check for alpha1_d
                             if(conserved_variables[cell][ALPHA1_D_INDEX] > 1.0) {
                               std::cerr << cell << std::endl;
                               std::cerr << "Exceding value of small-scale volume fraction " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }
                             else if(conserved_variables[cell][ALPHA1_D_INDEX] < 0.0) {
                               std::cerr << cell << std::endl;
                               std::cerr << "Negative small-scale volume fraction " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }
                             else if(std::isnan(conserved_variables[cell][ALPHA1_D_INDEX])) {
                               std::cerr << cell << std::endl;
                               std::cerr << "NaN small-scale volume fraction " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }

                             // Sanity check for Sigma_d
                             if(conserved_variables[cell][SIGMA_D_INDEX] < 0.0) {
                               std::cerr << cell << std::endl;
                               std::cerr << "Negative small-scale interfacial area" + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }
                             else if(std::isnan(conserved_variables[cell][SIGMA_D_INDEX])) {
                               std::cerr << cell << std::endl;
                               std::cerr << "NaN small-scale interfacial area " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar);
                               exit(1);
                             }

                             // Check data for mixture pressure
                             const auto alpha1 = alpha1_bar[cell]*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                             const auto rho1   = conserved_variables[cell][M1_INDEX]/alpha1; /*--- TODO: Add a check in case of zero volume fraction ---*/
                             p1[cell]          = EOS_phase1.pres_value(rho1);

                             const auto alpha2 = 1.0 - alpha1 - conserved_variables[cell][ALPHA1_D_INDEX];
                             const auto rho2   = conserved_variables[cell][M2_INDEX]/alpha2; /*--- TODO: Add a check in case of zero volume fraction ---*/
                             p2[cell]          = EOS_phase2.pres_value(rho2);

                             p_bar[cell]       = alpha1_bar[cell]*p1[cell] + (1.0 - alpha1_bar[cell])*p2[cell];
                             if(std::isnan(p_bar[cell])) {
                               std::cerr << cell << std::endl;
                               std::cerr << "NaN mxiture pressure " + op << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1_bar, p_bar);
                               exit(1);
                             }
                          });
  #endif
}

// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void StaticBubble<dim>::apply_relaxation() {
  /*--- Loop of Newton method. Conceptually, a loop over cells followed by a Newton loop
        over each cell would be more logic, but this would lead to issues to call 'update_geometry' ---*/
  std::size_t Newton_iter = 0;
  bool relaxation_applied = true;
  while(relaxation_applied == true) {
    relaxation_applied = false;
    Newton_iter++;

    // Loop over all cells.
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             #ifdef RUSANOV_FLUX
                               Rusanov_flux.perform_Newton_step_relaxation(std::make_unique<decltype(conserved_variables[cell])>(conserved_variables[cell]),
                                                                           H_bar[cell], dalpha1_bar[cell], alpha1_bar[cell],
                                                                           relaxation_applied);
                             #elifdef GODUNOV_FLUX
                               Godunov_flux.perform_Newton_step_relaxation(std::make_unique<decltype(conserved_variables[cell])>(conserved_variables[cell]),
                                                                           H_bar[cell], dalpha1_bar[cell], alpha1_bar[cell],
                                                                           relaxation_applied);
                             #endif

                           });

    // Recompute geometric quantities (curvature potentially changed in the Newton loop)
    //update_geometry();

    // Newton cycle diverged
    if(Newton_iter > max_Newton_iters) {
      std::cerr << "Netwon method not converged in the post-hyperbolic relaxation" << std::endl;
      save(fs::current_path(), "_diverged",
           conserved_variables, alpha1_bar, grad_alpha1_bar, normal, H_bar);
      exit(1);
    }
  }
}

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void StaticBubble<dim>::save(const fs::path& path,
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

// Execute postprocessing
//
template<std::size_t dim>
void StaticBubble<dim>::execute_postprocess_pressure_jump(const double time) {
  /*--- Compute pressure fields and maximum velocity ---*/
  p1.resize();
  p2.resize();
  p_bar.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto alpha1 = alpha1_bar[cell]*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                           const auto rho1   = conserved_variables[cell][M1_INDEX]/alpha1; /*--- TODO: Add a check in case of zero volume fraction ---*/
                           p1[cell]          = EOS_phase1.pres_value(rho1);

                           const auto alpha2 = 1.0 - alpha1 - conserved_variables[cell][ALPHA1_D_INDEX];
                           const auto rho2   = conserved_variables[cell][M2_INDEX]/alpha2; /*--- TODO: Add a check in case of zero volume fraction ---*/
                           p2[cell]          = EOS_phase2.pres_value(rho2);

                           p_bar[cell]       = alpha1_bar[cell]*p1[cell] + (1.0 - alpha1_bar[cell])*p2[cell];
                         });

  /*--- Threshold to define internal pressure ---*/
  const double alpha1_int = 0.99;

  /*--- Compute average pressure withint the droplet ---*/
  double p_in_avg = 0.0;
  double volume   = 0.0;
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           if(alpha1_bar[cell] > alpha1_int) {
                             p_in_avg += p_bar[cell]*std::pow(cell.length, dim);
                             volume   += std::pow(cell.length, dim);
                           }
                         });
  p_in_avg /= volume;

  /*--- Save the data ---*/
  const double p0   = EOS_phase1.get_p0();
  const double p_eq = p0 + sigma/R;
  pressure_data << std::fixed << std::setprecision(12) << time << '\t' << p_in_avg - p0 << '\t' << std::abs(p_in_avg - p_eq)/p_eq << std::endl;
}

/*---- IMPLEMENT THE FUNCTION THAT EFFECTIVELY SOLVES THE PROBLEM ---*/

// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void StaticBubble<dim>::run() {
  /*--- Default output arguemnts ---*/
  fs::path path = fs::current_path();
  filename = "static_bubble";
  #ifdef RUSANOV_FLUX
    filename += "_Rusanov";
  #elifdef GODUNOV_FLUX
    filename += "_Godunov";
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

  /*--- Create the flux variables ---*/
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
  #endif
  auto numerical_flux_st = SurfaceTension_flux.make_two_scale_capillarity(grad_alpha1_bar);

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, suffix_init, conserved_variables, alpha1_bar, grad_alpha1_bar, normal, H_bar, p1, p2, p_bar, vel);
  pressure_data.open("pressure_data.dat", std::ofstream::out);
  double t = 0.0;
  execute_postprocess_pressure_jump(t);

  /*--- Set initial time step ---*/
  const double dx = mesh.cell_length(mesh.max_level());
  max_velocity.open("max_velocity.dat", std::ofstream::out);
  double dt = std::min(Tf - t, cfl*dx/get_max_lambda(t));
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

    // Clear data to avoid small spurious negative values and recompute geoemtrical quantities
    check_data();
    update_geometry();

    // Capillarity contribution
    samurai::update_ghost_mr(grad_alpha1_bar);
    auto flux_st = numerical_flux_st(conserved_variables);
    #ifdef ORDER_2
      conserved_variables_tmp = conserved_variables - dt*flux_st;
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());
    #else
      conserved_variables_np1 = conserved_variables - dt*flux_st;
      std::swap(conserved_variables.array(), conserved_variables_np1.array());
    #endif

    /*--- Consider the second stage for the second order ---*/
    #ifdef ORDER_2
      // Apply the relaxation
      if(apply_relax) {
        // Apply relaxation if desired, which will modify alpha1 and, consequently, for what
        // concerns next time step, rho_alpha1
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                 dalpha1_bar[cell] = std::numeric_limits<typename Field::value_type>::infinity();
                               });
        apply_relaxation();
        update_geometry();
      }

      // Apply the numerical scheme
      // Convective operator
      samurai::update_ghost_mr(conserved_variables);
      flux_hyp = numerical_flux_hyp(conserved_variables);
      conserved_variables_tmp = conserved_variables - dt*flux_hyp;
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());

      // Check if spurious negative values arise and recompute geometrical quantities
      check_data();
      update_geometry();

      // Capillarity contribution
      samurai::update_ghost_mr(grad_alpha1_bar);
      flux_st = numerical_flux_st(conserved_variables);
      conserved_variables_tmp = conserved_variables - dt*flux_st;

      // Compute evaluation
      conserved_variables_np1.resize();
      conserved_variables_np1 = 0.5*(conserved_variables_old + conserved_variables_tmp);
      std::swap(conserved_variables.array(), conserved_variables_np1.array());

      // Recompute volume fraction gradient and curvature for the next time step
      update_geometry();
    #endif

    // Compute updated time step
    dt = std::min(Tf - t, cfl*dx/get_max_lambda(t));

    // Postprocess data
    execute_postprocess_pressure_jump(t);

    // Save the results
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, suffix, conserved_variables, alpha1_bar, grad_alpha1_bar, normal, H_bar, p1, p2, p_bar, vel);
    }
  } /*--- end of the time loop ---*/

  /*--- Close the output files ---*/
  pressure_data.close();
  max_velocity.close();
}
