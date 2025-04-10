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
#include <numbers>

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

/*--- Define preprocessor to check whether to control data or not ---*/
#define VERBOSE

/*--- Auxiliary function to compute the regualized Heaviside ---*/
template<typename T = double>
T CHeaviside(const T x, const T eps) {
  if(x < -eps) {
    return 0.0;
  }
  else if(x > eps) {
    return 1.0;
  }

  /*const double pi = 4.0*std::atan(1);
  return 0.5*(1.0 + x/eps + 1.0/pi*std::sin(pi*x/eps));*/

  return 0.5*(1.0 + std::tanh(8.0*(x/eps))/std::tanh(8.0));
}

/*--- Specify the use of this namespace where we just store the indices ---*/
using namespace EquationData;

/** This is the class for the simulation of a model
 *  for the waves-interface interaction
 **/
template<std::size_t dim>
class WaveInterface {
public:
  using Config = samurai::MRConfig<dim, 2, 2, 0>;

  WaveInterface() = default; /*--- Default constructor. This will do nothing
                                   and basically will never be used ---*/

  WaveInterface(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                const Simulation_Parameters& sim_param,
                const EOS_Parameters& eos_param); /*--- Class constrcutor with the arguments related
                                                        to the grid and to the physics ---*/

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
               dalpha1,
               p1,
               p2,
               p,
               rho,
               c_frozen,
               c_Wood,
               delta_p;

  Field_Vect   normal,
               grad_alpha1,
               u;

  using gradient_type = decltype(samurai::make_gradient_order2<decltype(alpha1)>());
  gradient_type gradient;

  using divergence_type = decltype(samurai::make_divergence_order2<decltype(normal)>());
  divergence_type divergence;

  const double sigma; /*--- Surface tension coefficient ---*/

  const double mod_grad_alpha1_min; /*--- Minimum threshold for which not computing anymore the unit normal ---*/

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

  double       MR_param;      /*--- Multiresolution parameter ---*/
  unsigned int MR_regularity; /*--- Multiresolution regularity ---*/

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void update_geometry(); /*--- Auxiliary routine to compute normals and curvature ---*/

  void init_variables(const double eps_interface_over_dx); /*--- Routine to initialize the variables
                                                                 (both conserved and auxiliary, this is problem dependent) ---*/

  double get_max_lambda(); /*--- Compute the estimate of the maximum eigenvalue ---*/

  void check_data(unsigned int flag = 0); /*--- Check if small negative values arise ---*/

  void perform_mesh_adaptation(); /*--- Perform the mesh adaptation ---*/

  void apply_relaxation(); /*--- Apply the relaxation ---*/
};

/*---- START WITH THE IMPLEMENTATION OF THE CONSTRUCTOR ---*/

// Implement class constructor
//
template<std::size_t dim>
WaveInterface<dim>::WaveInterface(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                                  const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                                  const Simulation_Parameters& sim_param,
                                  const EOS_Parameters& eos_param):
  box(min_corner, max_corner), mesh(box, sim_param.min_level, sim_param.max_level, {{false}}),
  apply_relax(sim_param.apply_relaxation), Tf(sim_param.Tf), cfl(sim_param.Courant),
  nfiles(sim_param.nfiles),
  gradient(samurai::make_gradient_order2<decltype(alpha1)>()),
  divergence(samurai::make_divergence_order2<decltype(normal)>()),
  sigma(sim_param.sigma),
  mod_grad_alpha1_min(sim_param.mod_grad_alpha1_min),
  max_Newton_iters(sim_param.max_Newton_iters),
  EOS_phase1(eos_param.p0_phase1, eos_param.rho0_phase1, eos_param.c0_phase1),
  EOS_phase2(eos_param.p0_phase2, eos_param.rho0_phase2, eos_param.c0_phase2),
  #ifdef RUSANOV_FLUX
    Rusanov_flux(EOS_phase1, EOS_phase2, sigma, mod_grad_alpha1_min,
                 sim_param.lambda, sim_param.atol_Newton, sim_param.rtol_Newton,
                 max_Newton_iters),
  #elifdef GODUNOV_FLUX
    Godunov_flux(EOS_phase1, EOS_phase2, sigma, mod_grad_alpha1_min,
                 sim_param.lambda, sim_param.atol_Newton, sim_param.rtol_Newton,
                 max_Newton_iters,
                 sim_param.atol_Newton_p_star, sim_param.rtol_Newton_p_star),
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
    init_variables(sim_param.eps_interface_over_dx);
  }

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void WaveInterface<dim>::init_variables(const double eps_interface_over_dx) {
  /*--- Create conserved and auxiliary fields ---*/
  conserved_variables = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved", mesh);

  alpha1   = samurai::make_field<typename Field::value_type, 1>("alpha1", mesh);
  H        = samurai::make_field<typename Field::value_type, 1>("H", mesh);
  dalpha1  = samurai::make_field<typename Field::value_type, 1>("dalpha1", mesh);

  normal      = samurai::make_field<typename Field::value_type, dim>("normal", mesh);
  grad_alpha1 = samurai::make_field<typename Field::value_type, dim>("grad_alpha1", mesh);

  p1  = samurai::make_field<typename Field::value_type, 1>("p1", mesh);
  p2  = samurai::make_field<typename Field::value_type, 1>("p2", mesh);
  p   = samurai::make_field<typename Field::value_type, 1>("p", mesh);
  rho = samurai::make_field<typename Field::value_type, 1>("rho", mesh);

  u = samurai::make_field<typename Field::value_type, dim>("u", mesh);

  c_frozen = samurai::make_field<typename Field::value_type, 1>("c_frozen", mesh);
  c_Wood   = samurai::make_field<typename Field::value_type, 1>("c_Wood", mesh);
  delta_p  = samurai::make_field<typename Field::value_type, 1>("delta_p", mesh);

  /*--- Declare some constant parameters associated to the grid and to the
        initial state ---*/
  const double x_shock       = 0.3;
  const double x_interface   = 0.7;
  const double dx            = mesh.cell_length(mesh.max_level());
  const double eps_interface = eps_interface_over_dx*dx;
  const double eps_shock     = 3.0*dx;

  /*--- Initialize some fields to define the bubble with a loop over all cells ---*/
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];

                           // Set volume fraction
                           alpha1[cell] = 1e-7 + (1.0 - 2e-7)*CHeaviside(x_interface - x, eps_interface)
                                               + (0.999999997719987 - 1.0 + 1e-7)*CHeaviside(x_shock - x , eps_shock);

                           // Set mass phase 1
                           double rho1;
                           if(x < x_shock) {
                             rho1 = 1001.8658;
                           }
                           else {
                             rho1 = 1000.0;
                           }
                           //const double rho1 = 1000.0 + (1001.8658 - 1000.0)*CHeaviside(x_shock - x, eps_shock);
                           p1[cell] = EOS_phase1.pres_value(rho1);
                           conserved_variables[cell][M1_INDEX] = alpha1[cell]*rho1;

                           // Set mass phase 2
                           p2[cell] = p1[cell];
                           const auto rho2 = EOS_phase2.rho_value(p2[cell]);
                           conserved_variables[cell][M2_INDEX] = (1.0 - alpha1[cell])*rho2;

                           // Set conserved variable associated to volume fraction
                           rho[cell] = conserved_variables[cell][M1_INDEX]
                                     + conserved_variables[cell][M2_INDEX];

                           conserved_variables[cell][RHO_ALPHA1_INDEX] = rho[cell]*alpha1[cell];

                           // Set momentum
                           if(x < x_shock) {
                             u[cell] = 3.0337923;
                           }
                           else {
                             u[cell] = 0.0;
                           }
                           //u[cell] = 3.0337923*CHeaviside(x_shock - x, eps_shock);
                           conserved_variables[cell][RHO_U_INDEX] = rho[cell]*u[cell];

                           // Set mixture pressure for output
                           p[cell] = alpha1[cell]*p1[cell] + (1.0 - alpha1[cell])*p2[cell];

                           delta_p[cell] = p1[cell] - p2[cell];

                           // Compute frozen speed of sound for output
                           c_frozen[cell] = std::sqrt((conserved_variables[cell][M1_INDEX]*
                                                       EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1) +
                                                       conserved_variables[cell][M2_INDEX]*
                                                       EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2))/rho[cell]);

                           // Compute Wood speed of sound
                           c_Wood[cell]  = std::sqrt(1.0/(rho[cell]*
                                                          (alpha1[cell]/(rho1*EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)) +
                                                          (1.0 - alpha1[cell])/(rho2*EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2)))));
                         });

  /*--- Consider Neumann bcs ---*/
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables, 0.0, 0.0, 0.0, 0.0);

  /*--- Set geometrical quantities ---*/
  update_geometry();
}

/*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/

// Auxiliary routine to compute normals and curvature
//
template<std::size_t dim>
void WaveInterface<dim>::update_geometry() {
  samurai::update_ghost_mr(alpha1);

  grad_alpha1 = gradient(alpha1);

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto mod_grad_alpha1 = std::sqrt(grad_alpha1[cell]*grad_alpha1[cell]);

                           if(mod_grad_alpha1 > mod_grad_alpha1_min) {
                             normal[cell] = grad_alpha1[cell]/mod_grad_alpha1;
                           }
                           else {
                             normal[cell] = nan("");
                           }
                         });
  samurai::update_ghost_mr(normal);
  H = -divergence(normal);
}

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double WaveInterface<dim>::get_max_lambda() {
  double res = 0.0;

  u.resize();
  c_frozen.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           #ifndef RELAX_RECONSTRUCTION
                              rho[cell]    = conserved_variables[cell][M1_INDEX]
                                           + conserved_variables[cell][M2_INDEX];
                              alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/rho[cell];
                           #endif
                           /*--- Compute the velocity ---*/
                           u[cell] = conserved_variables[cell][RHO_U_INDEX]/rho[cell];

                           /*--- Compute frozen speed of sound ---*/
                           const auto rho1 = conserved_variables[cell][M1_INDEX]/alpha1[cell];
                           /*--- TODO: Add a check in case of zero volume fraction ---*/
                           const auto rho2 = conserved_variables[cell][M2_INDEX]/(1.0 - alpha1[cell]);
                           /*--- TODO: Add a check in case of zero volume fraction ---*/
                           const auto c_squared = conserved_variables[cell][M1_INDEX]*
                                                  EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)
                                                + conserved_variables[cell][M2_INDEX]*
                                                  EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2);
                           c_frozen[cell] = std::sqrt(c_squared/rho[cell]);

                           /*--- Add term due to surface tension ---*/
                           const double r = sigma*std::sqrt(grad_alpha1[cell]*grad_alpha1[cell])/(rho[cell]*c_frozen[cell]*c_frozen[cell]);

                           /*--- Update eigenvalue estimate ---*/
                           res = std::max(std::abs(u[cell]) + c_frozen[cell]*(1.0 + 0.125*r), res);
                         });

  return res;
}

// Perform the mesh adaptation strategy.
//
template<std::size_t dim>
void WaveInterface<dim>::perform_mesh_adaptation() {
  samurai::update_ghost_mr(alpha1);
  auto MRadaptation = samurai::make_MRAdapt(alpha1);
  MRadaptation(MR_param, MR_regularity, conserved_variables);

  /*--- Sanity check after mesh adaptation ---*/
  rho.resize();
  alpha1.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           rho[cell]    = conserved_variables[cell][M1_INDEX]
                                        + conserved_variables[cell][M2_INDEX];
                           alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/rho[cell];
                         });
  #ifdef VERBOSE
    check_data(1);
  #endif
}

// Numerical artefact to avoid small negative values
//
#ifdef VERBOSE
template<std::size_t dim>
void WaveInterface<dim>::check_data(unsigned int flag) {
  std::string op;
  if(flag == 0) {
    op = "at the beginning of the relaxation";
  }
  else {
    op = "after mesh adaptation";
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

                           // Sanity rho_alpha1
                           if(conserved_variables[cell][RHO_ALPHA1_INDEX] < 0.0) {
                             std::cerr << cell << std::endl;
                             std::cerr << " Negative volume fraction " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                           else if(std::isnan(conserved_variables[cell][RHO_ALPHA1_INDEX]) < 0.0) {
                             std::cerr << cell << std::endl;
                             std::cerr << " NaN volume fraction " + op << std::endl;
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
                             std::cerr << "NaN large-scale mass of phase 1 " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }

                           // Sanity check for m2
                           if(conserved_variables[cell][M2_INDEX] < 0.0) {
                             std::cerr << cell << std::endl;
                             std::cerr << "Negative mass of phase 2 " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables);
                             exit(1);
                           }
                           else if(std::isnan(conserved_variables[cell][M2_INDEX])) {
                             std::cerr << cell << std::endl;
                             std::cerr << "NaN large-scale mass of phase 2 " + op << std::endl;
                             save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             exit(1);
                           }
                          });
}
#endif

// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void WaveInterface<dim>::apply_relaxation() {
  /*--- Loop of Newton method ---*/
  std::size_t Newton_iter = 0;
  bool relaxation_applied = true;
  while(relaxation_applied == true) {
    relaxation_applied = false;
    Newton_iter++;

    // Loop over all cells.
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             try {
                               #ifdef RUSANOV_FLUX
                                  Rusanov_flux.perform_Newton_step_relaxation(std::make_unique<decltype(conserved_variables[cell])>(conserved_variables[cell]),
                                                                              H[cell], dalpha1[cell], alpha1[cell],
                                                                              relaxation_applied);
                              #elifdef GODUNOV_FLUX
                                  Godunov_flux.perform_Newton_step_relaxation(std::make_unique<decltype(conserved_variables[cell])>(conserved_variables[cell]),
                                                                              H[cell], dalpha1[cell], alpha1[cell],
                                                                              relaxation_applied);
                              #endif
                             }
                             catch(std::exception& e) {
                               std::cerr << e.what() << std::endl;
                               save(fs::current_path(), "_diverged",
                                    conserved_variables, alpha1);
                               exit(1);
                             }
                           });

    // Recompute geometric quantities (curvature potentially changed in the Newton loop)
    update_geometry();

    // Newton cycle diverged
    if(Newton_iter > max_Newton_iters && relaxation_applied == true) {
      std::cerr << "Netwon method not converged in the post-hyperbolic relaxation" << std::endl;
      save(fs::current_path(), "_diverged",
           conserved_variables, alpha1, rho);
      exit(1);
    }
  }
}

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void WaveInterface<dim>::save(const fs::path& path,
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

/*---- IMPLEMENT THE FUNCTION THAT EFFECTIVELY SOLVES THE PROBLEM ---*/

// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void WaveInterface<dim>::run() {
  /*--- Default output arguemnts ---*/
  fs::path path = fs::current_path();
  filename = "waves_interface";
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

  /*--- Auxiliary variable to save updated fields ---*/
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
  #endif
  auto numerical_flux_st = SurfaceTension_flux.make_flux_capillarity(grad_alpha1);

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, suffix_init, conserved_variables, alpha1, rho, p1, p2, p, u, c_frozen, c_Wood, delta_p);

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
                             rho[cell]    = conserved_variables[cell][M1_INDEX]
                                          + conserved_variables[cell][M2_INDEX];
                             alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/rho[cell];
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
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               dalpha1[cell] = std::numeric_limits<typename Field::value_type>::infinity();
                             });
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
                               rho[cell]    = conserved_variables[cell][M1_INDEX]
                                            + conserved_variables[cell][M2_INDEX];
                               alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/rho[cell];
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
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                 dalpha1[cell] = std::numeric_limits<typename Field::value_type>::infinity();
                               });
        apply_relaxation();
        #ifdef RELAX_RECONSTRUCTION
          update_geometry();
        #endif
      }

      // Complete evaluation
      conserved_variables_np1.resize();
      conserved_variables_np1 = 0.5*(conserved_variables_old + conserved_variables);
      std::swap(conserved_variables.array(), conserved_variables_np1.array());

      // Recompute volume fraction gradient and curvature for the next time step (if needed)
      #ifdef RELAX_RECONSTRUCTION
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                 rho[cell]    = conserved_variables[cell][M1_INDEX]
                                              + conserved_variables[cell][M2_INDEX];
                                 alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/rho[cell];
                               });
        update_geometry();
      #endif
    #endif

    // Compute updated time step
    dt = std::min(Tf - t, cfl*dx/get_max_lambda());

    // Save the results
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";

      // Compute auxliary fields for the output
      p1.resize();
      p2.resize();
      p.resize();
      delta_p.resize();
      c_Wood.resize();
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               // Compute pressure fields
                               const auto rho1 = conserved_variables[cell][M1_INDEX]/alpha1[cell];
                               /*--- TODO: Add a check in case of zero volume fraction ---*/
                               p1[cell] = EOS_phase1.pres_value(rho1);

                               const auto rho2 = conserved_variables[cell][M2_INDEX]/(1.0 - alpha1[cell]);
                               /*--- TODO: Add a check in case of zero volume fraction ---*/
                               p2[cell] = EOS_phase2.pres_value(rho2);

                               p[cell] = alpha1[cell]*p1[cell]
                                       + (1.0 - alpha1[cell])*p2[cell];

                               delta_p[cell] = p1[cell] - p2[cell];

                               // Compute Wood speed of sound
                               c_Wood[cell] = std::sqrt(1.0/(rho[cell]*
                                                             (alpha1[cell]/(rho1*EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)) +
                                                             (1.0 - alpha1[cell])/(rho2*EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2)))));
                             });

      save(path, suffix, conserved_variables, alpha1, rho, p1, p2, p, u, c_frozen, c_Wood, delta_p);
    }
  }
}
