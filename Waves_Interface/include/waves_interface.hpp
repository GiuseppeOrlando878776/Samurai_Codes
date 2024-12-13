// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <numbers>

#include <filesystem>
namespace fs = std::filesystem;

// Add header file for the multiresolution
#include <samurai/mr/adapt.hpp>

// Add header with auxiliary structs
#include "containers.hpp"

// Include the headers with the numerical fluxes
//#define RUSANOV_FLUX
#define GODUNOV_FLUX

#ifdef RUSANOV_FLUX
  #include "Rusanov_flux.hpp"
#elifdef GODUNOV_FLUX
  #include "Exact_Godunov_flux.hpp"
#endif

// Define preprocessor to check whether to control data or not
#define VERBOSE

// Auxiliary function to compute the regualized Heaviside
template<typename T = double>
T CHeaviside(const T x, const T eps) {
  if(x < -eps) {
    return 0.0;
  }
  else if(x > eps) {
    return 1.0;
  }

  const double pi = 4.0*std::atan(1);
  return 0.5*(1.0 + x/eps + 1.0/pi*std::sin(pi*x/eps));

  //return 0.5 + 0.5*std::tanh(8.0*(x/eps + 0.5));
}

// Specify the use of this namespace where we just store the indices
// and, in this case, some parameters related to EOS
using namespace EquationData;

/** This is the class for the simulation of a model
 *  for the waves-interface interaction
 **/
template<std::size_t dim>
class WaveInterface {
public:
  using Config = samurai::MRConfig<dim, 2, 2, 2>;

  WaveInterface() = default; // Default constructor. This will do nothing
                             // and basically will never be used

  WaveInterface(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                const Simulation_Parameters& sim_param,
                const EOS_Parameters& eos_param); // Class constrcutor with the arguments related
                                                  // to the grid and to the physics.

  void run(); // Function which actually executes the temporal loop

  template<class... Variables>
  void save(const fs::path& path,
            const std::string& suffix,
            const Variables&... fields); // Routine to save the results

private:
  /*--- Now we declare some relevant variables ---*/
  const samurai::Box<double, dim> box;

  samurai::MRMesh<Config> mesh; // Variable to store the mesh

  using Field        = samurai::Field<decltype(mesh), double, EquationData::NVARS, false>;
  using Field_Scalar = samurai::Field<decltype(mesh), typename Field::value_type, 1, false>;
  using Field_Vect   = samurai::Field<decltype(mesh), typename Field::value_type, dim, false>;

  bool apply_relax; // Choose whether to apply or not the relaxation

  double Tf;  // Final time of the simulation
  double cfl; // Courant number of the simulation so as to compute the time step

  std::size_t nfiles; // Number of files desired for output

  Field conserved_variables; // The variable which stores the conserved variables,
                             // namely the varialbes for which we solve a PDE system

  // Now we declare a bunch of fields which depend from the state, but it is useful
  // to have it so as to avoid recomputation
  Field_Scalar alpha1,
               dalpha1,
               p1,
               p2,
               p,
               p_minus_p0,
               rho,
               c_frozen,
               c_Wood;

  Field_Vect   u;

  LinearizedBarotropicEOS<typename Field::value_type> EOS_phase1,
                                                      EOS_phase2; // The two variables which take care of the
                                                                  // barotropic EOS to compute the speed of sound
  #ifdef RUSANOV_FLUX
    samurai::RusanovFlux<Field> Rusanov_flux; // Auxiliary variable to compute the flux
  #elifdef GODUNOV_FLUX
    samurai::GodunovFlux<Field> Godunov_flux; // Auxiliary variable to compute the flux
  #endif

  std::string filename; // Auxiliary variable to store the name of output

  const double MR_param; // Multiresolution parameter
  const double MR_regularity; // Multiresolution regularity

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void init_variables(const double eps_interface_over_dx); // Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)

  double get_max_lambda() const; // Compute the estimate of the maximum eigenvalue

  #ifdef VERBOSE
    void check_data(unsigned int flag = 0); // Numerical artefact to avoid spurious small negative values
  #endif

  void perform_mesh_adaptation(); // Perform the mesh adaptation

  void apply_relaxation(); // Apply the relaxation
};

/*---- START WITH THE IMPLEMENTATION OF THE CONSTRUCTOR ---*/

// Implement class constructor
//
template<std::size_t dim>
WaveInterface<dim>::WaveInterface(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                                  const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                                  const Simulation_Parameters& sim_param,
                                  const EOS_Parameters& eos_param):
  box(min_corner, max_corner), mesh(box, sim_param.min_level, sim_param.max_level, {true}),
  apply_relax(sim_param.apply_relaxation), Tf(sim_param.Tf), cfl(sim_param.Courant),
  nfiles(sim_param.nfiles),
  EOS_phase1(eos_param.p0_phase1, eos_param.rho0_phase1, eos_param.c0_phase1),
  EOS_phase2(eos_param.p0_phase2, eos_param.rho0_phase2, eos_param.c0_phase2),
  #ifdef RUSANOV_FLUX
    Rusanov_flux(EOS_phase1, EOS_phase2),
  #elifdef GODUNOV_FLUX
    Godunov_flux(EOS_phase1, EOS_phase2),
  #endif
  MR_param(sim_param.MR_param), MR_regularity(sim_param.MR_regularity)
  {
    std::cout << "Initializing variables " << std::endl;
    std::cout << std::endl;
    init_variables(sim_param.eps_interface_over_dx);
  }

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void WaveInterface<dim>::init_variables(const double eps_interface_over_dx) {
  // Create conserved and auxiliary fields
  conserved_variables = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved", mesh);

  alpha1   = samurai::make_field<typename Field::value_type, 1>("alpha1", mesh);

  dalpha1  = samurai::make_field<typename Field::value_type, 1>("dalpha1", mesh);

  p1         = samurai::make_field<typename Field::value_type, 1>("p1", mesh);
  p2         = samurai::make_field<typename Field::value_type, 1>("p2", mesh);
  p          = samurai::make_field<typename Field::value_type, 1>("p", mesh);
  p_minus_p0 = samurai::make_field<typename Field::value_type, 1>("p_minus_p0", mesh);
  rho        = samurai::make_field<typename Field::value_type, 1>("rho", mesh);
  c_frozen   = samurai::make_field<typename Field::value_type, 1>("c_frozen", mesh);
  c_Wood     = samurai::make_field<typename Field::value_type, 1>("c_Wood", mesh);

  u          = samurai::make_field<typename Field::value_type, dim>("u", mesh);

  // Declare some constant parameters associated to the grid and to the
  // initial state
  const double x_interface   = 0.7;
  const double dx            = mesh.cell_length(mesh.max_level());
  const double eps_interface = eps_interface_over_dx*dx;

  // Initialize some fields to define the bubble with a loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];

                           // Set volume fraction
                           alpha1[cell] = (1.0 - 1e-7) - (1.0 - 2e-7)*CHeaviside(x - x_interface, eps_interface);

                           // Set mass phase 1
                           if(x >= 0.45 && x <= 0.55) {
                             p1[cell] = 1e5 + std::sin((2.0*4.0*std::atan(1)*(x - 0.5))/0.1);
                           }
                           else {
                             p1[cell] = 1e5;
                           }
                           const auto rho1 = EOS_phase1.rho_value(p1[cell]);
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
                           u[cell] = 0.0;
                           conserved_variables[cell][RHO_U_INDEX] = rho[cell]*u[cell];

                           // Set mixture pressure for output
                           p[cell] = alpha1[cell]*p1[cell]
                                   + (1.0 - alpha1[cell])*p2[cell];

                           p_minus_p0[cell] = p[cell] - 1e5;

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

  // Consider Neumann bcs
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables, 0.0, 0.0, 0.0, 0.0);
}

/*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double WaveInterface<dim>::get_max_lambda() const {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Compute the velocity
                           const auto rho_    = conserved_variables[cell][M1_INDEX]
                                              + conserved_variables[cell][M2_INDEX];
                           const auto alpha1_ = conserved_variables[cell][RHO_ALPHA1_INDEX]/rho_;

                           const auto vel = conserved_variables[cell][RHO_U_INDEX]/rho_;

                           // Compute frozen speed of sound
                           const auto rho1      = conserved_variables[cell][M1_INDEX]/alpha1_;
                           /*--- TODO: Add a check in case of zero volume fraction ---*/
                           const auto rho2      = conserved_variables[cell][M2_INDEX]/(1.0 - alpha1_);
                           /*--- TODO: Add a check in case of zero volume fraction ---*/
                           const auto c_squared = conserved_variables[cell][M1_INDEX]*
                                                  EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)
                                                + conserved_variables[cell][M2_INDEX]*
                                                  EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2);
                           const auto c         = std::sqrt(c_squared/rho_);

                           // Update eigenvalue estimate
                           res = std::max(std::abs(vel) + c, res);
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

  // Sanity check after mesh adaptation
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
                            // Start with rho_alpha1
                            if(conserved_variables[cell][RHO_ALPHA1_INDEX] < 0.0) {
                              std::cerr << " Negative volume fraction " + op << std::endl;
                              save(fs::current_path(), "_diverged", conserved_variables);
                              exit(1);
                            }
                            // Sanity check for m1
                            if(conserved_variables[cell][M1_INDEX] < 0.0) {
                              std::cerr << "Negative mass for phase 1 " + op << std::endl;
                              save(fs::current_path(), "_diverged", conserved_variables);
                              exit(1);
                            }
                            // Sanity check for m2
                            if(conserved_variables[cell][M2_INDEX] < 0.0) {
                              std::cerr << "Negative mass for phase 2 " + op << std::endl;
                              save(fs::current_path(), "_diverged", conserved_variables);
                              exit(1);
                            }
                          });
}
#endif

// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void WaveInterface<dim>::apply_relaxation() {
  const double tol    = 1e-12; // Tolerance of the Newton method
  const double lambda = 0.9;   // Parameter for bound preserving strategy

  // Loop of Newton method.
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
                                                                           dalpha1[cell], alpha1[cell], rho[cell], relaxation_applied, tol, lambda);
                             #elifdef GODUNOV_FLUX
                               Godunov_flux.perform_Newton_step_relaxation(std::make_unique<decltype(conserved_variables[cell])>(conserved_variables[cell]),
                                                                           dalpha1[cell], alpha1[cell], rho[cell], relaxation_applied, tol, lambda);
                             #endif

                           });

    // Newton cycle diverged
    if(Newton_iter > 60) {
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
  // Default output arguemnts
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

  // Auxiliary variable to save updated fields
  #ifdef ORDER_2
    auto conserved_variables_tmp   = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved_tmp", mesh);
    auto conserved_variables_tmp_2 = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved_tmp_2", mesh);
  #endif
  auto conserved_variables_np1 = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved_np1", mesh);

  // Create the flux variable
  #ifdef RUSANOV_FLUX
    auto numerical_flux = Rusanov_flux.make_flux();
  #elifdef GODUNOV_FLUX
    auto numerical_flux = Godunov_flux.make_flux();
  #endif

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, suffix_init, conserved_variables, alpha1, rho, p1, p2, p, p_minus_p0, u, c_frozen, c_Wood);

  // Start the loop
  const double dx = mesh.cell_length(mesh.max_level());
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  std::cout << "Number of elements = " << mesh[mesh_id_t::cells].nb_cells() << std::endl;
  std::cout << std::endl;

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

    std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

    /*--- Apply mesh adaptation ---*/
    perform_mesh_adaptation();

    /*--- Apply the numerical scheme without relaxation ---*/
    samurai::update_ghost_mr(conserved_variables);
    try {
      auto flux_conserved = numerical_flux(conserved_variables);
      #ifdef ORDER_2
        conserved_variables_tmp.resize();
        conserved_variables_tmp = conserved_variables - dt*flux_conserved;
        std::swap(conserved_variables.array(), conserved_variables_tmp.array());
      #else
        conserved_variables_np1.resize();
        conserved_variables_np1 = conserved_variables - dt*flux_conserved;
        std::swap(conserved_variables.array(), conserved_variables_np1.array());
      #endif
    }
    catch(std::exception& e) {
      std::cerr << e.what() << std::endl;
      save(fs::current_path(), "_diverged", conserved_variables);
      exit(1);
    }
    #ifdef VERBOSE
      check_data();
    #endif

    /*--- Apply relaxation ---*/
    if(apply_relax) {
      // Apply relaxation if desired, which will modify alpha1 and, consequently, for what
      // concerns next time step, rho_alpha1
      dalpha1.resize();
      alpha1.resize();
      rho.resize();
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               rho[cell]     = conserved_variables[cell][M1_INDEX]
                                             + conserved_variables[cell][M2_INDEX];
                               alpha1[cell]  = conserved_variables[cell][RHO_ALPHA1_INDEX]/rho[cell];
                               dalpha1[cell] = std::numeric_limits<typename Field::value_type>::infinity();
                             });
      apply_relaxation();
    }

    /*--- Consider the second stage for the second order ---*/
    #ifdef ORDER_2
      // Apply the numerical scheme
      samurai::update_ghost_mr(conserved_variables);
      try {
        auto flux_conserved = numerical_flux(conserved_variables);
        conserved_variables_tmp_2.resize();
        conserved_variables_tmp_2 = conserved_variables - dt*flux_conserved;
        conserved_variables_np1.resize();
        conserved_variables_np1 = 0.5*(conserved_variables_tmp + conserved_variables_tmp_2);
        std::swap(conserved_variables.array(), conserved_variables_np1.array());
      }
      catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
        save(fs::current_path(), "_diverged", conserved_variables);
        exit(1);
      }
      #ifdef VERBOSE
        check_data();
      #endif

      // Apply the relaxation
      if(apply_relax) {
        // Apply relaxation if desired, which will modify alpha1 and, consequently, for what
        // concerns next time step, rho_alpha1
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                 rho[cell]     = conserved_variables[cell][M1_INDEX]
                                               + conserved_variables[cell][M2_INDEX];
                                 alpha1[cell]  = conserved_variables[cell][RHO_ALPHA1_INDEX]/rho[cell];
                                 dalpha1[cell] = std::numeric_limits<typename Field::value_type>::infinity();
                               });
        apply_relaxation();
      }
    #endif

    /*--- Compute updated time step ---*/
    dt = std::min(Tf - t, cfl*dx/get_max_lambda());

    /*--- Save the results ---*/
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";

      // Compute auxliary fields for the output
      rho.resize();
      alpha1.resize();
      p1.resize();
      p2.resize();
      p.resize();
      u.resize();
      c_frozen.resize();
      c_Wood.resize();
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               // Recompute density and volmue fraction in case relaxation is not applied
                               // (and therefore they have not been updated)
                               if(!apply_relax) {
                                 rho[cell]    = conserved_variables[cell][M1_INDEX]
                                              + conserved_variables[cell][M2_INDEX];
                                 alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/rho[cell];
                               }

                               // Compute pressure fields
                               const auto rho1 = conserved_variables[cell][M1_INDEX]/alpha1[cell];
                               /*--- TODO: Add a check in case of zero volume fraction ---*/
                               p1[cell] = EOS_phase1.pres_value(rho1);

                               const auto rho2 = conserved_variables[cell][M2_INDEX]/(1.0 - alpha1[cell]);
                               /*--- TODO: Add a check in case of zero volume fraction ---*/
                               p2[cell] = EOS_phase2.pres_value(rho2);

                               p[cell] = alpha1[cell]*p1[cell]
                                       + (1.0 - alpha1[cell])*p2[cell];

                               p_minus_p0[cell] = p[cell] - 1e5;

                               // Compute velocity field
                               u[cell] = conserved_variables[cell][RHO_U_INDEX]/rho[cell];

                               // Compute frozen speed of sound
                               c_frozen[cell] = std::sqrt((conserved_variables[cell][M1_INDEX]*
                                                           EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1) +
                                                           conserved_variables[cell][M2_INDEX]*
                                                           EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2))/rho[cell]);

                               // Compute Wood speed of sound
                               c_Wood[cell] = std::sqrt(1.0/(rho[cell]*
                                                             (alpha1[cell]/(rho1*EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)) +
                                                             (1.0 - alpha1[cell])/(rho2*EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2)))));
                             });

      save(path, suffix, conserved_variables, alpha1, rho, p1, p2, p, p_minus_p0, u, c_frozen, c_Wood);
    }
  }
}
