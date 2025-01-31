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

// Add header for PETSC
#include <samurai/petsc.hpp>

// Add header with auxiliary structs
#include "containers.hpp"

// Add user implemented boundary condition
#include "user_bc.hpp"

// Include the headers with the numerical fluxes
//#define RUSANOV_FLUX
//#define HLL_FLUX
//#define HLLC_FLUX
#define GODUNOV_FLUX

#ifdef RUSANOV_FLUX
  #include "Rusanov_flux.hpp"
#elifdef HLL_FLUX
  #include "HLL_flux.hpp"
#elifdef HLLC_FLUX
  #include "HLLC_flux.hpp"
#elifdef GODUNOV_FLUX
  #include "Exact_Godunov_flux.hpp"
#endif

// Define preprocessor to check whether to control data or not
#define VERBOSE

// Define preprocessor to choose whether gravity has to be tretaed implicitly or not
#define GRAVITY_IMPLICIT

// Specify the use of this namespace where we just store the indices
// and, in this case, some parameters related to EOS
using namespace EquationData;

/** This is the class for the simulation of a model
 *  for the waves-interface interaction
 **/
template<std::size_t dim>
class DamBreak {
public:
  using Config = samurai::MRConfig<dim, 2, 2, 0>;

  DamBreak() = default; // Default constructor. This will do nothing
                        // and basically will never be used

  DamBreak(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
           const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
           const Simulation_Paramaters& sim_param,
           const EOS_Parameters& eos_param); // Class constrcutor with the arguments related
                                             // to the grid and to the physics.

  void run(); // Function which actually executes the temporal loop

  template<class... Variables>
  void save(const fs::path& path,
            const std::string& suffix,
            const Variables&... fields); // Routine to save the results

private:
  // Now we declare some relevant variables
  const samurai::Box<double, dim> box;

  samurai::MRMesh<Config> mesh; // Variable to store the mesh

  using Field        = samurai::Field<decltype(mesh), double, EquationData::NVARS, false>;
  using Field_Scalar = samurai::Field<decltype(mesh), typename Field::value_type, 1, false>;
  using Field_Vect   = samurai::Field<decltype(mesh), typename Field::value_type, dim, false>;

  bool apply_relax; // Choose whether to apply or not the relaxation

  double Tf;  // Final time of the simulation
  double cfl; // Courant number of the simulation so as to compute the time step

  double L0; // Initial length dam
  double H0; // Initial height dam
  double W0; // Initial weigth dam

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
               rho;

  Field_Vect   vel,
               grad_alpha1;

  using gradient_type = decltype(samurai::make_gradient_order2<decltype(alpha1)>());
  gradient_type gradient;

  LinearizedBarotropicEOS<typename Field::value_type> EOS_phase1,
                                                      EOS_phase2; // The two variables which take care of the
                                                                  // barotropic EOS to compute the speed of sound
  #ifdef RUSANOV_FLUX
    samurai::RusanovFlux<Field> Rusanov_flux; // Auxiliary variable to compute the flux
  #elifdef HLL_FLUX
    samurai::HLLFlux<Field> HLL_flux; // Auxiliary variable to compute the flux
  #elifdef HLLC_FLUX
    samurai::HLLCFlux<Field> HLLC_flux; // Auxiliary variable to compute the flux
  #elifdef GODUNOV_FLUX
    samurai::GodunovFlux<Field> Godunov_flux; // Auxiliary variable to compute the flux
  #endif

  std::string filename; // Auxiliary variable to store the name of output

  const double MR_param; // Multiresolution parameter
  const double MR_regularity; // Multiresolution regularity

  // Now, it's time to declare some member functions that we will employ
  void init_variables(); // Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)

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
DamBreak<dim>::DamBreak(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                        const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                        const Simulation_Paramaters& sim_param,
                        const EOS_Parameters& eos_param):
  box(min_corner, max_corner), mesh(box, sim_param.min_level, sim_param.max_level, {false, false, false}),
  apply_relax(sim_param.apply_relaxation),
  Tf(sim_param.Tf), cfl(sim_param.Courant),
  L0(sim_param.L0), H0(sim_param.H0), W0(sim_param.W0),
  nfiles(sim_param.nfiles),
  gradient(samurai::make_gradient_order2<decltype(alpha1)>()),
  EOS_phase1(eos_param.p0_phase1, eos_param.rho0_phase1, eos_param.c0_phase1),
  EOS_phase2(eos_param.p0_phase2, eos_param.rho0_phase2, eos_param.c0_phase2),
  #ifdef RUSANOV_FLUX
    Rusanov_flux(EOS_phase1, EOS_phase2),
  #elifdef HLL_FLUX
    HLL_flux(EOS_phase1, EOS_phase2),
  #elifdef HLLC_FLUX
    HLLC_flux(EOS_phase1, EOS_phase2),
  #elifdef GODUNOV_FLUX
    Godunov_flux(EOS_phase1, EOS_phase2),
  #endif
  MR_param(sim_param.MR_param), MR_regularity(sim_param.MR_regularity)
  {
    std::cout << "Initializing variables " << std::endl;
    std::cout << std::endl;
    init_variables();
  }

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void DamBreak<dim>::init_variables() {
  // Create conserved and auxiliary fields
  conserved_variables = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved", mesh);

  alpha1  = samurai::make_field<typename Field::value_type, 1>("alpha1", mesh);

  grad_alpha1 = samurai::make_field<typename Field::value_type, dim>("grad_alpha1", mesh);

  dalpha1 = samurai::make_field<typename Field::value_type, 1>("dalpha1", mesh);

  p1      = samurai::make_field<typename Field::value_type, 1>("p1", mesh);
  p2      = samurai::make_field<typename Field::value_type, 1>("p2", mesh);
  p       = samurai::make_field<typename Field::value_type, 1>("p", mesh);
  rho     = samurai::make_field<typename Field::value_type, 1>("rho", mesh);

  vel     = samurai::make_field<typename Field::value_type, dim>("u", mesh);

  // Initialize some fields to define the bubble with a loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];
                           const double y    = center[1];
                           const double z    = center[2];

                           // Set volume fraction
                           if(x < L0 && y < H0 && z < W0) {
                             alpha1[cell] = 1.0 - 1e-6;
                           }
                           else {
                             alpha1[cell] = 1e-6;
                           }

                           // Set mass phase 1
                           p1[cell] = EOS_phase1.get_p0();
                           conserved_variables[cell][M1_INDEX] = alpha1[cell]*EOS_phase1.rho_value(p1[cell]);

                           // Set mass phase 2
                           p2[cell] = EOS_phase2.get_p0();
                           conserved_variables[cell][M2_INDEX] = (1.0 - alpha1[cell])*EOS_phase2.rho_value(p2[cell]);

                           // Set conserved variable associated to volume fraction
                           rho[cell] = conserved_variables[cell][M1_INDEX]
                                     + conserved_variables[cell][M2_INDEX];

                           conserved_variables[cell][RHO_ALPHA1_INDEX] = rho[cell]*alpha1[cell];

                           // Set momentum
                           for(std::size_t d = 0; d < Field::dim; ++ d) {
                             vel[cell][d] = 0.0;
                             conserved_variables[cell][RHO_U_INDEX + d] = rho[cell]*vel[cell][d];
                           }

                           // Set mixture pressure for output
                           p[cell] = alpha1[cell]*p1[cell]
                                   + (1.0 - alpha1[cell])*p2[cell];
                         });

  // Consider non-reflecting bcs
  samurai::make_bc<Default>(conserved_variables, NonReflecting(conserved_variables));
}

/*---- FOCUS NOW ON THE AUXILIARY FUNCTIONS ---*/

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double DamBreak<dim>::get_max_lambda() const {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Compute the velocity along both horizontal and vertical direction
                           const auto rho_    = conserved_variables[cell][M1_INDEX]
                                              + conserved_variables[cell][M2_INDEX];
                           const auto alpha1_ = conserved_variables[cell][RHO_ALPHA1_INDEX]/rho_;

                           const auto vel_x = conserved_variables[cell][RHO_U_INDEX]/rho_;
                           const auto vel_y = conserved_variables[cell][RHO_U_INDEX + 1]/rho_;
                           const auto vel_z = conserved_variables[cell][RHO_U_INDEX + 2]/rho_;

                           // Compute frozen speed of sound
                           const auto rho1      = conserved_variables[cell][M1_INDEX]/alpha1_;
                           /*--- TODO: Add a check in case of zero volume fraction ---*/
                           const auto rho2      = conserved_variables[cell][M2_INDEX]/(1.0 - alpha1_);
                           /*--- TODO: Add a check in case of zero volume fraction ---*/
                           const auto c_squared = conserved_variables[cell][M1_INDEX]*EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)
                                                + conserved_variables[cell][M2_INDEX]*EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2);
                           const auto c         = std::sqrt(c_squared/rho_);

                           // Update eigenvalue estimate
                           res = std::max(std::max(std::max(std::abs(vel_x) + c,
                                                            std::abs(vel_y) + c),
                                                   std::abs(vel_z) + c),
                                          res);
                         });

  return res;
}

// Perform the mesh adaptation strategy.
//
template<std::size_t dim>
void DamBreak<dim>::perform_mesh_adaptation() {
  alpha1.resize();
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                          (conserved_variables[cell][M1_INDEX] +
                                           conserved_variables[cell][M2_INDEX]);
                         });
  samurai::update_ghost_mr(alpha1);
  grad_alpha1.resize();
  grad_alpha1 = gradient(alpha1);

  auto MRadaptation = samurai::make_MRAdapt(grad_alpha1);
  MRadaptation(MR_param, MR_regularity, conserved_variables);

  // Sanity check after mesh adaptation
  #ifdef VERBOSE
    check_data(1);
  #endif

  save(fs::current_path(), "_after_mesh_adaptation",
       conserved_variables, alpha1);
}

// Numerical artefact to avoid small negative values
//
#ifdef VERBOSE
template<std::size_t dim>
void DamBreak<dim>::check_data(unsigned int flag) {
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
                            // Start with the volume fraction
                            alpha1[cell] = conserved_variables[cell][RHO_ALPHA1_INDEX]/
                                           (conserved_variables[cell][M1_INDEX] +
                                            conserved_variables[cell][M2_INDEX]);
                            if(alpha1[cell] < 0.0) {
                              std::cerr << "Negative volume fraction for phase 1 " + op << std::endl;
                              save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                              exit(1);
                            }
                            else if(alpha1[cell] > 1.0) {
                              std::cerr << "Exceeding volume fraction for phase 1 " + op << std::endl;
                              save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                              //exit(1);
                            }
                            // Focus now on rho_alpha1
                            if(conserved_variables[cell][RHO_ALPHA1_INDEX] < 0.0) {
                              std::cerr << " Negative volume fraction " + op << std::endl;
                              save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                              exit(1);
                            }
                            // Sanity check for m1
                            if(conserved_variables[cell][M1_INDEX] < 0.0) {
                              std::cerr << "Negative mass for phase 1 " + op << std::endl;
                              save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                              exit(1);
                            }
                            // Sanity check for m2
                            if(conserved_variables[cell][M2_INDEX] < 0.0) {
                              std::cerr << "Negative mass for phase 2 " + op << std::endl;
                              save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                              exit(1);
                            }
                          });
}
#endif

// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void DamBreak<dim>::apply_relaxation() {
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
                             try {
                               #ifdef RUSANOV_FLUX
                                 Rusanov_flux.perform_Newton_step_relaxation(std::make_unique<decltype(conserved_variables[cell])>(conserved_variables[cell]),
                                                                             dalpha1[cell], alpha1[cell], rho[cell], relaxation_applied, tol, lambda);
                               #elifdef HLL_FLUX
                                 HLL_flux.perform_Newton_step_relaxation(std::make_unique<decltype(conserved_variables[cell])>(conserved_variables[cell]),
                                                                         dalpha1[cell], alpha1[cell], rho[cell], relaxation_applied, tol, lambda);
                               #elifdef HLLC_FLUX
                                 HLLC_flux.perform_Newton_step_relaxation(std::make_unique<decltype(conserved_variables[cell])>(conserved_variables[cell]),
                                                                          dalpha1[cell], alpha1[cell], rho[cell], relaxation_applied, tol, lambda);
                               #elifdef GODUNOV_FLUX
                                 Godunov_flux.perform_Newton_step_relaxation(std::make_unique<decltype(conserved_variables[cell])>(conserved_variables[cell]),
                                                                             dalpha1[cell], alpha1[cell], rho[cell], relaxation_applied, tol, lambda);
                               #endif
                             }
                             catch(std::exception& e) {
                               std::cerr << e.what() << std::endl;
                               save(fs::current_path(), "_diverged", conserved_variables, alpha1);
                             }

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
void DamBreak<dim>::save(const fs::path& path,
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
void DamBreak<dim>::run() {
  // Default output arguemnts
  fs::path path = fs::current_path();
  filename = "dam_break";
  #ifdef RUSANOV_FLUX
    filename += "_Rusanov";
  #elifdef HLL_FLUX
    filename += "_HLL";
  #elifdef HLLC_FLUX
    filename += "_HLLC";
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
  #elifdef HLL_FLUX
    auto numerical_flux = HLL_flux.make_flux();
  #elifdef HLLC_FLUX
    auto numerical_flux = HLLC_flux.make_flux();
  #elifdef GODUNOV_FLUX
    auto numerical_flux = Godunov_flux.make_flux();
  #endif

  // Create the gravity term
  #ifdef GRAVITY_IMPLICIT
    auto id = samurai::make_identity<decltype(conserved_variables)>(); // Identity matrix for the purpose of implicitation
  #endif
  using cfg = samurai::LocalCellSchemeConfig<samurai::SchemeType::LinearHomogeneous,
                                             Field::size,
                                             decltype(conserved_variables)>;
  auto gravity = samurai::make_cell_based_scheme<cfg>();
  gravity.coefficients_func() = [](double)
                                {
                                  samurai::StencilCoeffs<cfg> coeffs;

                                  coeffs[0].fill(0.0);

                                  coeffs[0](RHO_U_INDEX + 1, M1_INDEX) = -9.81;
                                  coeffs[0](RHO_U_INDEX + 1, M2_INDEX) = -9.81;

                                  return coeffs;
                                };

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, suffix_init, conserved_variables, alpha1, rho, p1, p2, p, vel);

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
        #ifdef GRAVITY_IMPLICIT
          auto implicit_operator = id - dt*gravity;
          auto rhs = conserved_variables - dt*flux_conserved;
          samurai::petsc::solve(implicit_operator, conserved_variables_tmp, rhs);
        #else
          conserved_variables_tmp = conserved_variables - dt*flux_conserved + dt*gravity(conserved_variables);
        #endif
        std::swap(conserved_variables.array(), conserved_variables_tmp.array());
      #else
        conserved_variables_np1.resize();
        #ifdef GRAVITY_IMPLICIT
          auto implicit_operator = id - dt*gravity;
          auto rhs = conserved_variables - dt*flux_conserved;
          samurai::petsc::solve(implicit_operator, conserved_variables_np1, rhs);
        #else
          conserved_variables_np1 = conserved_variables - dt*flux_conserved + dt*gravity(conserved_variables);
        #endif
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
        #ifdef GRAVITY_IMPLICIT
          auto implicit_operator = id - dt*gravity;
          auto rhs = conserved_variables - dt*flux_conserved;
          samurai::petsc::solve(implicit_operator, conserved_variables_tmp_2, rhs);
        #else
          conserved_variables_tmp_2 = conserved_variables - dt*flux_conserved + dt*gravity(conserved_variables);
        #endif
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
      p1.resize();
      p2.resize();
      p.resize();
      vel.resize();
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

                               // Compute velocity field
                               for(std::size_t d = 0; d < dim; ++d) {
                                 vel[cell][d] = conserved_variables[cell][RHO_U_INDEX + d]/rho[cell];
                               }
                             });

      save(path, suffix, conserved_variables, alpha1, rho, p1, p2, p, vel);
    }
  }
}
