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

#include <filesystem>
namespace fs = std::filesystem;

// Add header file for the multiresolution
#include <samurai/mr/adapt.hpp>

#define HLL_FLUX
//#define HLLC_FLUX
//#define RUSANOV_FLUX

#ifdef HLLC_FLUX
  #include "HLLC_flux.hpp"
#elifdef HLL_FLUX
  #include "HLL_flux.hpp"
#elifdef RUSANOV_FLUX
  #include "Rusanov_flux.hpp"
#endif

#include "containers.hpp"

// Define preprocessor to check whether to control data or not
#define VERBOSE

#define assertm(exp, msg) assert(((void)msg, exp))

// Specify the use of this namespace where we just store the indices
// and some parameters related to the equations of state
using namespace EquationData;

// This is the class for the simulation for the Euler equations
//
template<std::size_t dim>
class Euler_MR {
public:
  using Config = samurai::MRConfig<dim, 2>;

  Euler_MR() = default; // Default constructor. This will do nothing
                        // and basically will never be used

  Euler_MR(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
           const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
           const Simulation_Paramaters& sim_param,
           const EOS_Parameters& eos_param,
           const Riemann_Parameters& Riemann_param); // Class constrcutor with the arguments related
                                                     // to the grid and to the physics

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

  double Tf;  // Final time of the simulation
  double cfl; // Courant number of the simulation so as to compute the time step

  std::size_t nfiles; // Number of files desired for output

  Field conserved_variables; // The variable which stores the conserved variables,
                             // namely the varialbes for which we solve a PDE system

  const SG_EOS<typename Field::value_type> Euler_EOS; // Equation of state

  #ifdef RUSANOV_FLUX
    samurai::RusanovFlux<Field> numerical_flux; // variable to compute the numerical flux
                                                // (this is necessary to call 'make_flux')
  #elifdef HLLC_FLUX
    samurai::HLLCFlux<Field> numerical_flux; // variable to compute the numerical flux
                                             // (this is necessary to call 'make_flux')
  #elifdef HLL_FLUX
    samurai::HLLFlux<Field> numerical_flux; // variable to compute the numerical flux
                                            // (this is necessary to call 'make_flux')
  #endif

  std::string filename; // Auxiliary variable to store the name of output

  const double MR_param; // Multiresolution parameter
  const double MR_regularity; // Multiresolution regularity

  /*--- Now we declare a bunch of fields which depend from the state,
        but it is useful to have it for the output ---*/
  Field_Scalar p,
               c;

  Field_Vect vel;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void init_variables(const Riemann_Parameters& Riemann_param); // Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)

  void perform_mesh_adaptation(); // Perform the mesh adaptation

  void update_auxiliary_fields(); // Routine to update auxiliary fields for output and time step update

  double get_max_lambda() const; // Compute the estimate of the maximum eigenvalue

  void check_data(unsigned int flag = 0); // Auxiliary routine to check if small negative values are present
};

// Implement class constructor
//
template<std::size_t dim>
Euler_MR<dim>::Euler_MR(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                        const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                        const Simulation_Paramaters& sim_param,
                        const EOS_Parameters& eos_param,
                        const Riemann_Parameters& Riemann_param):
  box(min_corner, max_corner), mesh(box, sim_param.min_level, sim_param.max_level, {false}),
  Tf(sim_param.Tf), cfl(sim_param.Courant), nfiles(sim_param.nfiles),
  Euler_EOS(eos_param.gamma, eos_param.pi_infty, eos_param.q_infty),
  numerical_flux(Euler_EOS),
  MR_param(sim_param.MR_param), MR_regularity(sim_param.MR_regularity)
  {
    std::cout << "Initializing variables" << std::endl;
    std::cout << std::endl;
    init_variables(Riemann_param);
  }

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void Euler_MR<dim>::init_variables(const Riemann_Parameters& Riemann_param) {
  /*--- Create conserved and auxiliary fields ---*/
  conserved_variables = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved", mesh);

  p   = samurai::make_field<typename Field::value_type, 1>("p", mesh);
  c   = samurai::make_field<typename Field::value_type, 1>("c", mesh);
  vel = samurai::make_field<typename Field::value_type, dim>("vel", mesh);

  /*--- Set the initial state ---*/
  const double xd = Riemann_param.xd;

  // Initialize the fields with a loop over all cells
  const double velL = Riemann_param.uL;

  const double pL   = Riemann_param.pL;
  const double rhoL = Riemann_param.rhoL;

  const double velR = Riemann_param.uR;

  const double pR   = Riemann_param.pR;
  const double rhoR = Riemann_param.rhoR;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];

                           /*--- Left state (primitive variables) ---*/
                           if(x <= xd) {
                             conserved_variables[cell][RHO_INDEX] = rhoL;

                             vel[cell] = velL;
                             p[cell]   = pL;
                           }
                           /*--- Right state (primitive variables) ---*/
                           else {
                             conserved_variables[cell][RHO_INDEX] = rhoR;

                             vel[cell] = velR;
                             p[cell]   = pR;
                           }

                           /*--- Complete the conserved variables (and some auxiliary fields for the sake of completeness) ---*/
                           conserved_variables[cell][RHOU_INDEX] = conserved_variables[cell][RHO_INDEX]*vel[cell];

                           const auto e = Euler_EOS.e_value(conserved_variables[cell][RHO_INDEX], p[cell]);
                           conserved_variables[cell][RHOE_INDEX] = conserved_variables[cell][RHO_INDEX]*
                                                                   (e + 0.5*vel[cell]*vel[cell]);
                           c[cell] = Euler_EOS.c_value(conserved_variables[cell][RHO_INDEX], p[cell]);
                         });

  const xt::xtensor_fixed<int, xt::xshape<1>> left{-1};
  const xt::xtensor_fixed<int, xt::xshape<1>> right{1};
  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables,
                                          rhoL, rhoL*velL, rhoL*(Euler_EOS.e_value(rhoL, pL) + 0.5*velL*velL))->on(left);
  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables,
                                          rhoR, rhoR*velR, rhoR*(Euler_EOS.e_value(rhoR, pR) + 0.5*velR*velR))->on(right);
}

// Perform the mesh adaptation strategy.
//
template<std::size_t dim>
void Euler_MR<dim>::perform_mesh_adaptation() {
  save(fs::current_path(), "_before_mesh_adaptation",
       conserved_variables, vel, p, c);

  samurai::update_ghost_mr(c);
  auto MRadaptation = samurai::make_MRAdapt(c);
  MRadaptation(MR_param, MR_regularity, conserved_variables);

  #ifdef VERBOSE
    check_data(1);
  #endif
}

// Auxiliary routine to check if negative values arise
//
#ifdef VERBOSE
template<std::size_t dim>
void Euler_MR<dim>::check_data(unsigned int flag) {
  // Recompute data so as to save the whole state in case of diverging solution
  update_auxiliary_fields();

  // Check data
  std::string op;
  if(flag == 0) {
    op = "after hyperbolic operator";
  }
  else {
    op = "after mesh adaptation";
  }
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Sanity check for the density
                           if(conserved_variables[cell][RHO_INDEX] < 0.0) {
                             std::cerr << "Negative density " + op << std::endl;
                             save(fs::current_path(), "_diverged",
                                  conserved_variables, vel, p, c);
                             exit(1);
                           }

                           // Sanity check for the pressure
                           if(p[cell] < 0.0) {
                             std::cerr << "Negative pressure " + op << std::endl;
                             save(fs::current_path(), "_diverged",
                                  conserved_variables, vel, p, c);
                             exit(1);
                           }
                         });
}
#endif

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double Euler_MR<dim>::get_max_lambda() const {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           res = std::max(std::abs(vel[cell]) + c[cell], res);
                         });

  return res;
}

// Update auxiliary fields after solution of the system
//
template<std::size_t dim>
void Euler_MR<dim>::update_auxiliary_fields() {
  vel.resize();
  p.resize();
  c.resize();

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           vel[cell] = conserved_variables[cell][RHOU_INDEX]/conserved_variables[cell][RHO_INDEX];
                           auto e = conserved_variables[cell][RHOE_INDEX]/conserved_variables[cell][RHO_INDEX];
                           e -= 0.5*vel[cell]*vel[cell];
                           p[cell] = Euler_EOS.pres_value(conserved_variables[cell][RHO_INDEX], e);
                           c[cell] = Euler_EOS.c_value(conserved_variables[cell][RHO_INDEX], p[cell]);
                         });
}

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void Euler_MR<dim>::save(const fs::path& path,
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
void Euler_MR<dim>::run() {
  /*--- Default output arguemnts ---*/
  fs::path path = fs::current_path();
  #ifdef RUSANOV_FLUX
    filename = "Euler_MR_Rusanov";
  #elifdef HLLC_FLUX
    filename = "Euler_MR_HLLC";
  #elifdef HLL_FLUX
    filename = "Euler_MR_HLL";
  #endif

  #ifdef ORDER_2
    filename = filename + "_order2";
  #else
    filename = filename + "_order1";
  #endif

  const double dt_save = Tf/static_cast<double>(nfiles);

  /*--- Auxiliary variables to save updated fields ---*/
  #ifdef ORDER_2
    auto conserved_variables_tmp   = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved_tmp", mesh);
    auto conserved_variables_tmp_2 = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved_tmp_2", mesh);
  #endif
  auto conserved_variables_np1 = samurai::make_field<typename Field::value_type, EquationData::NVARS>("conserved_np1", mesh);

  /*--- Create the flux variable ---*/
  auto Discrete_flux = numerical_flux.make_flux();

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, suffix_init,
       conserved_variables, p, vel, c);

  /*--- Save mesh size ---*/
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  const double dx = samurai::cell_length(mesh[mesh_id_t::cells].max_level());

  /*--- Start the loop ---*/
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  double t          = 0.0;
  double dt         = std::min(Tf - t, cfl*dx/get_max_lambda());
  while(t != Tf) {
    t += dt;

    std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

    // Apply mesh adaptation
    perform_mesh_adaptation();

    // Apply the numerical scheme
    samurai::update_ghost_mr(conserved_variables);
    try {
      auto Cons_Flux = Discrete_flux(conserved_variables);
      #ifdef ORDER_2
        conserved_variables_tmp.resize();
        conserved_variables_tmp = conserved_variables - dt*Cons_Flux;
        std::swap(conserved_variables.array(), conserved_variables_tmp.array());
      #else
        conserved_variables_np1.resize();
        conserved_variables_np1 = conserved_variables - dt*Cons_Flux;
        std::swap(conserved_variables.array(), conserved_variables_np1.array());
      #endif
    }
    catch(std::exception& e) {
      std::cerr << e.what() << std::endl;
      save(fs::current_path(), "_diverged",
           conserved_variables, vel, p, c);
      exit(1);
    }
    #ifdef VERBOSE
      check_data();
    #endif

    // Consider the second stage for the second order
    #ifdef ORDER_2
      // Apply the numerical scheme
      samurai::update_ghost_mr(conserved_variables);
      try {
        auto Cons_Flux = Discrete_flux(conserved_variables);
        conserved_variables_tmp_2 = conserved_variables - dt*Cons_Flux;
        conserved_variables_np1 = 0.5*(conserved_variables_tmp + conserved_variables_tmp_2);
        std::swap(conserved_variables.array(), conserved_variables_np1.array());
      }
      catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
        save(fs::current_path(), "_diverged",
             conserved_variables, vel, p, c);
        exit(1);
      }
      #ifdef VERBOSE
        check_data();
      #endif
    #endif

    // Compute updated time step
    #if !defined VERBOSE
      update_auxiliary_fields();
    #endif
    dt = std::min(Tf - t, cfl*dx/get_max_lambda());

    // Save the results
    if(t >= static_cast<double>(nsave + 1)*dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, suffix,
           conserved_variables, p, vel, c);
    }
  }
}
