#ifndef containers_hpp
#define containers_hpp

// Declare a struct with the simulation parameters
// (domain, levels, final time, and Courant number)
struct Simulation_Paramaters {
  double xL;
  double xR;
  std::size_t min_level;
  std::size_t max_level;

  double Tf;
  double Courant;

  std::size_t nfiles;
};

// Declare a struct with EOS parameters
struct EOS_Parameters {
  double gamma_1;
  double pi_infty_1;
  double q_infty_1;
  
  double gamma_2;
  double pi_infty_2;
  double q_infty_2;
};

// Declare a struct with Riemann problem parameters
struct Riemann_Parameters {
  double xd;

  double alpha1L;
  double rho1L;
  double p1L;
  double vel1L;
  double rho2L;
  double p2L;
  double vel2L;

  double alpha1R;
  double rho1R;
  double p1R;
  double vel1R;
  double rho2R;
  double p2R;
  double vel2R;
};

#endif
