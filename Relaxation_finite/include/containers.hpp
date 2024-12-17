#ifndef containers_hpp
#define containers_hpp

// Declare a struct with the simulation parameters
// (domain, levels, final time, and Courant number)
struct Simulation_Parameters {
  double xL;
  double xR;
  double yL;
  double yR;
  std::size_t min_level;
  std::size_t max_level;

  double Tf;
  double Courant;

  std::size_t nfiles;

  bool apply_relaxation;

  bool apply_finite_rate_relaxation;
  double eps_u;
  double eps_p;
  double eps_T;

  bool relax_pressure;
  bool relax_temperature;
};

// Declare a struct with EOS parameters
struct EOS_Parameters {
  double gamma_1;
  double pi_infty_1;
  double q_infty_1;
  double cv_1;

  double gamma_2;
  double pi_infty_2;
  double q_infty_2;
  double cv_2;
};

// Declare a struct with Riemann problem parameters
struct Riemann_Parameters {
  double xd;

  double alpha1L;
  double rho1L;
  double p1L;
  double u1L;
  double v1L;
  double rho2L;
  double p2L;
  double u2L;
  double v2L;

  double alpha1R;
  double rho1R;
  double p1R;
  double u1R;
  double v1R;
  double rho2R;
  double p2R;
  double u2R;
  double v2R;
};

#endif
