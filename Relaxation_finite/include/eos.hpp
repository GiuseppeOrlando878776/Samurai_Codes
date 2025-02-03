#ifndef eos_hpp
#define eos_hpp

/**
  * Implementation of a generic lcass to handle the EOS. It has several
    pure virtual functions to be implementede for the specific EOS
  */
template<typename T = double>
class EOS {
public:
  static_assert(std::is_arithmetic_v<T>, "Template argument EOS not well suited for arithemtic operations");

  EOS() = default; // Default constructor

  EOS(const EOS&) = default; // Default copy-constructor

  virtual ~EOS() {} // Virtual destructor (it can be useful since we work thourgh the base class)

  virtual T pres_value_Rhoe(const T rho, const T e) const = 0; // Function to compute the pressure from the density and the internal energy

  virtual T rho_value_Pe(const T pres, const T e) const = 0; // Function to compute the density from the pressure and the internal energy

  virtual T e_value_RhoP(const T rho, const T pres) const = 0; // Function to compute the internal energy from density and pressure

  virtual T T_value_Rhoe(const T rho, const T e) const = 0; // Function to compute the temperature from density and internal energy

  virtual T T_value_RhoP(const T rho, const T pres) const = 0; // Function to compute the temperature from density and pressure

  virtual T rho_value_PT(const T pres, const T temp) const = 0; // Function to compute the density from pressure and temperature

  virtual T e_value_PT(const T pres, const T temp) const = 0; // Function to compute the internal energy from pressure and temperature

  virtual T c_value_RhoP(const T rho, const T pres) const = 0; // Function to compute the speed of sound from density and pressure

  virtual T de_drho_T(const T rho, const T temp) const = 0; // Function to compute the derivative of the internal energy w.r.t. density (fixed temperature)

  virtual T de_dT_rho(const T temp, const T rho) const = 0; // Function to compute the derivative of the internal energy w.r.t. temperature (fixed density)

  virtual T de_dT_P(const T temp, const T pres) const = 0; // Function to compute the derivative of the internal energy w.r.t. temperature (fixed pressure)

  virtual T de_dP_T(const T pres, const T temp) const = 0; // Function to compute the derivative of the internal energy w.r.t. pressure (fixed temperature)

  virtual T de_dP_rho(const T pres, const T rho) const = 0; // Function to compute the derivative of the internal energy w.r.t. pressure (fixed density)

  virtual T drho_dP_T(const T pres, const T temp) const = 0; // Function to compute the derivative of the density w.r.t. pressure (fixed temperature)

  virtual T drho_dT_P(const T temp, const T pres) const = 0; // Function to compute the derivative of the density w.r.t. temperature (fixed pressure)
};


/**
 * Implementation of the stiffened gas equation of state (SG-EOS)
 */
template<typename T = double>
class SG_EOS: public EOS<T> {
public:
  SG_EOS() = default; // Default constructor

  SG_EOS(const SG_EOS&) = default; // Default copy-constructor

  SG_EOS(const T gamma_,
         const T pi_infty_ = 0.0,
         const T q_infty_ = 0.0,
         const T c_v_ = 1.0);      // Constructor which accepts as arguments
                                   // the isentropic exponent and the three parameters
                                   // that characterize the fluid

  virtual T pres_value_Rhoe(const T rho, const T e) const override; // Function to compute the pressure from the density and the internal energy

  virtual T rho_value_Pe(const T pres, const T e) const override; // Function to compute the density from the pressure and the internal energy

  virtual T e_value_RhoP(const T rho, const T pres) const override; // Function to compute the internal energy from density and pressure

  virtual T T_value_Rhoe(const T rho, const T e) const override; // Function to compute the temperature from density and internal energy

  virtual T T_value_RhoP(const T rho, const T pres) const override; // Function to compute the temperature from density and pressure

  virtual T rho_value_PT(const T pres, const T temp) const override; // Function to compute the density from pressure and temperature

  virtual T e_value_PT(const T pres, const T temp) const override; // Function to compute the internal energy from pressure and temperature

  virtual T c_value_RhoP(const T rho, const T pres) const override; // Function to compute the speed of sound from density and pressure

  virtual T de_drho_T(const T rho, const T temp) const override; // Function to compute the derivative of the internal energy w.r.t. density (fixed temperature)

  virtual T de_dT_rho(const T temp, const T rho) const override; // Function to compute the derivative of the internal energy w.r.t. temperature (fixed density)

  virtual T de_dT_P(const T temp, const T pres) const override; // Function to compute the derivative of the internal energy w.r.t. temperature (fixed pressure)

  virtual T de_dP_T(const T pres, const T temp) const override; // Function to compute the derivative of the internal energy w.r.t. pressure (fixed temperature)

  virtual T de_dP_rho(const T pres, const T rho) const override; // Function to compute the derivative of the internal energy w.r.t. pressure (fixed density)

  virtual T drho_dP_T(const T pres, const T temp) const override; // Function to compute the derivative of the density w.r.t. pressure (fixed temperature)

  virtual T drho_dT_P(const T temp, const T pres) const override; // Function to compute the derivative of the density w.r.t. temperature (fixed pressure)

  inline T get_gamma() const; // Return the isentropic exponent

  inline T get_pi_infty() const; // Return the pressure at 'infinite'

  inline T get_q_infty() const; // Return the internal energy at 'infinite'

  inline T get_cv() const; // Return the specific heat at constant volume

private:
  const T gamma;    // Isentropic exponent
  const T pi_infty; // Pressure at 'infinite'
  const T q_infty;  // Internal energy at 'infinite'
  const T c_v;      // Specific heat at constant volume
};

// Implement the constructor
//
template<typename T>
SG_EOS<T>::SG_EOS(const T gamma_, const T pi_infty_, const T q_infty_, const T c_v_):
  EOS<T>(), gamma(gamma_), pi_infty(pi_infty_), q_infty(q_infty_), c_v(c_v_) {}

// Compute the pressure value from the density and the internal energy
//
template<typename T>
T SG_EOS<T>::pres_value_Rhoe(const T rho, const T e) const {
  return (gamma - 1.0)*rho*(e - q_infty) - gamma*pi_infty;
}

// Compute the density from the pressure and the internal energy
//
template<typename T>
T SG_EOS<T>::rho_value_Pe(const T pres, const T e) const {
  return (pres + gamma*pi_infty)/((gamma - 1.0)*(e - q_infty));
}

// Compute the internal energy from density and pressure
//
template<typename T>
T SG_EOS<T>::e_value_RhoP(const T rho, const T pres) const {
  return (pres + gamma*pi_infty)/((gamma - 1.0)*rho) + q_infty;
}

// Compute the temperature from density and internal energy
//
template<typename T>
T SG_EOS<T>::T_value_Rhoe(const T rho, const T e) const {
  return (e - q_infty - pi_infty/rho)/c_v;
}

// Compute the temperature from density and pressure
//
template<typename T>
T SG_EOS<T>::T_value_RhoP(const T rho, const T pres) const {
  return (pres + pi_infty)/((gamma - 1.0)*rho*c_v);
}

// Compute the density from pressure and temeperature
//
template<typename T>
T SG_EOS<T>::rho_value_PT(const T pres, const T temp) const {
  return (pres + pi_infty)/((gamma - 1.0)*c_v*temp);
}

// Compute the internal energy from pressure and temeperature
//
template<typename T>
T SG_EOS<T>::e_value_PT(const T pres, const T temp) const {
  return ((pres + gamma*pi_infty)/(pres + pi_infty))*c_v*temp + q_infty;
}

// Compute the speed of sound from density and pressure
//
template<typename T>
T SG_EOS<T>::c_value_RhoP(const T rho, const T pres) const {
  return std::sqrt(gamma*(pres + pi_infty)/rho);
}

// Compute the derivative of the internal energy w.r.t. density (fixed temperature)
//
template<typename T>
T SG_EOS<T>::de_drho_T(const T rho, const T temp) const {
  (void) temp;

  return -pi_infty/(rho*rho);
}

// Compute the derivative of the internal energy w.r.t. temperature (fixed density)
//
template<typename T>
T SG_EOS<T>::de_dT_rho(const T temp, const T rho) const {
  (void) temp;
  (void) rho;

  return c_v;
}

// Compute the derivative of the internal energy w.r.t. temperature (fixed pressure)
//
template<typename T>
T SG_EOS<T>::de_dT_P(const T temp, const T pres) const {
  (void) temp;

  return ((pres + gamma*pi_infty)/(pres + pi_infty))*c_v;
}

// Compute the derivative of the internal energy w.r.t. pressure (fixed temperature)
//
template<typename T>
T SG_EOS<T>::de_dP_T(const T pres, const T temp) const {
  return ((1.0 - gamma)*pi_infty)/((pres + pi_infty)*(pres + pi_infty))*c_v*temp;
}

// Compute the derivative of the internal energy w.r.t. pressure (fixed density)
//
template<typename T>
T SG_EOS<T>::de_dP_rho(const T pres, const T rho) const {
  (void) pres;

  return 1.0/((gamma - 1.0)*rho);
}

// Compute the derivative of the density w.r.t. pressure (fixed temperature)
//
template<typename T>
T SG_EOS<T>::drho_dP_T(const T pres, const T temp) const {
  (void) pres;

  return 1.0/((gamma - 1.0)*c_v*temp);
}

// Compute the derivative of the density w.r.t. temperature (fixed pressure)
//
template<typename T>
T SG_EOS<T>::drho_dT_P(const T temp, const T pres) const {
  return -(pres + pi_infty)/((gamma - 1.0)*c_v*temp*temp);
}

// Return the isentropic exponent
//
template<typename T>
inline T SG_EOS<T>::get_gamma() const {
  return gamma;
}

// Return the pressure at 'infinite'
//
template<typename T>
inline T SG_EOS<T>::get_pi_infty() const {
  return pi_infty;
}

// Return the internal energu at 'infinite'
//
template<typename T>
inline T SG_EOS<T>::get_q_infty() const {
  return q_infty;
}

// Return the specific heat at constant volume
//
template<typename T>
inline T SG_EOS<T>::get_cv() const {
  return c_v;
}

#endif
