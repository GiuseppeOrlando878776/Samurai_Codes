// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

/**
 * Implementation of a generic lcass to handle the EOS. It has several
   pure virtual functions to be implementede for the specific EOS
 */
template<typename T = double>
class EOS {
public:
  static_assert(std::is_arithmetic_v<T>, "Template argument EOS not well suited for arithemtic operations");

  /**
   * Default constructor
   */
  EOS() = default;

  /**
   * Default copy-constructor
   */
  EOS(const EOS&) = default;

  /**
   * Virtual destructor (it can be useful since we work through the base class)
   */
  virtual ~EOS() {}

  /**
   * Function to compute the pressure from density and internal energy
   * @param rho density value
   * @param e internal energy value
   */
  inline virtual T pres_value_Rhoe(const T rho, const T e) const = 0;

  /**
   * Function to compute the density from pressure and internal energy
   * @param pres pressure value
   * @param e internal energy value
   */
  inline virtual T rho_value_Pe(const T pres, const T e) const = 0;

  /**
   * Function to compute the internal energy from density and pressure
   * @param rho density value
   * @param pres pressure value
   */
  inline virtual T e_value_RhoP(const T rho, const T pres) const = 0;

  /**
   * Function to compute the speed of sound from density and pressure
   * @param rho density value
   * @param pres pressure value
   */
  inline virtual T c_value_RhoP(const T rho, const T pres) const = 0;

  /**
   * Function to compute the temperature from density and pressure
   * @param rho density value
   * @param pres pressure value
   */
  inline virtual T T_value_RhoP(const T rho, const T pres) const = 0;

  /**
   * Function to compute the Grüneisen coefficient from density and internal energy
   * @param rho density value
   * @param e internal energy value
   */
  inline virtual T Gruneisen_Rhoe(const T rho, const T e) const = 0;
};


/**
 * Implementation of the stiffened gas equation of state (SG-EOS)
 */
template<typename T = double>
class SG_EOS: public EOS<T> {
public:
  /**
   * Default constructor
   */
  SG_EOS() = default;

  /**
   * Default copy-constructor
   */
  SG_EOS(const SG_EOS&) = default;

  /**
   * Class constructor
   * @param gamma isentropic exponent
   * @param pi_infty pressure at 'infinity'
   * @param q_infty internal energy at 'infinity'
   * @param c_v specific heat at constant volume
   */
  SG_EOS(const T gamma_,
         const T pi_infty_ = static_cast<T>(0.0),
         const T q_infty_ = static_cast<T>(0.0),
         const T c_v_ = static_cast<T>(1.0));

  /**
   * Function to compute the pressure from density and internal energy
   * @param rho density value
   * @param e internal energy value
   */
  inline virtual T pres_value_Rhoe(const T rho, const T e) const override;

  /**
   * Function to compute the density from pressure and internal energy
   * @param pres pressure value
   * @param e internal energy value
   */
  inline virtual T rho_value_Pe(const T pres, const T e) const override;

  /**
   * Function to compute the internal energy from density and pressure
   * @param rho density value
   * @param pres pressure value
   */
  inline virtual T e_value_RhoP(const T rho, const T pres) const override;

  /**
   * Function to compute the speed of sound from density and pressure
   * @param rho density value
   * @param pres pressure value
   */
  inline virtual T c_value_RhoP(const T rho, const T pres) const override;

  /**
   * Function to compute the temperature from density and pressure
   * @param rho density value
   * @param pres pressure value
   */
  inline virtual T T_value_RhoP(const T rho, const T pres) const override;

  /**
   * Function to compute the Grüneisen coefficient from density and internal energy
   * @param rho density value
   * @param e internal energy value
   */
  inline virtual T Gruneisen_Rhoe(const T rho, const T e) const override;

  /**
   * Auxiliary function to return parameter gamma of EOS
   */
  inline T get_gamma() const;

  /**
   * Auxiliary function to return parameter pi_infty of EOS
   */
  inline T get_pi_infty() const;

  /**
   * Auxiliary function to return parameter q_infty of EOS
   */
  inline T get_q_infty() const; /*--- Auxiliary function to return parameter q_infty of EOS ---*/

  /**
   * Auxiliary function to return parameter c_v of EOS
   */
  inline T get_c_v() const;

private:
  const T gamma;    /*!< Isentropic exponent */
  const T pi_infty; /*!< Pressure at 'infinite' */
  const T q_infty;  /*!< Internal energy at 'infinite' */
  const T c_v;      /*!< Specific heat at constant volume */
};

// Implement the constructor
//
template<typename T>
SG_EOS<T>::SG_EOS(const T gamma_, const T pi_infty_,
                  const T q_infty_, const T c_v_):
  EOS<T>(), gamma(gamma_), pi_infty(pi_infty_), q_infty(q_infty_), c_v(c_v_) {}

// Compute the pressure value from the density and the internal energy
//
template<typename T>
inline T SG_EOS<T>::pres_value_Rhoe(const T rho, const T e) const {
  return (gamma - static_cast<T>(1.0))*rho*(e - q_infty) - gamma*pi_infty;
}

// Compute the density from the pressure and the internal energy
//
template<typename T>
inline T SG_EOS<T>::rho_value_Pe(const T pres, const T e) const {
  return (pres + gamma*pi_infty)/((gamma - static_cast<T>(1.0))*(e - q_infty));
}

// Compute the internal energy from density and pressure
//
template<typename T>
inline T SG_EOS<T>::e_value_RhoP(const T rho, const T pres) const {
  return (pres + gamma*pi_infty)/((gamma - static_cast<T>(1.0))*rho) + q_infty;
}

// Compute the speed of sound from density and pressure
//
template<typename T>
inline T SG_EOS<T>::c_value_RhoP(const T rho, const T pres) const {
  return std::sqrt(gamma*(pres + pi_infty)/rho);
}

// Compute the temperature from density and pressure
//
template<typename T>
inline T SG_EOS<T>::T_value_RhoP(const T rho, const T pres) const {
  return (pres + pi_infty)/((gamma - static_cast<T>(1.0))*rho*c_v);
}

// Compute the Grüneisen coefficient from density and internal energy
//
template<typename T>
inline T SG_EOS<T>::Gruneisen_Rhoe(const T rho, const T e) const {
  (void)rho;
  (void)e;
  return gamma - static_cast<T>(1.0);
}

// Auxiliary function to retrive gamma of SG-EOS
//
template<typename T>
inline T SG_EOS<T>::get_gamma() const {
  return gamma;
}

// Auxiliary function to retrive pi_infty of SG-EOS
//
template<typename T>
inline T SG_EOS<T>::get_pi_infty() const {
  return pi_infty;
}

// Auxiliary function to retrive q_infty of SG-EOS
//
template<typename T>
inline T SG_EOS<T>::get_q_infty() const {
  return q_infty;
}

// Auxiliary function to retrive c_v of SG-EOS
//
template<typename T>
inline T SG_EOS<T>::get_c_v() const {
  return c_v;
}
