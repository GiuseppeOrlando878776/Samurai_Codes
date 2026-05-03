// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include "utilities.hpp"

#include <filesystem>
namespace fs = std::filesystem;

/**
  * Auxiliary struct to save post-processing data
  */
template<typename Number>
struct IntegralQuantities {
  Number H_lig = static_cast<Number>(0.0);
  Number m_l_int = static_cast<Number>(0.0);
  Number m_d_int = static_cast<Number>(0.0);
  Number alpha_l_int = static_cast<Number>(0.0);
  Number grad_alpha_l_int = static_cast<Number>(0.0);
  Number Sigma_d_int = static_cast<Number>(0.0);
  Number alpha_d_int = static_cast<Number>(0.0);
  Number grad_alpha_d_int = static_cast<Number>(0.0);
  Number grad_alpha_liq_int = static_cast<Number>(0.0);
  Number alpha_l_bar_int = static_cast<Number>(0.0);
  Number grad_alpha_l_bar_int = static_cast<Number>(0.0);
  Number Etot_int = static_cast<Number>(0.0);
};

/**
  * Auxiliary class to perform post-processing
  */
template<typename Number>
class PostprocessWriter {
public:
  /*--- Open the file in the constructor ---*/
  explicit PostprocessWriter(const fs::path& output_dir) {
    open_stream(Hlig, output_dir / "Hlig.dat");
    open_stream(m_l_integral, output_dir / "m_l_integral.dat");
    open_stream(m_d_integral, output_dir / "m_d_integral.dat");
    open_stream(alpha_l_integral, output_dir / "alpha_l_integral.dat");
    open_stream(grad_alpha_l_integral, output_dir / "grad_alpha_l_integral.dat");
    open_stream(Sigma_d_integral, output_dir / "Sigma_d_integral.dat");
    open_stream(alpha_d_integral, output_dir / "alpha_d_integral.dat");
    open_stream(grad_alpha_d_integral, output_dir / "grad_alpha_d_integral.dat");
    open_stream(grad_alpha_liq_integral, output_dir / "grad_alpha_liq_integral.dat");
    open_stream(alpha_l_bar_integral, output_dir / "alpha_l_bar_integral.dat");
    open_stream(grad_alpha_l_bar_integral, output_dir / "grad_alpha_l_bar_integral.dat");
    open_stream(Etot_integral, output_dir / "Etot_integral.dat");
  }

  /*--- Default destructor ---*/
  ~PostprocessWriter() = default; /*--- std::ofstream closes itself in its own destructor ---*/

  /*--- Delete copy constructors ---*/
  PostprocessWriter(const PostprocessWriter&)            = delete;
  PostprocessWriter& operator=(const PostprocessWriter&) = delete;

  /*--- Allow move constructors ---*/
  PostprocessWriter(PostprocessWriter&&)            = default;
  PostprocessWriter& operator=(PostprocessWriter&&) = default;

  /*--- Perform writing operation ---*/
  void write(const Number time, const IntegralQuantities<Number>& q) {
    Utilities::write_data(Hlig, time, q.H_lig);
    Utilities::write_data(m_l_integral, time, q.m_l_int);
    Utilities::write_data(m_d_integral, time, q.m_d_int);
    Utilities::write_data(alpha_l_integral, time, q.alpha_l_int);
    Utilities::write_data(grad_alpha_l_integral, time, q.grad_alpha_l_int);
    Utilities::write_data(Sigma_d_integral, time, q.Sigma_d_int);
    Utilities::write_data(alpha_d_integral, time, q.alpha_d_int);
    Utilities::write_data(grad_alpha_d_integral, time, q.grad_alpha_d_int);
    Utilities::write_data(grad_alpha_liq_integral, time, q.grad_alpha_liq_int);
    Utilities::write_data(alpha_l_bar_integral, time, q.alpha_l_bar_int);
    Utilities::write_data(grad_alpha_l_bar_integral, time, q.grad_alpha_l_bar_int);
    Utilities::write_data(Etot_integral, time, q.Etot_int);
  }

private:
  /*--- Auxiliary output streams for post-processing ---*/
  std::ofstream Hlig;
  std::ofstream m_l_integral;
  std::ofstream m_d_integral;
  std::ofstream alpha_l_integral;
  std::ofstream grad_alpha_l_integral;
  std::ofstream Sigma_d_integral;
  std::ofstream alpha_d_integral;
  std::ofstream grad_alpha_d_integral;
  std::ofstream grad_alpha_liq_integral;
  std::ofstream alpha_l_bar_integral;
  std::ofstream grad_alpha_l_bar_integral;
  std::ofstream Etot_integral;

  /*--- Open stream ---*/
  static void open_stream(std::ofstream& stream, const fs::path& path) {
    stream.open(path);
    if(!stream.is_open()) {
      throw std::runtime_error("Cannot open output file: " + path.string());
    }
  }
};
