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
  Number m1_int = static_cast<Number>(0.0);
  Number m1_d_int = static_cast<Number>(0.0);
  Number alpha1_bar_int = static_cast<Number>(0.0);
  Number grad_alpha1_bar_int = static_cast<Number>(0.0);
  Number Sigma_d_int = static_cast<Number>(0.0);
  Number alpha1_d_int = static_cast<Number>(0.0);
  Number grad_alpha1_d_int = static_cast<Number>(0.0);
  Number grad_alpha1_int = static_cast<Number>(0.0);
  Number grad_alpha1_tot_int = static_cast<Number>(0.0);
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
    open_stream(m1_integral, output_dir / "m1_integral.dat");
    open_stream(m1_d_integral, output_dir / "m1_d_integral.dat");
    open_stream(alpha1_bar_integral, output_dir / "alpha1_bar_integral.dat");
    open_stream(grad_alpha1_bar_integral, output_dir / "grad_alpha1_bar_integral.dat");
    open_stream(Sigma_d_integral, output_dir / "Sigma_d_integral.dat");
    open_stream(alpha1_d_integral, output_dir / "alpha1_d_integral.dat");
    open_stream(grad_alpha1_d_integral, output_dir / "grad_alpha1_d_integral.dat");
    open_stream(grad_alpha1_integral, output_dir / "grad_alpha1_integral.dat");
    open_stream(grad_alpha1_tot_integral, output_dir / "grad_alpha1_tot_integral.dat");
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
    Utilities::write_data(m1_integral, time, q.m1_int);
    Utilities::write_data(m1_d_integral, time, q.m1_d_int);
    Utilities::write_data(alpha1_bar_integral, time, q.alpha1_bar_int);
    Utilities::write_data(grad_alpha1_bar_integral, time, q.grad_alpha1_bar_int);
    Utilities::write_data(Sigma_d_integral, time, q.Sigma_d_int);
    Utilities::write_data(alpha1_d_integral, time, q.alpha1_d_int);
    Utilities::write_data(grad_alpha1_d_integral, time, q.grad_alpha1_d_int);
    Utilities::write_data(grad_alpha1_integral, time, q.grad_alpha1_int);
    Utilities::write_data(grad_alpha1_tot_integral, time, q.grad_alpha1_tot_int);
  }

private:
  /*--- Auxiliary output streams for post-processing ---*/
  std::ofstream Hlig;
  std::ofstream m1_integral;
  std::ofstream m1_d_integral;
  std::ofstream alpha1_bar_integral;
  std::ofstream grad_alpha1_bar_integral;
  std::ofstream Sigma_d_integral;
  std::ofstream alpha1_d_integral;
  std::ofstream grad_alpha1_d_integral;
  std::ofstream grad_alpha1_integral;
  std::ofstream grad_alpha1_tot_integral;

  /*--- Open stream ---*/
  static void open_stream(std::ofstream& stream, const fs::path& path) {
    stream.open(path);
    if(!stream.is_open()) {
      throw std::runtime_error("Cannot open output file: " + path.string());
    }
  }
};
