/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(mist,FixMIST)

#else

#ifndef LMP_FIX_MIST_H
#define LMP_FIX_MIST_H

#include "fix.h"

namespace LAMMPS_NS {

class FixMIST : public Fix {
 public:
  FixMIST(class LAMMPS *, int, char **);
  ~FixMIST();
  int setmask();

  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);

};

}

#endif
#endif
