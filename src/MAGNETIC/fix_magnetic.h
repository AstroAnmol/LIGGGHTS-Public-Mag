/*
 * fix_magnetic.h
 *
 *      Author: ASikka
 */

#ifdef FIX_CLASS

FixStyle(magnetic,FixMagnetic)

#else

#ifndef LMP_FIX_MAGNETIC_H
#define LMP_FIX_MAGNETIC_H

#include "fix.h"
#include <Eigen/Dense>

namespace LAMMPS_NS {

class FixMagnetic : public Fix {
 public:
  FixMagnetic(class LAMMPS *, int, char **);
  ~FixMagnetic();
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void setup(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  double memory_usage();
  bigint N_magforce_timestep; // wait these many timesteps before computing force again
  std::string model_type; //Store the model type
  std::string moment_calc; //Store the moment calculation type

 private:
  double ex,ey,ez;
  int varflag;
  char *xstr,*ystr,*zstr;
  int xvar,yvar,zvar,xstyle,ystyle,zstyle;
  int nlevels_respa;
  class NeighList *list;
  int maxatom;

  class FixPropertyGlobal* susceptibility;

  bool warnflag;
  
  /* ----------------------------------------------------------------
  variables and functions needed for mag force calculation
  ----------------------------------------------------------------- */ 

  // variables
  double P4 = M_PI*4;
  double MU0 = P4*1e-7;

  //functions
  Eigen::Matrix3d Mom_Mat_ij(double sep_ij, Eigen::Vector3d SEP_ij_vec);
  void compute_magForce_linalg();
  void compute_magForce_converge();

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix efield requires atom attribute q

Self-explanatory.

E: Variable name for fix efield does not exist

Self-explanatory.

E: Variable for fix efield is invalid style

Only equal-style variables can be used.

*/
