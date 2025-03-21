/* ----------------------------------------------------------------------
    This is the

    ██╗     ██╗ ██████╗  ██████╗  ██████╗ ██╗  ██╗████████╗███████╗
    ██║     ██║██╔════╝ ██╔════╝ ██╔════╝ ██║  ██║╚══██╔══╝██╔════╝
    ██║     ██║██║  ███╗██║  ███╗██║  ███╗███████║   ██║   ███████╗
    ██║     ██║██║   ██║██║   ██║██║   ██║██╔══██║   ██║   ╚════██║
    ███████╗██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║   ██║   ███████║
    ╚══════╝╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝®

    DEM simulation engine, released by
    DCS Computing Gmbh, Linz, Austria
    http://www.dcs-computing.com, office@dcs-computing.com

    LIGGGHTS® is part of CFDEM®project:
    http://www.liggghts.com | http://www.cfdem.com

    Core developer and main author:
    Christoph Kloss, christoph.kloss@dcs-computing.com

    LIGGGHTS® is open-source, distributed under the terms of the GNU Public
    License, version 2 or later. It is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
    received a copy of the GNU General Public License along with LIGGGHTS®.
    If not, see http://www.gnu.org/licenses . See also top-level README
    and LICENSE files.

    LIGGGHTS® and CFDEM® are registered trade marks of DCS Computing GmbH,
    the producer of the LIGGGHTS® software and the CFDEM®coupling software
    See http://www.cfdem.com/terms-trademark-policy for details.

-------------------------------------------------------------------------
    Contributing author and copyright for this file:
    This file is from LAMMPS
    LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
    http://lammps.sandia.gov, Sandia National Laboratories
    Steve Plimpton, sjplimp@sandia.gov

    Copyright (2003) Sandia Corporation.  Under the terms of Contract
    DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
    certain rights in this software.  This software is distributed under
    the GNU General Public License.
------------------------------------------------------------------------- 
    Modified for the Magnetic Force Model by author:
    Anmol Sikka
    University of Maryland College Park
    anmolsikka09@gmail.com
------------------------------------------------------------------------- */

#include <cmath>
#include <string.h>
#include "compute_property_atom.h"
#include "math_extra.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "atom_vec_line.h"
#include "update.h"
#include "domain.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputePropertyAtom::ComputePropertyAtom(LAMMPS *lmp, int &iarg, int narg, char **arg) :
  Compute(lmp, iarg, narg, arg)
{
  if (narg < iarg+1) error->all(FLERR,"Illegal compute property/atom command");

  peratom_flag = 1;
  nvalues = narg - iarg;
  if (nvalues == 1) size_peratom_cols = 0;
  else size_peratom_cols = nvalues;

  // parse input values
  // customize a new keyword by adding to if statement

  pack_choice = new FnPtrPack[nvalues];
  index = new int[nvalues];

  int i;
  const int arg_offset = iarg;
  for (; iarg < narg; iarg++) {
    i = iarg-arg_offset;

    if (strcmp(arg[iarg],"id") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_id;
    } else if (strcmp(arg[iarg],"mol") == 0) {
      if (!atom->molecule_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_molecule;
    } else if (strcmp(arg[iarg],"type") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_type;
    } else if (strcmp(arg[iarg],"mass") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_mass;

    } else if (strcmp(arg[iarg],"volume") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_vol;
    } else if (strcmp(arg[iarg],"eqradius") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_eq_radius;
    } else if (strcmp(arg[iarg],"x") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_x;
    } else if (strcmp(arg[iarg],"y") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_y;
    } else if (strcmp(arg[iarg],"z") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_z;
    } else if (strcmp(arg[iarg],"xs") == 0) {
      if (domain->triclinic)
        pack_choice[i] = &ComputePropertyAtom::pack_xs_triclinic;
      else pack_choice[i] = &ComputePropertyAtom::pack_xs;
    } else if (strcmp(arg[iarg],"ys") == 0) {
      if (domain->triclinic)
        pack_choice[i] = &ComputePropertyAtom::pack_ys_triclinic;
      else pack_choice[i] = &ComputePropertyAtom::pack_ys;
    } else if (strcmp(arg[iarg],"zs") == 0) {
      if (domain->triclinic)
        pack_choice[i] = &ComputePropertyAtom::pack_zs_triclinic;
      else pack_choice[i] = &ComputePropertyAtom::pack_zs;
    } else if (strcmp(arg[iarg],"xu") == 0) {
      if (domain->triclinic)
        pack_choice[i] = &ComputePropertyAtom::pack_xu_triclinic;
      else pack_choice[i] = &ComputePropertyAtom::pack_xu;
    } else if (strcmp(arg[iarg],"yu") == 0) {
      if (domain->triclinic)
        pack_choice[i] = &ComputePropertyAtom::pack_yu_triclinic;
      else pack_choice[i] = &ComputePropertyAtom::pack_yu;
    } else if (strcmp(arg[iarg],"zu") == 0) {
      if (domain->triclinic)
        pack_choice[i] = &ComputePropertyAtom::pack_zu_triclinic;
      else pack_choice[i] = &ComputePropertyAtom::pack_zu;
    } else if (strcmp(arg[iarg],"ix") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_ix;
    } else if (strcmp(arg[iarg],"iy") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_iy;
    } else if (strcmp(arg[iarg],"iz") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_iz;

    } else if (strcmp(arg[iarg],"vx") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_vx;
    } else if (strcmp(arg[iarg],"vy") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_vy;
    } else if (strcmp(arg[iarg],"vz") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_vz;
    } else if (strcmp(arg[iarg],"fx") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_fx;
    } else if (strcmp(arg[iarg],"fy") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_fy;
    } else if (strcmp(arg[iarg],"fz") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_fz;

    } else if (strcmp(arg[iarg],"magfx") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_mag_fx;
    } else if (strcmp(arg[iarg],"magfy") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_mag_fy;
    } else if (strcmp(arg[iarg],"magfz") == 0) {
      pack_choice[i] = &ComputePropertyAtom::pack_mag_fz;

    } else if (strcmp(arg[iarg],"q") == 0) {
      if (!atom->q_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_q;
    } else if (strcmp(arg[iarg],"mux") == 0) {
      if (!atom->mu_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_mux;
    } else if (strcmp(arg[iarg],"muy") == 0) {
      if (!atom->mu_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_muy;
    } else if (strcmp(arg[iarg],"muz") == 0) {
      if (!atom->mu_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_muz;
    } else if (strcmp(arg[iarg],"mu") == 0) {
      if (!atom->mu_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_mu;

    } else if (strcmp(arg[iarg],"radius") == 0) {
      if (!atom->radius_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_radius;
    } else if (strcmp(arg[iarg],"diameter") == 0) {
      if (!atom->radius_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_diameter;
    } else if (strcmp(arg[iarg],"omegax") == 0) {
      if (!atom->omega_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_omegax;
    } else if (strcmp(arg[iarg],"omegay") == 0) {
      if (!atom->omega_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_omegay;
    } else if (strcmp(arg[iarg],"omegaz") == 0) {
      if (!atom->omega_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_omegaz;
    } else if (strcmp(arg[iarg],"angmomx") == 0) {
      if (!atom->angmom_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_angmomx;
    } else if (strcmp(arg[iarg],"angmomy") == 0) {
      if (!atom->angmom_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_angmomy;
    } else if (strcmp(arg[iarg],"angmomz") == 0) {
      if (!atom->angmom_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_angmomz;

    } else if (strcmp(arg[iarg],"shapex") == 0) {
      avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
      if (!avec_ellipsoid) error->all(FLERR,"Compute property/atom for "
                                      "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_shapex;
    } else if (strcmp(arg[iarg],"shapey") == 0) {
      avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
      if (!avec_ellipsoid) error->all(FLERR,"Compute property/atom for "
                                      "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_shapey;
    } else if (strcmp(arg[iarg],"shapez") == 0) {
      avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
      if (!avec_ellipsoid) error->all(FLERR,"Compute property/atom for "
                                      "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_shapez;

    } else if (strcmp(arg[iarg],"quatw") == 0) {
      avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
      avec_body = (AtomVecBody *) atom->style_match("body");
      if (!avec_ellipsoid && !avec_body)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_quatw;
    } else if (strcmp(arg[iarg],"quati") == 0) {
      avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
      avec_body = (AtomVecBody *) atom->style_match("body");
      if (!avec_ellipsoid && !avec_body)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_quati;
    } else if (strcmp(arg[iarg],"quatj") == 0) {
      avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
      avec_body = (AtomVecBody *) atom->style_match("body");
      if (!avec_ellipsoid && !avec_body)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_quatj;
    } else if (strcmp(arg[iarg],"quatk") == 0) {
      avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
      avec_body = (AtomVecBody *) atom->style_match("body");
      if (!avec_ellipsoid && !avec_body)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_quatk;

    } else if (strcmp(arg[iarg],"tqx") == 0) {
      if (!atom->torque_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_tqx;
    } else if (strcmp(arg[iarg],"tqy") == 0) {
      if (!atom->torque_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_tqy;
    } else if (strcmp(arg[iarg],"tqz") == 0) {
      if (!atom->torque_flag)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_tqz;
   } else if (strcmp(arg[iarg],"end1x") == 0) {
      avec_line = (AtomVecLine *) atom->style_match("line");
      if (!avec_line) error->all(FLERR,"Compute property/atom for "
                                 "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_end1x;
    } else if (strcmp(arg[iarg],"end1y") == 0) {
      avec_line = (AtomVecLine *) atom->style_match("line");
      if (!avec_line) error->all(FLERR,"Compute property/atom for "
                                 "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_end1y;
    } else if (strcmp(arg[iarg],"end1z") == 0) {
      avec_line = (AtomVecLine *) atom->style_match("line");
      if (!avec_line) error->all(FLERR,"Compute property/atom for "
                                 "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_end1z;
    } else if (strcmp(arg[iarg],"end2x") == 0) {
      avec_line = (AtomVecLine *) atom->style_match("line");
      if (!avec_line) error->all(FLERR,"Compute property/atom for "
                                 "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_end2x;
    } else if (strcmp(arg[iarg],"end2y") == 0) {
      avec_line = (AtomVecLine *) atom->style_match("line");
      if (!avec_line) error->all(FLERR,"Compute property/atom for "
                                 "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_end2y;
    } else if (strcmp(arg[iarg],"end2z") == 0) {
      avec_line = (AtomVecLine *) atom->style_match("line");
      if (!avec_line) error->all(FLERR,"Compute property/atom for "
                                 "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertyAtom::pack_end2z;
    } else if (strstr(arg[iarg],"i_") == arg[iarg]) {
      int flag;
      index[i] = atom->find_custom(&arg[iarg][2],flag);
      if (index[i] < 0 || flag != 0)
        error->all(FLERR,"Compute property/atom floating point "
                   "vector does not exist");
      pack_choice[i] = &ComputePropertyAtom::pack_iname;
    } else if (strstr(arg[iarg],"d_") == arg[iarg]) {
      int flag;
      index[i] = atom->find_custom(&arg[iarg][2],flag);
      if (index[i] < 0 || flag != 1)
        error->all(FLERR,"Compute property/atom integer "
                   "vector does not exist");
      pack_choice[i] = &ComputePropertyAtom::pack_dname;

    } else error->all(FLERR,"Invalid keyword in compute property/atom command");
  }

  nmax = 0;
  vector = NULL;
  array = NULL;
}

/* ---------------------------------------------------------------------- */

ComputePropertyAtom::~ComputePropertyAtom()
{
  delete [] pack_choice;
  delete [] index;
  memory->destroy(vector);
  memory->destroy(array);
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::init()
{
  avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  avec_line = (AtomVecLine *) atom->style_match("line");
  avec_tri = (AtomVecTri *) atom->style_match("tri");
  avec_body = (AtomVecBody *) atom->style_match("body");
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow vector or array if necessary

  if (atom->nlocal > nmax) {
    nmax = atom->nmax;
    if (nvalues == 1) {
      memory->destroy(vector);
      memory->create(vector,nmax,"property/atom:vector");
      vector_atom = vector;
    } else {
      memory->destroy(array);
      memory->create(array,nmax,nvalues,"property/atom:array");
      array_atom = array;
    }
  }

  // fill vector or array with per-atom values

  if (nvalues == 1) {
    buf = vector;
    (this->*pack_choice[0])(0);
  } else {
    if (nmax) buf = &array[0][0];
    else buf = NULL;
    for (int n = 0; n < nvalues; n++)
      (this->*pack_choice[n])(n);
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputePropertyAtom::memory_usage()
{
  double bytes = nmax*nvalues * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   one method for every keyword compute property/atom can output
   the atom property is packed into buf starting at n with stride nvalues
   customize a new keyword by adding a method
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_id(int n)
{
  int *tag = atom->tag;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = tag[i];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_molecule(int n)
{
  int *molecule = atom->molecule;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = molecule[i];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_type(int n)
{
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = type[i];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_vol(int n)
{
  double *vol = atom->volume;
  double *rad = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    double volume = 0.0;
#ifdef SUPERQUADRIC_ACTIVE_FLAG
    if(atom->superquadric_flag)
      volume = vol[i];
    else
      volume = 4.0*M_PI*rad[i]*rad[i]*rad[i]/3.0;
#else
    volume = 4.0*M_PI*rad[i]*rad[i]*rad[i]/3.0;
#endif
    if (mask[i] & groupbit) buf[n] = volume;
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_eq_radius(int n)
{
  double *vol = atom->volume;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    double req = atom->radius[i];
#ifdef SUPERQUADRIC_ACTIVE_FLAG
    if(atom->superquadric_flag)
      req = cbrt(3.0*vol[i]/(4.0*M_PI));
#endif
    if (mask[i] & groupbit) buf[n] = req;
    else buf[n] = 0.0;
    n += nvalues;
  }
}
/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_mass(int n)
{
  int *type = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (rmass) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = rmass[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = mass[type[i]];
      else buf[n] = 0.0;
      n += nvalues;
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_x(int n)
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = x[i][0];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_y(int n)
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = x[i][1];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_z(int n)
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = x[i][2];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_xs(int n)
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double boxxlo = domain->boxlo[0];
  double invxprd = 1.0/domain->xprd;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = (x[i][0] - boxxlo) * invxprd;
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_ys(int n)
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double boxylo = domain->boxlo[1];
  double invyprd = 1.0/domain->yprd;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = (x[i][1] - boxylo) * invyprd;
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_zs(int n)
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double boxzlo = domain->boxlo[2];
  double invzprd = 1.0/domain->zprd;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = (x[i][2] - boxzlo) * invzprd;
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_xs_triclinic(int n)
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double *boxlo = domain->boxlo;
  double *h_inv = domain->h_inv;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit)
      buf[n] = h_inv[0]*(x[i][0]-boxlo[0]) +
        h_inv[5]*(x[i][1]-boxlo[1]) + h_inv[4]*(x[i][2]-boxlo[2]);
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_ys_triclinic(int n)
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double *boxlo = domain->boxlo;
  double *h_inv = domain->h_inv;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit)
      buf[n] = h_inv[1]*(x[i][1]-boxlo[1]) + h_inv[3]*(x[i][2]-boxlo[2]);
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_zs_triclinic(int n)
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double *boxlo = domain->boxlo;
  double *h_inv = domain->h_inv;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit)
      buf[n] = h_inv[2]*(x[i][2]-boxlo[2]);
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_xu(int n)
{
  double **x = atom->x;
  tagint *image = atom->image;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double xprd = domain->xprd;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit)
      buf[n] = x[i][0] + ((image[i] & IMGMASK) - IMGMAX) * xprd;
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_yu(int n)
{
  double **x = atom->x;
  tagint *image = atom->image;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double yprd = domain->yprd;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit)
      buf[n] = x[i][1] + ((image[i] >> IMGBITS & IMGMASK) - IMGMAX) * yprd;
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_zu(int n)
{
  double **x = atom->x;
  tagint *image = atom->image;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double zprd = domain->zprd;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit)
      buf[n] = x[i][2] + ((image[i] >> IMG2BITS) - IMGMAX) * zprd;
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_xu_triclinic(int n)
{
  double **x = atom->x;
  tagint *image = atom->image;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double *h = domain->h;
  int xbox,ybox,zbox;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      xbox = (image[i] & IMGMASK) - IMGMAX;
      ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
      zbox = (image[i] >> IMG2BITS) - IMGMAX;
      buf[n] = x[i][0] + h[0]*xbox + h[5]*ybox + h[4]*zbox;
    } else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_yu_triclinic(int n)
{
  double **x = atom->x;
  tagint *image = atom->image;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double *h = domain->h;
  int ybox,zbox;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
      zbox = (image[i] >> IMG2BITS) - IMGMAX;
      buf[n] = x[i][1] + h[1]*ybox + h[3]*zbox;
    } else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_zu_triclinic(int n)
{
  double **x = atom->x;
  tagint *image = atom->image;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double *h = domain->h;
  int zbox;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      zbox = (image[i] >> IMG2BITS) - IMGMAX;
      buf[n] = x[i][2] + h[2]*zbox;
    } else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_ix(int n)
{
  tagint *image = atom->image;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = (image[i] & IMGMASK) - IMGMAX;
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_iy(int n)
{
  tagint *image = atom->image;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_iz(int n)
{
  tagint *image = atom->image;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = (image[i] >> IMG2BITS) - IMGMAX;
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_vx(int n)
{
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = v[i][0];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_vy(int n)
{
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = v[i][1];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_vz(int n)
{
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = v[i][2];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_fx(int n)
{
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = f[i][0];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_fy(int n)
{
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = f[i][1];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_fz(int n)
{
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = f[i][2];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_mag_fx(int n)
{
  double **mag_f = atom->mag_f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = mag_f[i][0];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_mag_fy(int n)
{
  double **mag_f = atom->mag_f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = mag_f[i][1];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_mag_fz(int n)
{
  double **mag_f = atom->mag_f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = mag_f[i][2];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_q(int n)
{
  double *q = atom->q;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = q[i];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_mux(int n)
{
  double **mu = atom->mu;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = mu[i][0];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_muy(int n)
{
  double **mu = atom->mu;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = mu[i][1];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_muz(int n)
{
  double **mu = atom->mu;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = mu[i][2];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_mu(int n)
{
  double **mu = atom->mu;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = mu[i][3];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_radius(int n)
{
  double *radius = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = radius[i];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_diameter(int n)
{
  double *radius = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = 2.0*radius[i];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_omegax(int n)
{
  double **omega = atom->omega;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = omega[i][0];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_omegay(int n)
{
  double **omega = atom->omega;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = omega[i][1];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_omegaz(int n)
{
  double **omega = atom->omega;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = omega[i][2];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_angmomx(int n)
{
  double **angmom = atom->angmom;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = angmom[i][0];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_angmomy(int n)
{
  double **angmom = atom->angmom;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = angmom[i][1];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_angmomz(int n)
{
  double **angmom = atom->angmom;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = angmom[i][2];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_shapex(int n)
{
  AtomVecEllipsoid::Bonus *bonus = avec_ellipsoid->bonus;
  int *ellipsoid = atom->ellipsoid;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if ((mask[i] & groupbit) && ellipsoid[i] >= 0)
      buf[n] = bonus[ellipsoid[i]].shape[0];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_shapey(int n)
{
  AtomVecEllipsoid::Bonus *bonus = avec_ellipsoid->bonus;
  int *ellipsoid = atom->ellipsoid;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if ((mask[i] & groupbit) && ellipsoid[i] >= 0)
      buf[n] = bonus[ellipsoid[i]].shape[1];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_shapez(int n)
{
  AtomVecEllipsoid::Bonus *bonus = avec_ellipsoid->bonus;
  int *ellipsoid = atom->ellipsoid;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if ((mask[i] & groupbit) && ellipsoid[i] >= 0)
      buf[n] = bonus[ellipsoid[i]].shape[2];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_quatw(int n)
{
  if (avec_ellipsoid) {
    AtomVecEllipsoid::Bonus *bonus = avec_ellipsoid->bonus;
    int *ellipsoid = atom->ellipsoid;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if ((mask[i] & groupbit) && ellipsoid[i] >= 0)
        buf[n] = bonus[ellipsoid[i]].quat[0];
      else buf[n] = 0.0;
      n += nvalues;
    }

  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_quati(int n)
{
  if (avec_ellipsoid) {
    AtomVecEllipsoid::Bonus *bonus = avec_ellipsoid->bonus;
    int *ellipsoid = atom->ellipsoid;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if ((mask[i] & groupbit) && ellipsoid[i] >= 0)
        buf[n] = bonus[ellipsoid[i]].quat[1];
      else buf[n] = 0.0;
      n += nvalues;
    }

  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_quatj(int n)
{
  if (avec_ellipsoid) {
    AtomVecEllipsoid::Bonus *bonus = avec_ellipsoid->bonus;
    int *ellipsoid = atom->ellipsoid;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if ((mask[i] & groupbit) && ellipsoid[i] >= 0)
        buf[n] = bonus[ellipsoid[i]].quat[2];
      else buf[n] = 0.0;
      n += nvalues;
    }

  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_quatk(int n)
{
  if (avec_ellipsoid) {
    AtomVecEllipsoid::Bonus *bonus = avec_ellipsoid->bonus;
    int *ellipsoid = atom->ellipsoid;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if ((mask[i] & groupbit) && ellipsoid[i] >= 0)
        buf[n] = bonus[ellipsoid[i]].quat[3];
      else buf[n] = 0.0;
      n += nvalues;
    }

  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_tqx(int n)
{
  double **torque = atom->torque;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = torque[i][0];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_tqy(int n)
{
  double **torque = atom->torque;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = torque[i][1];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_tqz(int n)
{
  double **torque = atom->torque;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = torque[i][2];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_end1x(int n)
{
  AtomVecLine::Bonus *bonus = avec_line->bonus;
  int *line = atom->line;
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if ((mask[i] & groupbit) && line[i] >= 0)
      buf[n] = x[i][0] - 0.5*bonus[line[i]].length*cos(bonus[line[i]].theta);
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_end1y(int n)
{
  AtomVecLine::Bonus *bonus = avec_line->bonus;
  int *line = atom->line;
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if ((mask[i] & groupbit) && line[i] >= 0)
      buf[n] = x[i][1] - 0.5*bonus[line[i]].length*sin(bonus[line[i]].theta);
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_end1z(int n)
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = x[i][2];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_end2x(int n)
{
  AtomVecLine::Bonus *bonus = avec_line->bonus;
  int *line = atom->line;
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if ((mask[i] & groupbit) && line[i] >= 0)
      buf[n] = x[i][0] + 0.5*bonus[line[i]].length*cos(bonus[line[i]].theta);
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_end2y(int n)
{
  AtomVecLine::Bonus *bonus = avec_line->bonus;
  int *line = atom->line;
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if ((mask[i] & groupbit) && line[i] >= 0)
      buf[n] = x[i][1] + 0.5*bonus[line[i]].length*sin(bonus[line[i]].theta);
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_end2z(int n)
{
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = x[i][2];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_iname(int n)
{
  int *ivector = atom->ivector[index[n]];
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = ivector[i];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyAtom::pack_dname(int n)
{
  double *dvector = atom->dvector[index[n]];
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = dvector[i];
    else buf[n] = 0.0;
    n += nvalues;
  }
}
