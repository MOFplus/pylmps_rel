/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Pierre de Buyl (KU Leuven)
                        Rochus Schmid (RUB)
                        Note: this is a rip off of Pierre de Buyl's h5md dump
                              to write hdf5 based pdlp files. Thanks to Pierre for the clear code!
------------------------------------------------------------------------- */

/* This is an experiment .. first we get rid of everything and only write positions in the default interval
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include "hdf5.h"
#include "dump_pdlp.h"
#include "domain.h"
#include "atom.h"
#include "update.h"
#include "group.h"
#include "output.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "version.h"

using namespace LAMMPS_NS;

#define MYMIN(a,b) ((a) < (b) ? (a) : (b))
#define MYMAX(a,b) ((a) > (b) ? (a) : (b))

/** Scan common options for the dump elements
 */
static int element_args(int narg, char **arg, int *every)
{
  int iarg=0;
  while (iarg<narg) {
    if (strcmp(arg[iarg], "every")==0) {
      if (narg<2) return -1;
      *every = atoi(arg[iarg+1]);
      iarg+=2;
    } else {
      break;
    }
  }
  return iarg;
}

/* ---------------------------------------------------------------------- */

DumpPDLP::DumpPDLP(LAMMPS *lmp, int narg, char **arg) : Dump(lmp, narg, arg)
{
  if (narg<6) error->all(FLERR,"Illegal dump pdlp command");
  if (binary || compressed || multifile || multiproc)
    error->all(FLERR,"Invalid dump pdlp filename");

  if (domain->triclinic!=0)
    error->all(FLERR,"Invalid domain for dump pdlp. Only orthorombic domains supported.");

  size_one = 6;
  sort_flag = 1;
  sortcol = 0;
  format_default = NULL;
  flush_flag = 0;
  unwrap_flag = 0;
  stage_name=NULL;

  every_dump = force->inumeric(FLERR,arg[3]);
  
  every_xyz = -1;
  /* RS
  every_position = every_image = -1;
  every_velocity = every_force = every_species = -1;
  every_charge = -1;
  */

  int iarg=5;
  int n_parsed, default_every;
  size_one=0;
  if (every_dump==0) default_every=0; else default_every=1;

  while (iarg<narg) {
    if (strcmp(arg[iarg], "xyz")==0) {
      every_xyz=default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_xyz);
      if (n_parsed<0) error->all(FLERR, "Illegal dump pdlp command");
      iarg += n_parsed;
      size_one+=domain->dimension;
    } else if (strcmp(arg[iarg], "stage")==0) {
      if (iarg+1>=narg) {
        error->all(FLERR, "Invalid number of arguments in dump pdlp");
      }
      if (stage_name==NULL) {
        stage_name = new char[strlen(arg[iarg])+1];
        strcpy(stage_name, arg[iarg+1]);
      } else {
        error->all(FLERR, "Illegal dump pdlp command: stage name argument repeated");
      }
      iarg+=2;
    /*  RS
    } else if (strcmp(arg[iarg], "image")==0) {
      if (every_position<0) error->all(FLERR, "Illegal dump h5md command");
      iarg+=1;
      size_one+=domain->dimension;
      every_image = every_position;
    } else if (strcmp(arg[iarg], "velocity")==0) {
      every_velocity = default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_velocity);
      if (n_parsed<0) error->all(FLERR, "Illegal dump h5md command");
      iarg += n_parsed;
      size_one+=domain->dimension;
    } else if (strcmp(arg[iarg], "force")==0) {
      every_force = default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_force);
      if (n_parsed<0) error->all(FLERR, "Illegal dump h5md command");
      iarg += n_parsed;
      size_one+=domain->dimension;
    } else if (strcmp(arg[iarg], "species")==0) {
      every_species=default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_species);
      if (n_parsed<0) error->all(FLERR, "Illegal dump h5md command");
      iarg += n_parsed;
      size_one+=1;
    } else if (strcmp(arg[iarg], "charge")==0) {
      if (!atom->q_flag)
        error->all(FLERR, "Requesting non-allocated quantity q in dump_h5md");
      every_charge = default_every;
      iarg+=1;
      n_parsed = element_args(narg-iarg, &arg[iarg], &every_charge);
      if (n_parsed<0) error->all(FLERR, "Illegal dump h5md command");
      iarg += n_parsed;
      size_one+=1;
    } else if (strcmp(arg[iarg], "file_from")==0) {
      if (iarg+1>=narg) {
        error->all(FLERR, "Invalid number of arguments in dump h5md");
      }
      if (box_is_set||create_group_is_set)
        error->all(FLERR, "Cannot set file_from in dump h5md after box or create_group");
      int idump;
      for (idump = 0; idump < output->ndump; idump++)
        if (strcmp(arg[iarg+1],output->dump[idump]->id) == 0) break;
      if (idump == output->ndump) error->all(FLERR,"Cound not find dump_modify ID");
      datafile_from_dump = idump;
      do_box=false;
      create_group=false;
      iarg+=2;
    } else if (strcmp(arg[iarg], "box")==0) {
      if (iarg+1>=narg) {
        error->all(FLERR, "Invalid number of arguments in dump h5md");
      }
      box_is_set = true;
      if (strcmp(arg[iarg+1], "yes")==0)
        do_box=true;
      else if (strcmp(arg[iarg+1], "no")==0)
        do_box=false;
      else
        error->all(FLERR, "Illegal dump h5md command");
      iarg+=2;
    } else  if (strcmp(arg[iarg], "create_group")==0) {
      if (iarg+1>=narg) {
        error->all(FLERR, "Invalid number of arguments in dump h5md");
      }
      create_group_is_set = true;
      if (strcmp(arg[iarg+1], "yes")==0)
        create_group=true;
      else if (strcmp(arg[iarg+1], "no")==0) {
        create_group=false;
      }
      else
        error->all(FLERR, "Illegal dump h5md command");
      iarg+=2;
    } else if (strcmp(arg[iarg], "author")==0) {
      if (iarg+1>=narg) {
        error->all(FLERR, "Invalid number of arguments in dump h5md");
      }
      if (author_name==NULL) {
        author_name = new char[strlen(arg[iarg])+1];
        strcpy(author_name, arg[iarg+1]);
      } else {
        error->all(FLERR, "Illegal dump h5md command: author argument repeated");
      }
      iarg+=2;
    */
    } else {
      error->all(FLERR, "Invalid argument to dump h5md");
    }
  }

  // allocate global array for atom coords

  bigint n = group->count(igroup);
  natoms = static_cast<int> (n);

  if (every_xyz>=0)
    memory->create(dump_xyz,domain->dimension*natoms,"dump:xyz");
  /* RS
  if (every_image>=0)
    memory->create(dump_image,domain->dimension*natoms,"dump:image");
  if (every_velocity>=0)
    memory->create(dump_velocity,domain->dimension*natoms,"dump:velocity");
  if (every_force>=0)
    memory->create(dump_force,domain->dimension*natoms,"dump:force");
  if (every_species>=0)
    memory->create(dump_species,natoms,"dump:species");
  if (every_charge>=0)
    memory->create(dump_charge,natoms,"dump:charge");
  */

  // RS here the file is opened .. we need to see if we can just pass the hid_t of the hdf5 file and access it
  openfile();
  ntotal = 0;
}

/* ---------------------------------------------------------------------- */

DumpPDLP::~DumpPDLP()
{
  //  needs fixing!! RS
  if (every_xyz>=0) {
    memory->destroy(dump_xyz);
    if (me==0) {
      H5Dclose(xyz_dset);

    }
  }
  /* RS
  if (every_image>=0) {
    memory->destroy(dump_image);
    if (me==0) h5md_close_element(particles_data.image);
  }
  if (every_velocity>=0) {
    memory->destroy(dump_velocity);
    if (me==0) h5md_close_element(particles_data.velocity);
  }
  if (every_force>=0) {
    memory->destroy(dump_force);
    if (me==0) h5md_close_element(particles_data.force);
  }
  if (every_species>=0) {
    memory->destroy(dump_species);
    if (me==0) h5md_close_element(particles_data.species);
  }
  if (every_charge>=0) {
    memory->destroy(dump_charge);
    if (me==0) h5md_close_element(particles_data.charge);
  }
  */
  if (me==0){
    H5Gclose(traj_group);
    H5Gclose(stage_group);
    H5Fclose(pdlpfile);
  }
}

/* ---------------------------------------------------------------------- */

void DumpPDLP::init_style()
{
  if (sort_flag == 0 || sortcol != 0)
    error->all(FLERR,"Dump pdlp requires sorting by atom ID");
}

/* ---------------------------------------------------------------------- */

void DumpPDLP::openfile()
{
  int dims[2];
  /*
  char *boundary[3];
  for (int i=0; i<3; i++) {
    boundary[i] = new char[9];
    if (domain->periodicity[i]==1) {
      strcpy(boundary[i], "periodic");
    } else {
      strcpy(boundary[i], "none");
    }
  }
  */

  if (me == 0) {
    // me == 0 _> do only on master node
    
    pdlpfile = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    stage_group = H5Gopen(pdlpfile, stage_name, H5P_DEFAULT);
    traj_group  = H5Gopen(stage_group, "traj", H5P_DEFAULT);

    if (every_xyz>0)
      xyz_dset    = H5Dopen(traj_group, "xyz", H5P_DEFAULT);

    printf("DEBUG : hdf5 IDs  %d %d %d %d\n", pdlpfile, stage_group, traj_group, xyz_dset);
    // only needed if we generate the memspace/dataspace etc here once
    dims[0] = natoms;
    dims[1] = domain->dimension;
  }
}

/* ---------------------------------------------------------------------- */

void DumpPDLP::write_header(bigint nbig)
{
  return;
}

/* ---------------------------------------------------------------------- */

void DumpPDLP::pack(tagint *ids)
{
  int m,n;

  tagint *tag = atom->tag;
  double **x = atom->x;
  /*
  double **v = atom->v;
  double **f = atom->f;
  int *species = atom->type;
  double *q = atom->q;
  */
  imageint *image = atom->image;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int dim=domain->dimension;

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  m = n = 0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (every_xyz>=0) {
        int ix = (image[i] & IMGMASK) - IMGMAX;
        int iy = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
        int iz = (image[i] >> IMG2BITS) - IMGMAX;
        if (unwrap_flag == 1) {
          buf[m++] = (x[i][0] + ix * xprd);
          buf[m++] = (x[i][1] + iy * yprd);
          if (dim>2) buf[m++] = (x[i][2] + iz * zprd);
        } else {
          buf[m++] = x[i][0];
          buf[m++] = x[i][1];
          if (dim>2) buf[m++] = x[i][2];
        }
        /*
        if (every_image>=0) {
          buf[m++] = ix;
          buf[m++] = iy;
          if (dim>2) buf[m++] = iz;
        }
        */
      }
      /*
      if (every_velocity>=0) {
        buf[m++] = v[i][0];
        buf[m++] = v[i][1];
        if (dim>2) buf[m++] = v[i][2];
      }
      if (every_force>=0) {
        buf[m++] = f[i][0];
        buf[m++] = f[i][1];
        if (dim>2) buf[m++] = f[i][2];
      }
      if (every_species>=0)
        buf[m++] = species[i];
      if (every_charge>=0)
        buf[m++] = q[i];
      */
      ids[n++] = tag[i];
    }
}

/* ---------------------------------------------------------------------- */

void DumpPDLP::write_data(int n, double *mybuf)
{
  // copy buf atom coords into global array

  int m = 0;
  int dim = domain->dimension;
  int k = dim*ntotal;
  /*
  int k_image = dim*ntotal;
  int k_velocity = dim*ntotal;
  int k_force = dim*ntotal;
  int k_species = ntotal;
  int k_charge = ntotal;
  */
  for (int i = 0; i < n; i++) {
    if (every_xyz>=0) {
      for (int j=0; j<dim; j++) {
        dump_xyz[k++] = mybuf[m++];
      }
      /*
      if (every_image>=0)
        for (int j=0; j<dim; j++) {
          dump_image[k_image++] = mybuf[m++];
        }
      */
    }
    /*
    if (every_velocity>=0)
      for (int j=0; j<dim; j++) {
        dump_velocity[k_velocity++] = mybuf[m++];
      }
    if (every_force>=0)
      for (int j=0; j<dim; j++) {
        dump_force[k_force++] = mybuf[m++];
      }
    if (every_species>=0)
      dump_species[k_species++] = mybuf[m++];
    if (every_charge>=0)
      dump_charge[k_charge++] = mybuf[m++];
    */
    ntotal++;
  }

  // if last chunk of atoms in this snapshot, write global arrays to file

  if (ntotal == natoms) {
    if (every_xyz>0) {
      write_frame();
      ntotal = 0;
    } 
    /*else {
      write_fixed_frame();
    }
    */
  }
}

/* ---------------------------------------------------------------------- */

int DumpPDLP::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"unwrap") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal dump_modify command");
    if (strcmp(arg[1],"yes") == 0) unwrap_flag = 1;
    else if (strcmp(arg[1],"no") == 0) unwrap_flag = 0;
    else error->all(FLERR,"Illegal dump_modify command");
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

void DumpPDLP::write_frame()
{
  herr_t  status;
  hsize_t dims[3], start[3], count[3];
  hid_t   fspace, mspace;

  int local_step;
  double local_time;
  double edges[3];
  int i;
  local_step = update->ntimestep;
  local_time = local_step * update->dt;
  edges[0] = boxxhi - boxxlo;
  edges[1] = boxyhi - boxylo;
  edges[2] = boxzhi - boxzlo;

  printf("DEBUG: this is write frame every_dump=%d every_xyz=%d local_step=%d\n", every_dump,every_xyz,local_step);
  
  if (every_xyz>0) {
    if (local_step % (every_xyz*every_dump) == 0) {
      // get fspace of dataset
      fspace = H5Dget_space(xyz_dset);
      // get current dims
      H5Sget_simple_extent_dims(fspace, dims, NULL);

      printf("DEBUG: in pdlp dump, dims before extend: %llu %llu %llu\n", dims[0], dims[1], dims[2]);
      printf("xyz first numbers %12.6f %12.6f %12.6f\n", dump_xyz[0], dump_xyz[1], dump_xyz[2]);


      // increment by one frame
      dims[0] += 1;
      status = H5Dset_extent(xyz_dset, dims);
      if (status<0){
        printf("Extending pdlp dataset went wrong! status is %d\n", status);
      }
      H5Sclose(fspace);

      // Now get fspace again
      fspace = H5Dget_space(xyz_dset);
      // generate a mspace for the data in memory
      mspace = H5Screate_simple(2, dims+1, NULL);
      // create start and offset
      start[0] = dims[0]-1;
      count[0] = 1;
      for (i=1; i<3; i++) {
        start[i] = 0;
        count[i] = dims[i];
      }
      // select part of file to be writen
      status = H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);
      if (status<0){
        printf("Selecting hyperslab went wrong! status is %d\n", status);
      }
      printf("DEBUG: in pdlp dump, start: %llu %llu %llu\n", start[0], start[1], start[2]);
      printf("DEBUG: in pdlp dump, count: %llu %llu %llu\n", count[0], count[1], count[2]);
      // write the data
      status = H5Dwrite(xyz_dset, H5T_IEEE_F64LE, mspace, fspace, H5P_DEFAULT, dump_xyz);
      if (status<0){
        printf("Writing data went wrong! status is %d\n", status);
      }      
      // close selections
      H5Sclose(fspace);
      H5Sclose(mspace);
      /*
      if (every_image>0)
        h5md_append(particles_data.image, dump_image, local_step, local_time);
      */
    }
  /*
  } else {
    if (do_box) h5md_append(particles_data.box_edges, edges, local_step, local_time);
  }
  if (every_velocity>0 && local_step % (every_velocity*every_dump) == 0) {
    h5md_append(particles_data.velocity, dump_velocity, local_step, local_time);
  }
  if (every_force>0 && local_step % (every_force*every_dump) == 0) {
    h5md_append(particles_data.force, dump_force, local_step, local_time);
  }
  if (every_species>0 && local_step % (every_species*every_dump) == 0) {
    h5md_append(particles_data.species, dump_species, local_step, local_time);
  }
  if (every_charge>0 && local_step % (every_charge*every_dump) == 0) {
    h5md_append(particles_data.charge, dump_charge, local_step, local_time);
  */
  }
}

/*
void DumpPDLP::write_fixed_frame()
{
  double edges[3];
  int dims[2];
  char *boundary[3];

  for (int i=0; i<3; i++) {
    boundary[i] = new char[9];
    if (domain->periodicity[i]==1) {
      strcpy(boundary[i], "periodic");
    } else {
      strcpy(boundary[i], "none");
    }
  }

  dims[0] = natoms;
  dims[1] = domain->dimension;

  edges[0] = boxxhi - boxxlo;
  edges[1] = boxyhi - boxylo;
  edges[2] = boxzhi - boxzlo;
  if (every_position==0) {
    particles_data.position = h5md_create_fixed_data_simple(particles_data.group, "position", 2, dims, H5T_NATIVE_DOUBLE, dump_position);
    h5md_create_box(&particles_data, dims[1], boundary, false, edges, NULL);
    if (every_image==0)
      particles_data.image = h5md_create_fixed_data_simple(particles_data.group, "image", 2, dims, H5T_NATIVE_INT, dump_image);
  }
  if (every_velocity==0)
    particles_data.velocity = h5md_create_fixed_data_simple(particles_data.group, "velocity", 2, dims, H5T_NATIVE_DOUBLE, dump_velocity);
  if (every_force==0)
    particles_data.force = h5md_create_fixed_data_simple(particles_data.group, "force", 2, dims, H5T_NATIVE_DOUBLE, dump_force);
  if (every_species==0)
    particles_data.species = h5md_create_fixed_data_simple(particles_data.group, "species", 1, dims, H5T_NATIVE_INT, dump_species);
  if (every_charge==0) {
    particles_data.charge = h5md_create_fixed_data_simple(particles_data.group, "charge", 1, dims, H5T_NATIVE_INT, dump_charge);
    h5md_write_string_attribute(particles_data.group, "charge", "type", "effective");
  }

  for (int i=0; i<3; i++) {
    delete [] boundary[i];
  }
}

*/