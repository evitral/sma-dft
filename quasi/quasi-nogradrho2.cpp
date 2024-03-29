/**********************************************************
 *                                                        *
 *                 quasi-nogradrho.cpp                    *
 *                                                        *
 *     Author: Eduardo Vitral Freigedo Rodrigues          *
 *     Affilitation: University of Minnesota              *
 *     Last mod: 01/21/2020                               *
 *                                                        *
 *    Parallel FFTW implementation of a phase field       *
 *    model for a 3D smectic-isotropic system of          *
 *    varying density. We adopt quasi-incompressibility,  *
 *    and the density is computed at every time step      *
 *    from the order parameter (constitutive relation).   *
 *    The model is built of an equation for the order     *
 *    parameter, a balance of mass and a balance of       *
 *    linear momentum. Details can be found at:           *
 *    i)  Vitral et al, PRE 100(3)                        *
 *    ii) Vitral et al, in preparation                    *
 *                                                        *
 *    Independent variables: order parameter, velocitiy   *
 *                                                        *
 *    Note: gradients of rho turned off                   *
 *                                                        *
 *    Boundary conditions: no flux bcs                    *
 *    i)  Order parameter: Neumann (DCT)                  *
 *    ii) Velocity: no-slip (DST)                         *
 *                                                        *
 **********************************************************/

/* General */

#include <vector>        // std containers
#include <cassert>       
#include <cstdlib>       // std::exit()
#include <fftw3-mpi.h>

/* Input, output, string */

#include <string>
#include <iostream>
#include <iomanip>       // std::setw
#include <fstream>       // read, write

/* Math */

#include <algorithm>     // for max/min
#include <cmath>
#include <complex>

/* Time control (need c++11 for chrono) */

#include <ctime>
#include <cstdio>
#include <chrono>


/************** Compilation *****************

MSI:

module load intel ompi/intel

mpicxx -I /soft/fftw/intel-ompi/3.3-double/include -o code code.cpp
-L /soft/fftw/intel-ompi/3.3-double/lib -lfftw3_mpi -lfftw3 -lm -std=c++11 

COMET:

module load gnutools
module load intel/2018.1.163 mvapich2_ib 

mpicxx -I /opt/fftw/3.3.8/intel/mvapich2_ib/include -O2 -o code code.cpp 
-L /opt/fftw/3.3.8/intel/mvapich2_ib/lib -lfftw3_mpi -lfftw3 -lm -std=c++11 

********************************************/

/********************************************
 *                                          *
 *               FUNCTIONS                  *
 *                                          *
 *******************************************/    

// Possible wall potential (default off)

double wallPotential(int z)
{
  double wP;
  double sigma = 0.0; // 1
  double z0 = 0.0001; //0.0001;
  wP = sigma*exp(-z/z0);
  return wP;
}

void saveMid(int Nx, int Ny, int Nz, int local_n0, long long rank, double alloc_slice, 
	     std::vector<double> psiSlice_local, std::vector<double> psiSlice,
	     std::vector<double> target_local, std::ofstream& target_output,
	     std::string strBox, std::string target_file)
{
  int i, j, k, i_local, index, index2;

  j = Ny/2;
  for( k = 0; k < Nz ; k++ ){
  for( i_local = 0; i_local < local_n0 ; i_local++ ){
    index  = (i_local*Ny +j)*Nz + k;
    index2 = i_local*Nz + k;
    psiSlice_local[index2] = target_local[index];
  }}

  MPI::COMM_WORLD.Gather(psiSlice_local.data(),alloc_slice,MPI::DOUBLE,
			 psiSlice.data(),alloc_slice, MPI::DOUBLE,0);

  if (rank == 0 )
  {
    target_output.open(strBox+target_file,std::ios_base::app);
									
    assert(target_output.is_open());

    for ( i = 0; i < Nx; i++ ) {
    for ( k = 0; k < Nz; k++ ) {
			
      index = i*Nz + k;
			
      target_output << psiSlice[index] << "\n";
    }}
    
    target_output.close();

  }
}



/********************************************
 *                                          *
 *                 MAIN                     *
 *                                          *
 *******************************************/    

int main(int argc, char* argv[]) {

/* FFTW plans */

  fftw_plan planCT, planSTx, planSTy, planSTz, iPlanSTx, iPlanSTy, iPlanSTz, iPlanCT;

/* Indices and mpi related numbers */

  int i, j, k, index, i_local, size;
  long long rank;

/* Fourier space doubles */

  double mq2, opSH, dotSqVq;
  double Sx, Sy, Sz;

/* L1 related doubles + output */

  double L1, limL1, sumA, sumA_local, sumB, sumB_local;
  std::ofstream L1_output;

/* mass output */

  double rho_sum, rho_sum_local;
  std::ofstream mass_output;

/* energy output */

  double energy, energy_sum;
  std::ofstream energy_output;

/* Ints and doubles for surface info */

  int index1, index2, i2, j2, k2, track;
	
/* Load/save parameters */

  int load = atof(argv[4]);  // (load YES == 1)

  int swtPsi = 0;  // (switch: psi.dat/psiB.dat)

  std::string swt_str;

  std::string strPsi = "psi";

  std::string strLoad = "/oasis/scratch/comet/evitral/temp_project/quasi_fc/gradrho_off/qsi-fc140-nw8-nu";	
	
  strLoad += argv[1] + std::string("-e0d") + argv[2] 
    + std::string("/save/");

  std::ofstream psiMid_output, surf_output, velS_output, 
    curvH_output, curvK_output, vx_output, vy_output, vz_output,
    vsolx_output, vsoly_output, vsolz_output, 
    divv_output, rho_output, p_output, info_output,
    fx_output, fy_output, fz_output,
    sx_output, sy_output, sz_output,
    energyMid_output, surface_output;

  std::ofstream *swt_output;

  std::string strBox = "/oasis/scratch/comet/evitral/temp_project/quasi_fc/gradrho_off/qsi-fc140-nw8-nu";

  strBox += argv[1] + std::string("-e0d") + argv[2] 
    + std::string("/");

 
  /* Save intervals */ 

  const int stepL1 = 50; // 50
  const int stepSave = 10; // 4 or 10
  const int divvSwitch = 200; // 100 or 200
	
/* ptrdiff_t: integer type, optimizes large transforms 64bit machines */

  const ptrdiff_t Nx = 256, Ny = 256, Nz = 256;
  const ptrdiff_t NG = Nx*Ny*Nz;
  const ptrdiff_t Nslice = Ny*Nz;
	
  ptrdiff_t alloc_local, local_n0, local_0_start;

  const int Nym = Ny-1, Nzm = Nz-1;

/* Constants and variables for morphologies (Nx = Ny = Nz) */

  const double mid = Ny/2; 
  const double aE = atof(argv[3]);   // focal conic dimensions
  const double bE = atof(argv[3]);   // same to maintain layer spacing
  double aE2, bE2; // 2nd focal conic
  double gap = 0; //32

  double xs, ys, zs, ds;

/* Phase Field parameters */

  const double gamma =  1.0; //1.0;
  const double beta  =  2.0; //2.0;
  const double alpha =  1.0;
  double ep_arg    = atof(argv[2]); 
  const double ep = -0.001*ep_arg;
  const double q0    =  1.0; 
  const double q02   = q0*q0;

/* Balance of Linear Momentum parameters */

  double nu = atof(argv[1]);
  double Amp = 1.34164; //1.328; 
  double rho_0 = 0.005; // 0.5
  double kp = 0.5; // 0.3755 for rho_s = 1, with rho_0 = 0.5
  double rho_s = kp*Amp+rho_0;
  double rho_m = rho_s/2; // was divided by 2
  double lambda = -2*nu/3;

/* Points per wavelength, time step */
	
  const int    Nw = 8;
  const double dt = 0.001; // 0.0005 for nw = 16, 0.001 for nw = 8	
  const double dtd2  = dt/2;

/* System size and scaling for FFT */

  const double Lx = Nx*2.0*M_PI/(q0*Nw);
  double dx = Lx/(Nx);
  const double Ly = Ny*2.0*M_PI/(q0*Nw);
  double dy = Ly/(Ny);
  const double Lz = Nz*2.0*M_PI/(q0*Nw);
  double dz = Lz/(Nz);

  const double tdx = 2*dx;
  const double tdy = 2*dy;
  const double tdz = 2*dz;
	
  double scale = 0.125/((Nx)*(Ny)*(Nz));

 /* Perturbed smectic layers (off for focal conic) */

  double Qi  = 0.125/2; //0.125/2; // Perturbation wavelength (was over 2)
  double phi = Nw;  // Nw



/********************************************
 *                                          *
 *           Initialize MPI                 *
 *                                          *
 *******************************************/    

  MPI::Init();
  fftw_mpi_init();

  rank = MPI::COMM_WORLD.Get_rank();
  size = MPI::COMM_WORLD.Get_size();

  alloc_local = fftw_mpi_local_size_3d(Nx,Ny,Nz,MPI::COMM_WORLD,
					     &local_n0, &local_0_start);
	
  double alloc_surf = local_n0*Ny;

  double alloc_slice = local_n0*Nz;
	
/* Check: np should divide evenly into Nx, Ny and Nz */

/*

  if (( Nx%size != 0) || ( Ny%size != 0) || ( Nz%size != 0)) 
  {
  if ( rank == 0) 
  {
    std::cout << "!ERROR! : size =  " << size
    << " does not divide evenly into Nx, Ny and Nz."
    << std::endl;
  }
    std::exit(1);
  }

*/

/* Number of processors and initial time */

  if ( rank == 0 ) 
  {
    std::cout << "Using " << size << " processors." << std::endl;
	
    time_t now = time(0);
    char* dNow = ctime(&now);
    		   
    std::cout << "The initial date and time is: " << dNow << std::endl;
  }

/********************************************
 *                                          *
 *              Containers                  *
 *                                          *
 *******************************************/    

// std::vector<double> psi(size*alloc_local);

/* Local data containers */

  std::vector <double> Vqx(local_n0), Vqy(Ny), Vqz(Nz);

  std::vector<double> aLin(alloc_local);
  std::vector<double> C1(alloc_local);
  std::vector<double> C2(alloc_local);
  std::vector<double> psi_local(alloc_local);
  std::vector<double> psiq_local(alloc_local);
  std::vector<double> psi_old_local(alloc_local);
  std::vector<double> Nl_local(alloc_local);
  std::vector<double> Nl_old_local(alloc_local);
	
/* Local data containers (wall potential) */
	
//  std::vector<double> wall(alloc_local);
//  std::vector<double> substrate(alloc_local);

/* Local data containers (advection) */

  std::vector <double> Vsx(local_n0), Vsy(Ny), Vsz(Nz);

  std::vector<double> psiGradx_local(alloc_local);
  std::vector<double> psiGrady_local(alloc_local);
  std::vector<double> psiGradz_local(alloc_local);
	
  std::vector<double> Sx_local(alloc_local);
  std::vector<double> Sy_local(alloc_local);
  std::vector<double> Sz_local(alloc_local);

  std::vector<double> Sx2_local(alloc_local);
  std::vector<double> Sy2_local(alloc_local);
  std::vector<double> Sz2_local(alloc_local);

  std::vector<double> fx_local(alloc_local);
  std::vector<double> fy_local(alloc_local);
  std::vector<double> fz_local(alloc_local);

  std::vector<double> vsolx_local(alloc_local);
  std::vector<double> vsoly_local(alloc_local);
  std::vector<double> vsolz_local(alloc_local);

  std::vector<double> velx_local(alloc_local);
  std::vector<double> vely_local(alloc_local);
  std::vector<double> velz_local(alloc_local);
	
  std::vector<double> psi_temp(Nslice);
  std::vector<double> psi_front(Nslice);
  std::vector<double> psi_back(Nslice);
  std::vector<double> psi_front2(Nslice);
  std::vector<double> psi_back2(Nslice);
  std::vector<double> psi_front3(Nslice);
  std::vector<double> psi_back3(Nslice);

  std::vector<double> CM1x(alloc_local);
  std::vector<double> CM1y(alloc_local);
  std::vector<double> CM1z(alloc_local);
  std::vector<double> CM2x(alloc_local);
  std::vector<double> CM2y(alloc_local);
  std::vector<double> CM2z(alloc_local);
  
  std::vector<double> trans_local(alloc_local);

/* Local data containers (density) */

  std::vector<double> rho_local(alloc_local);
  std::vector<double> rhoq_local(alloc_local);	
  std::vector<double> rho_old_local(alloc_local);

  std::vector<double> p_local(alloc_local);	

  std::vector<double> divv_local(alloc_local);	

  std::vector<double> mu_local(alloc_local);	
  std::vector<double> energy_local(alloc_local);	

  // std::vector<double> psiLapDx_local(alloc_local);
  // std::vector<double> psiLapDy_local(alloc_local);
  // std::vector<double> psiLapDz_local(alloc_local);

  std::vector<double> rhoDx_local(alloc_local);
  std::vector<double> rhoDy_local(alloc_local);
  std::vector<double> rhoDz_local(alloc_local);

  std::vector<double> mq2c(alloc_local);	

  std::vector<double> dfDlapPsi_local(alloc_local);
  std::vector<double> lapRhoDfDlapPsi_local(alloc_local);

  std::vector<double> psiq_old_local(alloc_local);

/* Local data containers (surface info) */

  std::vector<double> psiSlice_local(alloc_slice);
  std::vector<double> surfZ_local(alloc_surf);
		
/* Global data containers (surface info)*/

  std::vector<double> psiSlice(size*alloc_slice);
  std::vector<double> surfZ(size*alloc_surf);

/********************************************
 *                                          *
 *         Wavenumbers for r2r DCT          *
 *                                          *
 *   Note: for some reason the other one    *
 *   also seems to work, but I think this   *
 *   is the right definition.               *
 *                                          *
 *******************************************/


/* Wavenumbers (regular DFT) */

/*
  Vqx[0] = 0.0; Vqx[Nx/2] = 0.5*M_PI/dx;

  for ( i = 1; i < Nx/2; i++ )
  {
    Vqx[i] = 1.0*M_PI*i/(dx*Nx);
    Vqx[Nx/2+i] = -(Nx/2-i)*1.0*M_PI/(dx*Nx);
  }

  Vqy[0] = 0.0; Vqy[Ny/2] = 0.5*M_PI/dy;

  for ( j = 1; j < Ny/2; j++ )
  {
    Vqy[j] = 1.0*M_PI*j/(dy*Ny);
    Vqy[Ny/2+j] = -(Ny/2-j)*1.0*M_PI/(dy*Ny);
  }
*/


/* Wavenumbers (DCT) */

  for ( i_local = 0; i_local < local_n0; i_local++ ) 
  {	
    i = i_local + local_0_start;

    Vqx[i_local] = 1.0*M_PI*i/(dx*Nx);
  }	

  for ( j = 0; j < Ny; j++ )
  {
    Vqy[j] = M_PI*(j)/(dy*Ny);
  }

  for ( k = 0; k < Nz; k++ )
  {
    Vqz[k] = M_PI*(k)/(dz*Nz);
  }

/* Wavenumbers (DST) */

  for ( i_local = 0; i_local < local_n0; i_local++ ) 
  {	
    i = i_local + local_0_start;

    Vsx[i_local] = 1.0*M_PI*(i+1)/(dx*Nx);
  }	

  for ( j = 0; j < Ny; j++ )
  {
    Vsy[j] = M_PI*(j+1)/(dy*Ny);
  }

  for ( k = 0; k < Nz; k++ )
  {
    Vsz[k] = M_PI*(k+1)/(dz*Nz);
  }

	
/********************************************
 *                                          *
 *               FFTW plans                 *
 *                                          *
 *     	 Notes:                             *
 *                                          *
 *   a. REDFT10 has REDFT01 as inverse      *
 *   + 2*N for scaling (in each dim).       *
 *   It seems to be the fastest one.        *
 *                                          *
 *   b. REDTF00 inverse is also REDTF00     *
 *   + 2*(N-1) for scaling (in each dim).   *
 *                                          *
 *******************************************/

  planCT = fftw_mpi_plan_r2r_3d
    (Nx,Ny,Nz,
     trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
     FFTW_REDFT10,FFTW_REDFT10,FFTW_REDFT10,
     FFTW_MEASURE);

  iPlanCT = fftw_mpi_plan_r2r_3d
    (Nx,Ny,Nz,
     trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
     FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,
     FFTW_MEASURE);

  planSTx = fftw_mpi_plan_r2r_3d
    (Nx,Ny,Nz,
     trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
     FFTW_RODFT10,FFTW_REDFT10,FFTW_REDFT10,
     FFTW_MEASURE);

  planSTy = fftw_mpi_plan_r2r_3d
    (Nx,Ny,Nz,
     trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
     FFTW_REDFT10,FFTW_RODFT10,FFTW_REDFT10,
     FFTW_MEASURE);

  planSTz = fftw_mpi_plan_r2r_3d
    (Nx,Ny,Nz,
     trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
     FFTW_REDFT10,FFTW_REDFT10,FFTW_RODFT10,
     FFTW_MEASURE);
	
  iPlanSTx = fftw_mpi_plan_r2r_3d
    (Nx,Ny,Nz,
     trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
     FFTW_RODFT01,FFTW_REDFT01,FFTW_REDFT01,
     FFTW_MEASURE);

  iPlanSTy = fftw_mpi_plan_r2r_3d
    (Nx,Ny,Nz,
     trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
     FFTW_REDFT01,FFTW_RODFT01,FFTW_REDFT01,
     FFTW_MEASURE);

  iPlanSTz = fftw_mpi_plan_r2r_3d
    (Nx,Ny,Nz,
     trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
     FFTW_REDFT01,FFTW_REDFT01,FFTW_RODFT01,
     FFTW_MEASURE);

		
/********************************************
 *                                          *
 *       Initial condition (New/Load)       *
 *                                          *
 *******************************************/

/* A. Initial condition - New */

  if ( load != 1 )
  {


/*************** Not in use ****************

	double Qi  = 2.0; // Perturbation wavelength

	if ( (k > Nx/5) && ( k < 4*Nx/5))
	psi_local[index] = Amp*cos(q0*k*dz);
					 + Amp*0.5*sin(q0*k*dz)*(cos(Qi*i*dx)+cos(Qi*j*dy)); 
					 + Amp*0.5*(cos(Qi*i*dx)+cos(Qi*j*dy));
	
********************************************/

    std::fill(psi_local.begin(),psi_local.end(),0);  

    /** Perturbed smectic **/

    // for ( i_local = 0; i_local < local_n0; i_local++ ) {
    // for ( j = 0; j < Ny; j++ ) {
    // for ( k = 0; k < Nz; k++ ) 
    // {	

    //   if ( (k > Nx/4) && ( k < 3*Nx/4))
    //   {
    // 	i = i_local + local_0_start;

    // 	k2 = k + (int)round(phi*sin(Qi*i*dx));

    // 	index = (i_local*Ny + j) * Nz + k2;

    // 	psi_local[index] = Amp*cos(q0*k*dz);
    //   }
    // }}}

    /** Multiple focal conics **/

    // for ( i_local = 0; i_local < local_n0; i_local++ )
    //   {
    // 	i = i_local + local_0_start;
	
    // 	for ( j = 0; j < Ny; j++ ) {
    // 	for ( k = 0; k < Nz; k++ )
    // 	{
    // 	  index = (i_local*Ny + j) * Nz + k;

    // 	  // 1024 x 1024 x 512                                            
    // 	  // if ( i >= Nx/2 & j >= Ny/2) {i2 = i - Nx/2; j2 = j - Ny/2;}  
    // 	  // 1024 x 512 x 512                                             
    // 	  if (i >= Nx/2) { i2 = i - Nx/2; j2 = j; aE2 = aE + gap; bE2 = bE + gap; }
	      
    // 	  else { i2 = i; j2 = j; aE2 = aE; bE2 = bE; }

    // 	  //aE2 = aE*(1.2-0.4*static_cast<double>(i)/512);
    // 	  bE2 = bE*(1.2-0.4*static_cast<double>(i)/512);

    // 	  if ( k <  bE2 + 1 ) // 18 110 // 24 232  // 62 450               
    // 	  {
    // 	    xs = i2 - mid;
    // 	    ys = j2 - mid;
    // 	    // zs = k + mid*3/4;                                    
    // 	    zs = k;
    // 	    // zs = k-mid for hyperboloid in the middle             
    // 	    // zs = k for hyperboloid in the botton                 
    // 	    ds = sqrt(xs*xs+ys*ys);
    // 	    if (ds < mid)
    // 	    {
    // 	      if (sqrt(pow((ds-mid)/aE2,2)+pow(zs/bE2,2)) > 1)
    // 	      {
    // 		psi_local[index] = 0.0;
    // 	      }
    // 	      else
    // 	      {
    // 		psi_local[index] = Amp*cos(((-0.4*q0/512)*static_cast<double>(i)+1.2*q0)*dz*
    // 					   sqrt(pow((bE2/aE2)*(ds-mid),2)+zs*zs));
    // 	      }
    // 	    }
    // 	    else
    // 	    {
    // 	      if (abs(zs) < bE2)
    // 	      {
    // 		psi_local[index] = Amp*cos(((-0.4*q0/512)*static_cast<double>(i)+1.2*q0)*zs*dz);
    // 	      }
    // 	      else
    // 	      {
    // 		psi_local[index] = 0.0;
    // 	      }
    // 	    }
    // 	  }
    // 	  else
    // 	  {
    // 	    psi_local[index] = 0.0;
    // 	  }
    // 	}}
    //   } // close IC assign     

    /** Single focal conic **/

    for ( i_local = 0; i_local < local_n0; i_local++ ) 
    {
      i = i_local + local_0_start;

      for ( j = 0; j < Ny; j++ ) {
      for ( k = 0; k < Nz; k++ ) 
      {	
        index = (i_local*Ny + j) * Nz + k;
        if ( k <  bE + 1 ) // 18 110 // 24 232  // 62 450
        {		
          xs = i - mid;
          ys = j - mid;
          // zs = k + mid*3/4; 
          zs = k;
          // zs = k-mid for hyperboloid in the middle
          // zs = k for hyperboloid in the botton
          ds = sqrt(xs*xs+ys*ys);
          if (ds < mid)
          {
    	if (sqrt(pow((ds-mid)/aE,2)+pow(zs/bE,2)) > 1)
    	{
    	  psi_local[index] = 0.0;
    	}
    	else
    	{
    	  psi_local[index] = Amp*cos(q0*dz*
    				     sqrt(pow((bE/aE)*(ds-mid),2)+zs*zs));
    	}
          }
          else
          {
    	if (abs(zs) < bE)
    	{
    	  psi_local[index] = Amp*cos(q0*zs*dz);
    	}
    	else
    	{
    	  psi_local[index] = 0.0;
    	}
          }		 
        }
        else
          {
    	psi_local[index] = 0.0;
          }
      }}
    } // close IC assign


/* Output IC to file  */

    /** Create Psi output **/

    strPsi += std::to_string(rank);
    strPsi += ".dat";
    strPsi = strBox + strPsi;
	
    std::ofstream psi_output(strPsi.c_str());
    assert(psi_output.is_open());
    psi_output.close();

  } // End: new psi (A)

	
/* B. Initial condition - Read profile data */

  if ( load == 1 )
  {

    /** Create Psi output **/

    strPsi += std::to_string(rank);
    strPsi += ".dat";
    strLoad = strLoad + strPsi;
    strPsi  = strBox + strPsi;	
    
    std::ifstream psidata(strLoad.c_str());
    assert(psidata.is_open());

    for ( i_local = 0; i_local < local_n0; i_local++ ) 
    {
      i = i_local + local_0_start;
	  
      for ( j = 0; j < Ny; j++ ) {
      for ( k = 0; k < Nz; k++ ) 
      {	
	index = (i_local*Ny + j) * Nz + k;
	psidata >> psi_local[index];
      }}
    }

    psidata.close();

    // This segment if for loading a single .dat
/*	
    strPsi += std::to_string(rank);
    strPsi += ".dat";
    strPsi = strBox + strPsi;

    std::ofstream psi_output(strPsi.c_str());
    assert(psi_output.is_open());
    psi_output.close();
    
    if ( rank == 0 )
    {
	  
    // Open file and obtain IC for global psi
	 
      std::ifstream psidata("/oasis/scratch/comet/evitral/temp_project/dct1024/pyramid.dat");
      assert(psidata.is_open());

      std::cout << "Reading from the file" << std::endl;

      for ( i = 0; i < Nx; i++ ) {
      for ( j = 0; j < Ny; j++ ) {
      for ( k = 0; k < Nz; k++ ) 
      {
        index = (i*Ny + j) * Nz + k;			

	psidata >> psi[index];
      }}}

      psidata.close();
    }	

    // Scatter global psi data

    MPI::COMM_WORLD.Barrier();

    MPI::COMM_WORLD.Scatter(psi.data(),alloc_local,MPI::DOUBLE,
    psi_local.data(),alloc_local, MPI::DOUBLE,0);

*/	
    // End 1 data load here

  } // End: load psi (B)


  /** Create output files **/
		
  if (rank == 0 )
  {	

    /** Create info output **/

    std::ofstream info_output(strBox+"info");
    assert(info_output.is_open());

    info_output << "Nx: " << Nx << "\n";
    info_output << "Ny: " << Ny << "\n";
    info_output << "Nz: " << Nz << "\n";
    info_output << "Points per wavelength (nw): " << Nw << "\n";
    info_output << "q0 (z direction): " << q0 << "\n";
    info_output << "epsilon: " << ep << "\n";
    info_output << "beta: " << beta << "\n";
    info_output << "gamma: " << gamma << "\n";
    info_output << "Viscosity (nu): " << nu << "\n";
    info_output << "dx: " << dx << "\n";
    info_output << "dt: " << dt << "\n";
    info_output << "stepL1: " << stepL1 << "\n";
    info_output << "stepSave: " << stepSave << "\n";
    info_output << "divv switch: " << divvSwitch << "\n";
    info_output << "Initial amp: " << Amp << "\n";
    info_output << "kp: " << kp << "\n";
    info_output << "rho_0: " << rho_0 << "\n";
    info_output << "Perturbation wavenumber Qx (flat): " << Qi << "\n";
    info_output << "Perturbation amplitude phi (flat): " << phi << "\n";
    info_output << "Focal conic initial size (fc): " << aE << "\n";

    info_output.close();
		
    /** Create L1 output **/

    std::ofstream L1_output(strBox+"L1.dat");
    assert(L1_output.is_open());
    L1_output.close();

    /** Create mass output **/

    std::ofstream mass_output(strBox+"mass.dat");
    assert(mass_output.is_open());
    mass_output.close();

    /** Create energy output **/

    std::ofstream energy_output(strBox+"energy.dat");
    assert(energy_output.is_open());
    energy_output.close();

    /** Create psiMid output **/

    std::ofstream psiMid_output(strBox+"psiMid.dat");
    assert(psiMid_output.is_open());
    psiMid_output.close();

    /** Create energyMid output **/

    std::ofstream energyMid_output(strBox+"energyMid.dat");
    assert(energyMid_output.is_open());
    energyMid_output.close();

    /** Create velocity outputs **/

    std::ofstream vx_output(strBox+"vx.dat");
    std::ofstream vy_output(strBox+"vy.dat");
    std::ofstream vz_output(strBox+"vz.dat");
    assert(vx_output.is_open());
    assert(vy_output.is_open());
    assert(vz_output.is_open());
    vx_output.close();
    vy_output.close();
    vz_output.close();

    /** Create solenoidal vel outputs **/

    std::ofstream vsolx_output(strBox+"vsolx.dat");
    std::ofstream vsoly_output(strBox+"vsoly.dat");
    std::ofstream vsolz_output(strBox+"vsolz.dat");
    assert(vsolx_output.is_open());
    assert(vsoly_output.is_open());
    assert(vsolz_output.is_open());
    vsolx_output.close();
    vsoly_output.close();
    vsolz_output.close();

    /** Create projected force outputs **/

    std::ofstream fx_output(strBox+"fx.dat");
    std::ofstream fy_output(strBox+"fy.dat");
    std::ofstream fz_output(strBox+"fz.dat");
    assert(fx_output.is_open());
    assert(fy_output.is_open());
    assert(fz_output.is_open());
    fx_output.close();
    fy_output.close();
    fz_output.close();

    /** Create actual force outputs **/

    std::ofstream sx_output(strBox+"sx.dat");
    std::ofstream sy_output(strBox+"sy.dat");
    std::ofstream sz_output(strBox+"sz.dat");
    assert(sx_output.is_open());
    assert(sy_output.is_open());
    assert(sz_output.is_open());
    sx_output.close();
    sy_output.close();
    sz_output.close();

    /** Create divv and rho outputs **/

    std::ofstream divv_output(strBox+"divv.dat");
    assert(divv_output.is_open());
    divv_output.close();

    std::ofstream rho_output(strBox+"rho.dat");
    assert(rho_output.is_open());
    rho_output.close();

    std::ofstream p_output(strBox+"p.dat");
    assert(p_output.is_open());
    p_output.close();


    // Surface output
    std::ofstream surf_output(strBox+"surfPsi.dat");
    assert(surf_output.is_open());
    surf_output.close();		
  }


	
/********************************************
 *                                          *
 *         FS constants + 1st Nr            *
 *                                          *
 *   C1,C2: pointwise multiplication        *
 *          constants for Fourier Space     *
 *          LinOp (CrankNic/AdamsBash)      *
 *                                          *
 *   Nr_local: nonlinear terms (pre loop)   *
 *                                          *
 *******************************************/


  for ( i_local = 0; i_local < local_n0; i_local++ )
  {
    i = i_local + local_0_start;

    for ( j = 0; j < Ny; j++ ) {
    for ( k = 0; k < Nz; k++ )
    {
      index =  (i_local*Ny + j)*Nz + k;
	   
      mq2c[index] = pow(Vqx[i_local],2)+pow(Vqy[j],2)+pow(Vqz[k],2);
      opSH = alpha*pow(q02-mq2c[index],2);
      aLin[index] = ep - opSH;
      C1[index] = (2.0-dt*rho_m*aLin[index]);
      C2[index] = (1.5-dt*rho_m*aLin[index]);
	   
      mq2 = pow(Vsx[i_local],2)+pow(Vqy[j],2)+pow(Vqz[k],2);
      CM1x[index] = scale/(nu*mq2);
      CM2x[index] = Vsx[i_local]/mq2;

      mq2 = pow(Vqx[i_local],2)+pow(Vsy[j],2)+pow(Vqz[k],2);
      CM1y[index] = scale/(nu*mq2);
      CM2y[index] = Vsy[j]/mq2;

      mq2 = pow(Vqx[i_local],2)+pow(Vqy[j],2)+pow(Vsz[k],2);
      CM1z[index] = scale/(nu*mq2);
      CM2z[index] = Vsz[k]/mq2;		 
		 
    }}}


  /* Move psi_local to Fourier Space */

  trans_local = psi_local;
  fftw_execute(planCT);
  psiq_local = trans_local;


  // Compute Nr adding the substrate penalty * wall potential
  // Also, scale psi

  for ( i_local = 0; i_local < local_n0; i_local++ ){
  for ( j = 0; j < Ny; j++ ) {
  for ( k = 0; k < Nz; k++ )
  {
    index =  (i_local*Ny + j)*Nz + k;

    psiq_local[index] = scale*psiq_local[index];

  }}}

  trans_local = psiq_local;
  fftw_execute(iPlanCT);
  psi_local = trans_local;

  for ( i_local = 0; i_local < local_n0; i_local++ ){
  for ( j = 0; j < Ny; j++ ) {
  for ( k = 0; k < Nz; k++ )
  {
    index =  (i_local*Ny + j)*Nz + k;

    Nl_local[index] = beta*pow(psi_local[index],3)
      - gamma*pow(psi_local[index],5); // + psiNew_local[index]*wall[index];
  }}}
  //

  psiq_old_local = psiq_local; // first time step


 /* Move Nl_local to Fourier Space */
	
  trans_local = Nl_local;
  fftw_execute(planCT);
  Nl_local = trans_local;

  psi_old_local = psi_local;

  std::fill(divv_local.begin(),divv_local.end(),0);  

 /********************************************
  *                                          *
  *            Pre loop routine              *
  *                                          *
  *******************************************/

 /* Pre loop values, indices */

 //	sleep(5);

  L1 = 1.0;

  int countL1 = 0;

  int countSave = 0;

  limL1 = pow(10.0,-6);

  int nLoop = 0;

  MPI::COMM_WORLD.Barrier();

 /* Pre loop announcement */

  std::clock_t startcputime; 
  auto wcts = std::chrono::system_clock::now();

  if ( rank == 0 )
  {
    time_t now = time(0);
    char* dNow = ctime(&now);	   
    std::cout << "The pre loop local date and time is: " 
	      << dNow << std::endl;
    startcputime = std::clock();	
  }


    /** Empty out containers **/

    std::fill(divv_local.begin(),divv_local.end(),0);

    std::fill(psiGradx_local.begin(),psiGradx_local.end(),0);
    std::fill(psiGrady_local.begin(),psiGrady_local.end(),0);
    std::fill(psiGradz_local.begin(),psiGradz_local.end(),0);

    std::fill(rhoDx_local.begin(),rhoDx_local.end(),0);
    std::fill(rhoDy_local.begin(),rhoDy_local.end(),0);
    std::fill(rhoDz_local.begin(),rhoDz_local.end(),0);

    std::fill(velx_local.begin(),velx_local.end(),0);
    std::fill(vely_local.begin(),vely_local.end(),0);
    std::fill(velz_local.begin(),velz_local.end(),0);


 /********************************************
  *                                          *
  *   Time Loop (L1 as dynamics criterion)   *
  *                                          *
  *******************************************/

  //for(int tst=0;tst < 10;tst++) 
  while (L1 > limL1)
  {

    countL1++;
    nLoop++;

    /** Empty out containers **/

    std::fill(psiGradx_local.begin(),psiGradx_local.end(),0);
    std::fill(psiGrady_local.begin(),psiGrady_local.end(),0);
    std::fill(psiGradz_local.begin(),psiGradz_local.end(),0);

    std::fill(rhoDx_local.begin(),rhoDx_local.end(),0);
    std::fill(rhoDy_local.begin(),rhoDy_local.end(),0);
    std::fill(rhoDz_local.begin(),rhoDz_local.end(),0);

    /** Previous Nq_local is now NqPast_local  **/

    Nl_old_local = Nl_local;

    /** Compute dfDlapPsi  **/

    for ( i_local = 0; i_local < local_n0; i_local++ ){
    for ( j = 0; j < Ny; j++ ) {
    for ( k = 0; k < Nz; k++ )
    {

      index =  (i_local*(Ny) + j)*(Nz) + k;

      dfDlapPsi_local[index] = alpha*(q02-mq2c[index])*psiq_local[index];

      //      lapRhoDfDlapPsi_local[index] = 
      //	-mq2c[index]*dfDlapPsi_local[index];
    }}}	

    /** Move psi and derivatives back to real space **/

    trans_local = dfDlapPsi_local;
    fftw_execute(iPlanCT);
    dfDlapPsi_local = trans_local;

    // trans_local = lapRhoDfDlapPsi_local;
    // fftw_execute(iPlanCT);
    // lapRhoDfDlapPsi_local = trans_local;

    /** COMPUTE: gradients of psi and rho **/
    // partial_x psi (parallelized direction)

    i_local = 0;
	 
    for( j = 0; j < Ny; j++ ){
    for( k = 0; k < Nz; k++ ){

      index2 = (i_local*Ny + j) * Nz + k;
      index = j*Nz + k;

      psi_back[index] = psiq_local[index2];
      psi_back2[index] = rhoq_local[index2];
    }}

    if (rank == size-1){
	   
      MPI::COMM_WORLD.Send(psi_back.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);

      MPI::COMM_WORLD.Send(psi_back2.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);
	   
    } else if (rank % 2 != 0){

      MPI::COMM_WORLD.Send(psi_back.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);

      MPI::COMM_WORLD.Recv(psi_front.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);

      MPI::COMM_WORLD.Send(psi_back2.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);

      MPI::COMM_WORLD.Recv(psi_front2.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);

    } else if (rank != 0){

      MPI::COMM_WORLD.Recv(psi_front.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);

      MPI::COMM_WORLD.Send(psi_back.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);

      MPI::COMM_WORLD.Recv(psi_front2.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);

      MPI::COMM_WORLD.Send(psi_back2.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);

    } else {

      MPI::COMM_WORLD.Recv(psi_front.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);

      MPI::COMM_WORLD.Recv(psi_front2.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);

    }		 	 

    if (rank != size-1 ) 
    {
		   
      i_local = local_n0-1;
		 
      for ( j = 0; j < Ny; j++ ) {
      for ( k = 0; k < Nz; k++ )
      {		     
	index = (i_local*Ny + j) * Nz + k;	     
	index2 = j * Nz + k;

	psiGradx_local[index] = -Vsx[i_local]*psi_front[index2];
	rhoDx_local[index] = -Vsx[i_local]*psi_front2[index2];
      }}
    }
	 
    for ( i_local = 0; i_local < local_n0-1; i_local++ ) {
    for ( j = 0; j < Ny; j++ ) {
    for ( k = 0; k < Nz; k++ )
    {
      index = (i_local*Ny + j) * Nz + k;
      index2 = ((i_local+1)*Ny + j) * Nz + k;
	   
      psiGradx_local[index] = -Vsx[i_local]*psiq_local[index2];
      rhoDx_local[index] = -Vsx[i_local]*rhoq_local[index2];
    }}}

    // partial_y psi

    for ( i_local = 0; i_local < local_n0; i_local++ ){
    for ( j = 0; j < Ny-1; j++ ) {
    for ( k = 0; k < Nz; k++ )
    {
      index =  (i_local*(Ny) + j)*(Nz) + k;
      index2 =  (i_local*(Ny) + j+1)*(Nz) + k;
	 
      psiGrady_local[index] = -Vsy[j]*psiq_local[index2];
      rhoDy_local[index] = -Vsy[j]*rhoq_local[index2];
    }}}	

    // partial_z psi

    for ( i_local = 0; i_local < local_n0; i_local++ ){
    for ( j = 0; j < Ny; j++ ) {
    for ( k = 0; k < Nz-1; k++ )
    {
      index =  (i_local*(Ny) + j)*(Nz) + k;	 
      index2 =  (i_local*(Ny) + j)*(Nz) + k+1;	 
	    
      psiGradz_local[index] = -Vsz[k]*psiq_local[index2];
      rhoDz_local[index] = -Vsz[k]*rhoq_local[index2];	   
    }}}	

    
    /** COMPUTE: laplacians of gradients of psi **/
    
    // for ( i_local = 0; i_local < local_n0; i_local++ ){
    // for ( j = 0; j < Ny; j++ ) {
    // for ( k = 0; k < Nz; k++ )
    // {
    //   index =  (i_local*(Ny) + j)*(Nz) + k;

    //   mq2 = pow(Vsx[i_local],2)+pow(Vqy[j],2)+pow(Vqz[k],2);			 
    //   psiLapDx_local[index] = -mq2*psiGradx_local[index];

    //   mq2 = pow(Vqx[i_local],2)+pow(Vsy[j],2)+pow(Vqz[k],2);			 
    //   psiLapDy_local[index] = -mq2*psiGrady_local[index];

    //   mq2 = pow(Vqx[i_local],2)+pow(Vqy[j],2)+pow(Vsz[k],2);			 
    //   psiLapDz_local[index] = -mq2*psiGradz_local[index];	   
    // }}}	

    // Move grad psi to real space

    trans_local = psiGradx_local;
    fftw_execute(iPlanSTx);
    psiGradx_local = trans_local;

    trans_local = psiGrady_local;
    fftw_execute(iPlanSTy);
    psiGrady_local = trans_local;

    trans_local = psiGradz_local;
    fftw_execute(iPlanSTz);
    psiGradz_local = trans_local;

    // Move grad rho to real space

    trans_local = rhoDx_local;
    fftw_execute(iPlanSTx);
    rhoDx_local = trans_local;

    trans_local = rhoDy_local;
    fftw_execute(iPlanSTy);
    rhoDy_local = trans_local;

    trans_local = rhoDz_local;
    fftw_execute(iPlanSTz);
    rhoDz_local = trans_local;

    // Move laplacian grad psi to real space

    // trans_local = psiLapDx_local;
    // fftw_execute(iPlanSTx);
    // psiLapDx_local = trans_local;

    // trans_local = psiLapDy_local;
    // fftw_execute(iPlanSTy);
    // psiLapDy_local = trans_local;

    // trans_local = psiLapDz_local;
    // fftw_execute(iPlanSTz);
    // psiLapDz_local = trans_local;


    /** Density constitutive law **/

    for ( i_local = 0; i_local < local_n0; i_local++ ){
    for ( j = 0; j < Ny; j++ ) {
    for ( k = 0; k < Nz; k++ ) 
    {
      index =  (i_local*Ny + j)*Nz + k;

      i = i_local + local_0_start;
	   
      rho_local[index] = 
	//(static_cast<double>(i)/512)*
	kp*sqrt(q02*pow(psi_local[index],2)
		+pow(psiGradx_local[index],2)
		+pow(psiGrady_local[index],2)
		+pow(psiGradz_local[index],2))+rho_0;
	//+(1-static_cast<double>(i)/512)*kp*Amp;
      // if (i < 256 & k > 160){
      // 	rho_local[index] = rho_local[index]+0.5;
      // }
    }}}

    trans_local = rho_local;
    fftw_execute(planCT);
    rhoq_local = trans_local;

    for ( i_local = 0; i_local < local_n0; i_local++ ){
    for ( j = 0; j < Ny; j++ ) {
    for ( k = 0; k < Nz; k++ ) 
    {
      index =  (i_local*Ny + j)*Nz + k;
	   
      rhoq_local[index] = scale*rhoq_local[index]
	*exp(-1.57*1.57*mq2c[index]*2);	   // /2 *8  	       
    }}}

    // Move smooth rho to real space

    trans_local = rhoq_local;
    fftw_execute(iPlanCT);
    rho_local = trans_local;


    /* Compute divv and lapRhoDfDlapPsi */

    if (load == 1 && nLoop == 1) {
      rho_old_local = rho_local;
    }

    if (nLoop > divvSwitch || load == 1){
      for ( i_local = 0; i_local < local_n0; i_local++ ){
      for ( j = 0; j < Ny; j++ ) {
      for ( k = 0; k < Nz; k++ ) 
      {
	index =  (i_local*Ny + j)*Nz + k;

	divv_local[index] =
	  -((rho_local[index]-rho_old_local[index])/dt
	    + velx_local[index]*rhoDx_local[index]
	    + vely_local[index]*rhoDy_local[index]
	    + velz_local[index]*rhoDz_local[index]) 
	  / (0.5*(rho_local[index]+rho_old_local[index]));

      }}}
    }

    
    for ( i_local = 0; i_local < local_n0; i_local++ ){
    for ( j = 0; j < Ny; j++ ) {
    for ( k = 0; k < Nz; k++ ) 
    {
      index =  (i_local*Ny + j)*Nz + k;

      lapRhoDfDlapPsi_local[index] = 
	-mq2c[index]*alpha*(q02-mq2c[index])*psiq_local[index];

    }}}

    trans_local = lapRhoDfDlapPsi_local;
    fftw_execute(iPlanCT);
    lapRhoDfDlapPsi_local = trans_local;

    for ( i_local = 0; i_local < local_n0; i_local++ ){
    for ( j = 0; j < Ny; j++ ) {
    for ( k = 0; k < Nz; k++ ) 
    {
      index =  (i_local*Ny + j)*Nz + k;

      lapRhoDfDlapPsi_local[index] = 
	rho_local[index]*lapRhoDfDlapPsi_local[index];

    }}}


    rho_old_local = rho_local;
    
    trans_local = divv_local;
    fftw_execute(planCT);
    divv_local = trans_local;

	   
    /** Compute div T^r  and move it to Fourier Space **/
	   
    for ( i_local = 0; i_local < local_n0; i_local++ ){
    for ( j = 0; j < Ny; j++ ) {
    for ( k = 0; k < Nz; k++ ) 
    {

      index =  (i_local*Ny + j)*Nz + k;

      energy_local[index] = -0.5*ep*pow(psi_local[index],2)
	+0.5*pow(dfDlapPsi_local[index],2) 
	-0.25*beta*pow(psi_local[index],4)
	+(gamma/6)*pow(psi_local[index],6);

      mu_local[index] = lapRhoDfDlapPsi_local[index]+ rho_local[index]*
	(dfDlapPsi_local[index] -ep*psi_local[index] 
	 - beta*pow(psi_local[index],3) + gamma*pow(psi_local[index],5));

      Sx_local[index] =	mu_local[index]*psiGradx_local[index] + 
	0*energy_local[index]*rhoDx_local[index];

      Sy_local[index] = mu_local[index]*psiGrady_local[index] + 
	0*energy_local[index]*rhoDy_local[index];

      Sz_local[index] = mu_local[index]*psiGradz_local[index] + 
	0*energy_local[index]*rhoDz_local[index];

    }}}

    Sx2_local = Sx_local;
    Sy2_local = Sy_local;
    Sz2_local = Sz_local;

    trans_local = Sx_local;
    fftw_execute(planSTx);
    Sx_local = trans_local;

    trans_local = Sy_local;
    fftw_execute(planSTy);
    Sy_local = trans_local;

    trans_local = Sz_local;
    fftw_execute(planSTz);
    Sz_local = trans_local;

    // Note: planSTx moves modes +1 in x etc. 
    // Hence, for computing velx I need to move Sy_local +1 in x
    // and by -1 in y.

    i_local = local_n0-1;
	 
    for( j = 0; j < Ny; j++ ){
    for( k = 0; k < Nz; k++ ){

      index2 = (i_local*Ny + j) * Nz + k;
      index = j*Nz + k;

      psi_front[index] = Sx_local[index2];
    }}

    if (rank == 0){
      
      MPI::COMM_WORLD.Send(psi_front.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);
	   
    } else if (rank % 2 == 0){

      MPI::COMM_WORLD.Send(psi_front.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);

      MPI::COMM_WORLD.Recv(psi_back.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);

    } else if (rank != size-1){

      MPI::COMM_WORLD.Recv(psi_back.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);

      MPI::COMM_WORLD.Send(psi_front.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);

    } else {

      MPI::COMM_WORLD.Recv(psi_back.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);
    }		 	 

    if (rank == 0 ) 
    {
      std::fill(psi_back.begin(),psi_back.end(),0);
    }
	 
    for ( i_local = 0; i_local < local_n0; i_local++ ) {
    for ( j = 0; j < Ny; j++ ) {
    for ( k = 0; k < Nz; k++ )
    {
      index = (i_local*Ny + j) * Nz + k;
      i = i_local + local_0_start;

      if (i_local > 0){
	index2 = ((i_local-1)*Ny + j) * Nz + k;
	Sx = Sx_local[index2];
      } else {
	index2 = j * Nz + k;
	Sx = psi_back[index2];
      }

      if (j > 0){
	index2 = (i_local*Ny + j-1) * Nz + k;
	Sy = Sy_local[index2];
      } else {
	Sy = 0;
      }

      if (k > 0){
	Sz = Sz_local[index-1];
      } else {
	Sz = 0;
      }
	     
      if (i+j+k != 0){
	p_local[index] = 
	  scale*(-(Vqx[i_local]*Sx+Vqy[j]*Sy+Vqz[k]*Sz)/mq2c[index]
		 +(lambda+2*nu)*divv_local[index]);
      } else {
	p_local[index] = 0;
      }
    }}}

    trans_local = p_local;
    fftw_execute(iPlanCT);
    p_local = trans_local;

    /** COMPUTE: Velocity field **/
    // A. velx (parallelized direction)

    /** Empty out velocity containers **/

    std::fill(vsolx_local.begin(),vsolx_local.end(),0);
    std::fill(vsoly_local.begin(),vsoly_local.end(),0);
    std::fill(vsolz_local.begin(),vsolz_local.end(),0);

    std::fill(velx_local.begin(),velx_local.end(),0);
    std::fill(vely_local.begin(),vely_local.end(),0);
    std::fill(velz_local.begin(),velz_local.end(),0);

    std::fill(fx_local.begin(),fx_local.end(),0);
    std::fill(fy_local.begin(),fy_local.end(),0);
    std::fill(fz_local.begin(),fz_local.end(),0);

    // Send Sz and Sy i_local=0 data to previous rank

    i_local = 0;
	 
    for( j = 0; j < Ny; j++ ){
    for( k = 0; k < Nz; k++ ){

      index = j*Nz + k;

      if ( j != 0 ) {
	index2 = (i_local*Ny + j-1) * Nz + k;
	psi_back[index] = Sy_local[index2];
      } else {
	psi_back[index] = 0;
      }

      if ( k != 0 ) {
	index2 = (i_local*Ny + j) * Nz + k-1;
	psi_back2[index] = Sz_local[index2];
      } else {
	psi_back2[index] = 0;

      }

      index2 = (i_local*Ny + j) * Nz + k;
      psi_back3[index] = divv_local[index2];
    }}

    if (rank == size-1){
	   
      MPI::COMM_WORLD.Send(psi_back.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);

      MPI::COMM_WORLD.Send(psi_back2.data(),Nslice,
			   MPI::DOUBLE,rank-1,1);

      MPI::COMM_WORLD.Send(psi_back3.data(),Nslice,
			   MPI::DOUBLE,rank-1,2);
	   
    } else if (rank % 2 != 0){

      MPI::COMM_WORLD.Send(psi_back.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);

      MPI::COMM_WORLD.Recv(psi_front.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);

      MPI::COMM_WORLD.Send(psi_back2.data(),Nslice,
			   MPI::DOUBLE,rank-1,1);

      MPI::COMM_WORLD.Recv(psi_front2.data(),Nslice,
			   MPI::DOUBLE,rank+1,1);

      MPI::COMM_WORLD.Send(psi_back3.data(),Nslice,
			   MPI::DOUBLE,rank-1,2);

      MPI::COMM_WORLD.Recv(psi_front3.data(),Nslice,
			   MPI::DOUBLE,rank+1,2);
  
    } else if (rank != 0){

      MPI::COMM_WORLD.Recv(psi_front.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);

      MPI::COMM_WORLD.Send(psi_back.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);

      MPI::COMM_WORLD.Recv(psi_front2.data(),Nslice,
			   MPI::DOUBLE,rank+1,1);

      MPI::COMM_WORLD.Send(psi_back2.data(),Nslice,
			   MPI::DOUBLE,rank-1,1);

      MPI::COMM_WORLD.Recv(psi_front3.data(),Nslice,
			   MPI::DOUBLE,rank+1,2);

      MPI::COMM_WORLD.Send(psi_back3.data(),Nslice,
			   MPI::DOUBLE,rank-1,2);

    } else {

      MPI::COMM_WORLD.Recv(psi_front.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);
      
      MPI::COMM_WORLD.Recv(psi_front2.data(),Nslice,
			   MPI::DOUBLE,rank+1,1);

      MPI::COMM_WORLD.Recv(psi_front3.data(),Nslice,
			   MPI::DOUBLE,rank+1,2);
    }		 	 

    // Use front data to compute velx at i_local=local_n0-1

    i_local = local_n0-1;
    if (rank != size-1 ) 
    {		   		 
      for ( j = 0; j < Ny; j++ ) {
      for ( k = 0; k < Nz; k++ )
      {
	index = (i_local*Ny + j) * Nz + k;
	index2 = j * Nz + k;
	
	Sy = psi_front[index2];

	Sz = psi_front2[index2];
		     
	dotSqVq = Vsx[i_local]*Sx_local[index] + Vqy[j]*Sy + Vqz[k]*Sz;

	vsolx_local[index] = 
	  CM1x[index]*(Sx_local[index] - CM2x[index]*dotSqVq);
	     
	velx_local[index] = vsolx_local[index]+
	  CM1x[index]*(nu*Vsx[i_local]*psi_front3[index2]);

	fx_local[index] = scale*(Sx_local[index] - CM2x[index]*dotSqVq);

      }}
    } 	 

    // Compute velx for the rest

    // 1. Case j > 0, k > 0

    for ( i_local = 0; i_local < local_n0-1; i_local++ ) {
    for ( j = 1; j < Ny; j++ ) {
    for ( k = 1; k < Nz; k++ )
    {
      index = (i_local*Ny + j) * Nz + k;

      index2 = ((i_local+1)*Ny + j-1) * Nz + k;
      Sy = Sy_local[index2];

      index2 = ((i_local+1)*Ny + j) * Nz + k-1;
      Sz = Sz_local[index2];

      dotSqVq = Vsx[i_local]*Sx_local[index] + Vqy[j]*Sy + Vqz[k]*Sz;

      vsolx_local[index] =
	CM1x[index]*(Sx_local[index] - CM2x[index]*dotSqVq);

      velx_local[index] = vsolx_local[index]+
	CM1x[index]*(nu*Vsx[i_local]*divv_local[index+Ny*Nz]);

      fx_local[index] = scale*(Sx_local[index] - CM2x[index]*dotSqVq);

    }}}


    // 2. Case j = 0, k > 0

    j = 0;

    for ( i_local = 0; i_local < local_n0-1; i_local++ ) {
    for ( k = 1; k < Nz; k++ )
    {
      index = (i_local*Ny + j) * Nz + k;

      // Sy = 0;

      index2 = ((i_local+1)*Ny + j) * Nz + k-1;
      Sz = Sz_local[index2];
	     
      dotSqVq = Vsx[i_local]*Sx_local[index] + Vqz[k]*Sz;	   	     

      vsolx_local[index] = 
	CM1x[index]*(Sx_local[index] - CM2x[index]*dotSqVq);

      velx_local[index] = vsolx_local[index]+
	CM1x[index]*(nu*Vsx[i_local]*divv_local[index+Ny*Nz]);

      fx_local[index] = scale*(Sx_local[index] - CM2x[index]*dotSqVq);

    }}

    // 3. Case j > 0, k = 0

    k = 0;

    for ( i_local = 0; i_local < local_n0-1; i_local++ ) {
    for ( j = 1; j < Ny; j++ )
    {
      index = (i_local*Ny + j) * Nz + k;

      index2 = ((i_local+1)*Ny + j-1) * Nz + k;
      Sy = Sy_local[index2];	   
	   
      //Sz = 0;	   
	     
      dotSqVq = Vsx[i_local]*Sx_local[index] + Vqy[j]*Sy;

      vsolx_local[index] = 
	CM1x[index]*(Sx_local[index] - CM2x[index]*dotSqVq);
	   	     
      velx_local[index] = vsolx_local[index] +
	CM1x[index]*(nu*Vsx[i_local]*divv_local[index+Ny*Nz]);

      fx_local[index] = scale*(Sx_local[index] - CM2x[index]*dotSqVq);
    
    }}


    // 4. Case j = 0, k = 0

    j = 0;
    k = 0;

    for ( i_local = 0; i_local < local_n0-1; i_local++ )
    {
      index = (i_local*Ny + j) * Nz + k;
	    
      //Sy = 0;
	   
      //Sz = 0;	   
	     
      dotSqVq = Vsx[i_local]*Sx_local[index];

      vsolx_local[index] = 
	CM1x[index]*(Sx_local[index] - CM2x[index]*dotSqVq);
	   	     
      velx_local[index] = vsolx_local[index] +
	CM1x[index]*(nu*Vsx[i_local]*divv_local[index+Ny*Nz]);
    
      fx_local[index] = scale*(Sx_local[index] - CM2x[index]*dotSqVq);

    }

    // B. vely and velz 

    // Send i_local=local_n0-1 Sx data to back rank 

    std::fill(psi_back.begin(),psi_back.end(),0);

    i_local = local_n0-1;
	 
    for( j = 0; j < Ny; j++ ){
    for( k = 0; k < Nz; k++ )
    {
      index = j*Nz + k;
      index2 = (i_local*Ny + j) * Nz + k;

      psi_front[index] = Sx_local[index];	       
    }}
    
    if (rank == size-1){
	   
      MPI::COMM_WORLD.Recv(psi_back.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);
	   
    } else if (rank % 2 != 0){

      MPI::COMM_WORLD.Recv(psi_back.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);

      MPI::COMM_WORLD.Send(psi_front.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);

    } else if (rank != 0){

      MPI::COMM_WORLD.Send(psi_front.data(),Nslice,
				  MPI::DOUBLE,rank+1,0);
      
      MPI::COMM_WORLD.Recv(psi_back.data(),Nslice,
			   MPI::DOUBLE,rank-1,0);

    } else {

      MPI::COMM_WORLD.Send(psi_front.data(),Nslice,
			   MPI::DOUBLE,rank+1,0);
    }		 	 
    
    // Compute vely and velz

    // use back data to compute vel_y and vel_z for i_local = 0

    i_local = 0;
    for ( j = 0; j < Ny-1; j++ ) {
    for ( k = 0; k < Nz-1; k++ )
    {
      index = (i_local*Ny + j) * Nz + k;

      index2 = (j+1)*Nz+k;
      Sx = psi_back[index2];

      if ( k !=0) {
	index2 = ((i_local)*Ny + j+1) * Nz + k-1;
	Sz = Sz_local[index2];
      } else {
	Sz = 0;
      }
	     
      dotSqVq = Vqx[i_local]*Sx + Vsy[j]*Sy_local[index] + Vqz[k]*Sz;

      vsoly_local[index] = 
	CM1y[index]*(Sy_local[index] - CM2y[index]*dotSqVq);
	   	     
      vely_local[index] = vsoly_local[index] + 
	CM1y[index]*(nu*Vsy[j]*divv_local[index+Nz]);
	   
      fy_local[index] = scale*(Sy_local[index] - CM2y[index]*dotSqVq);

      index2 = j*Nz+k+1;
      Sx = psi_back[index2];	     	     

      if ( j !=0) {
	index2 = ((i_local)*Ny + j-1) * Nz + k+1;
	Sy = Sy_local[index2];
      } else {
	Sy = 0;
      }
	     
      dotSqVq = Vqx[i_local]*Sx + Vqy[j]*Sy + Vsz[k]*Sz_local[index];

      vsolz_local[index] = 
	CM1z[index]*(Sz_local[index] - CM2z[index]*dotSqVq);
	   	     
      velz_local[index] = vsolz_local[index]+
	CM1z[index]*(nu*Vsz[k]*divv_local[index+1]);

      fz_local[index] = scale*(Sz_local[index] - CM2z[index]*dotSqVq);

    }}

    // Compute vely and velz for the rest
    
    // 1. Case j > 0, k > 0

    for ( i_local = 1; i_local < local_n0; i_local++ ) {
    for ( j = 1; j < Ny-1; j++ ) {
    for ( k = 1; k < Nz-1; k++ )
    {
      index = (i_local*Ny + j) * Nz + k;

      // vely

      index2 = ((i_local-1)*Ny + j+1) * Nz + k;
      Sx = Sx_local[index2];

      index2 = ((i_local)*Ny + j+1) * Nz + k-1;
      Sz = Sz_local[index2];
	     
      dotSqVq = Vqx[i_local]*Sx + Vsy[j]*Sy_local[index] + Vqz[k]*Sz;	   

      vsoly_local[index] = 
	CM1y[index]*(Sy_local[index] - CM2y[index]*dotSqVq);
	     
      vely_local[index] = vsoly_local[index] +
	CM1y[index]*(nu*Vsy[j]*divv_local[index+Nz]);

      fy_local[index] = scale*(Sy_local[index] - CM2y[index]*dotSqVq);

      // velz

      index2 = ((i_local-1)*Ny + j) * Nz + k+1;
      Sx = Sx_local[index2];

      index2 = ((i_local)*Ny + j-1) * Nz + k+1;
      Sy = Sy_local[index2];
	     
      dotSqVq = Vqx[i_local]*Sx + Vqy[j]*Sy + Vsz[k]*Sz_local[index];	   

      vsolz_local[index] = 
	CM1z[index]*(Sz_local[index] - CM2z[index]*dotSqVq);
	     
      velz_local[index] = vsolz_local[index] +
	CM1z[index]*(nu*Vsz[k]*divv_local[index+1]);

      fz_local[index] = scale*(Sz_local[index] - CM2z[index]*dotSqVq);

    }}}

    // 2. Case j = 0, k > 0

    j = 0;

    for ( i_local = 1; i_local < local_n0; i_local++ ) {
    for ( k = 1; k < Nz-1; k++ )
    {
      index = (i_local*Ny + j) * Nz + k;

      // vely

      index2 = ((i_local-1)*Ny + j+1) * Nz + k;
      Sx = Sx_local[index2];
	    
      index2 = ((i_local)*Ny + j+1) * Nz + k-1;
      Sz = Sz_local[index2];
  	     
      dotSqVq = Vqx[i_local]*Sx + Vsy[j]*Sy_local[index] + Vqz[k]*Sz;	   

      vsoly_local[index] = 
	CM1y[index]*(Sy_local[index] - CM2y[index]*dotSqVq);
	     
      vely_local[index] = vsoly_local[index] +
	CM1y[index]*(nu*Vsy[j]*divv_local[index+Nz]);

      fy_local[index] = scale*(Sy_local[index] - CM2y[index]*dotSqVq);

      // velz , Sy = 0;

      index2 = ((i_local-1)*Ny + j) * Nz + k+1;
      Sx = Sx_local[index2];
	     
      dotSqVq = Vqx[i_local]*Sx + Vsz[k]*Sz_local[index];	   

      vsolz_local[index] = 
	CM1z[index]*(Sz_local[index] - CM2z[index]*dotSqVq);
	     
      velz_local[index] = vsolz_local[index] +
	CM1z[index]*(nu*Vsz[k]*divv_local[index+1]);

      fz_local[index] = scale*(Sz_local[index] - CM2z[index]*dotSqVq);

    }}
    
    // 3. Case j > 0, k = 0

    k = 0;

    for ( i_local = 1; i_local < local_n0; i_local++ ) {
    for ( j = 1; j < Ny-1; j++ )
    {
      index = (i_local*Ny + j) * Nz + k;

      // vely, Sz = 0;

      index2 = ((i_local-1)*Ny + j+1) * Nz + k;
      Sx = Sx_local[index2];
      
      dotSqVq = Vqx[i_local]*Sx + Vsy[j]*Sy_local[index];	   

      vsoly_local[index] = 
	CM1y[index]*(Sy_local[index] - CM2y[index]*dotSqVq);
      
      vely_local[index] = vsoly_local[index] +
	CM1y[index]*(nu*Vsy[j]*divv_local[index+Nz]);
      
      fy_local[index] = scale*(Sy_local[index] - CM2y[index]*dotSqVq);

      // velz

      index2 = ((i_local-1)*Ny + j) * Nz + k+1;
      Sx = Sx_local[index2];
	 
      index2 = ((i_local)*Ny + j-1) * Nz + k+1;
      Sy = Sy_local[index2];
	     
      dotSqVq = Vqx[i_local]*Sx + Vqy[j]*Sy + Vsz[k]*Sz_local[index];	   

      vsolz_local[index] = 
	CM1z[index]*(Sz_local[index] - CM2z[index]*dotSqVq);
	     
      velz_local[index] = vsolz_local[index] +
	CM1z[index]*(nu*Vsz[k]*divv_local[index+1]);

      fz_local[index] = scale*(Sz_local[index] - CM2z[index]*dotSqVq);

    }}


    // 4. Case j = 0, k = 0

    j = 0;
    k = 0;

    for ( i_local = 1; i_local < local_n0; i_local++ )
    {
      index = (i_local*Ny + j) * Nz + k;

      // vely, Sz = 0;

      index2 = ((i_local-1)*Ny + j+1) * Nz + k;
      Sx = Sx_local[index2];
	     
      dotSqVq = Vqx[i_local]*Sx + Vsy[j]*Sy_local[index];	   

      vsoly_local[index] = 
	CM1y[index]*(Sy_local[index] - CM2y[index]*dotSqVq);
	     
      vely_local[index] = vsoly_local[index] +
	CM1y[index]*(nu*Vsy[j]*divv_local[index+Nz]);

      fy_local[index] = scale*(Sy_local[index] - CM2y[index]*dotSqVq);

      // velz, Sy = 0;	   

      index2 = ((i_local-1)*Ny + j) * Nz + k+1;
      Sx = Sx_local[index2];	  
	     
      dotSqVq = Vqx[i_local]*Sx + Vsz[k]*Sz_local[index];	   

      vsolz_local[index] = 
	CM1z[index]*(Sz_local[index] - CM2z[index]*dotSqVq);
	     
      velz_local[index] = vsolz_local[index] +
	CM1z[index]*(nu*Vsz[k]*divv_local[index+1]);
   
      fz_local[index] = scale*(Sz_local[index] - CM2z[index]*dotSqVq);

    }

    // solenoidal + irrotational velocity

    trans_local = velx_local;
    fftw_execute(iPlanSTx);
    velx_local = trans_local;
		 
    trans_local = vely_local;
    fftw_execute(iPlanSTy);
    vely_local = trans_local;

    trans_local = velz_local;
    fftw_execute(iPlanSTz);
    velz_local = trans_local;


    /* COMPUTE: CURRENT Nr_local (S)*/

    for ( i_local = 0; i_local < local_n0; i_local++ ){

      i = i_local + local_0_start;

      for ( j = 0; j < Ny; j++ ) {
      for ( k = 0; k < Nz; k++ ) 
      {
	index =  (i_local*Ny + j)*Nz + k;

	Nl_local[index] = 
	  // rho_local[index]*
	  // (ep*psi_local[index]-dfDlapPsi_local[index] 
	  //  + beta*pow(psi_local[index],3) - gamma*pow(psi_local[index],5))
	  // - lapRhoDfDlapPsi_local[index]
	  - mu_local[index]
	  - velx_local[index]*psiGradx_local[index]
	  - vely_local[index]*psiGrady_local[index]
	  - velz_local[index]*psiGradz_local[index];

	if (nLoop > divvSwitch + 1 ){

	  Nl_local[index] +=
	    (p_local[index]+rho_local[index]*energy_local[index])*kp*
	    sqrt(pow(rhoDx_local[index],2)+pow(rhoDy_local[index],2)
		 +pow(rhoDz_local[index],2))/pow(rho_local[index],2);   // or rho_s if low rho_a
	}
      }}}
    
    /* Obtain current Nq_local */

    trans_local = Nl_local;
    fftw_execute(planCT);	
    Nl_local = trans_local;

    if (nLoop == 1){
      Nl_old_local = Nl_local;
    }

 
    /* COMPUTE: NEW PSI IN FOURIER SPACE (CN/AB scheme) */

    trans_local = psiq_local;

    for ( i_local = 0; i_local < local_n0; i_local++ ){
    for ( j = 0; j < Ny; j++ ) {
    for ( k = 0; k < Nz; k++ )
    {
      index =  (i_local*Ny + j)*Nz + k;

      psiq_local[index] = 
	(C1[index]*psiq_local[index] - 0.5*psiq_old_local[index]
	 + dtd2*scale*(3.0*Nl_local[index]-Nl_old_local[index]))/C2[index];
    }}}	
		 
    psiq_old_local = trans_local;

    /** Obtain new psi in real space **/

    psi_old_local = psi_local;

    trans_local = psiq_local;
    fftw_execute(iPlanCT);
    psi_local = trans_local;
		 
    /* COMPUTE: L1 (under count condition) */
		 
    if ( countL1 == stepL1 ) //50
    {

      sumA_local = 0.0; sumB_local = 0.0;
      sumA = 0.0;       sumB = 0.0;

      rho_sum_local = 0.0;
      rho_sum = 0.0;

      energy_sum = 0.0;
      energy = 0.0;

      for ( i_local = 0; i_local < local_n0; i_local++ ) {
      for ( j = 0; j < Ny; j++ ) {
      for ( k = 0; k < Nz; k++ )
      {
	index = (i_local*Ny + j) * Nz + k;
		   
	sumA_local = sumA_local  
	  + fabs(psi_local[index] - psi_old_local[index]);
		     
	sumB_local = sumB_local + fabs(psi_local[index]);

	rho_sum_local = rho_sum_local + rho_local[index];

	energy_sum = energy_sum + energy_local[index];

      }}}

      MPI::COMM_WORLD.Reduce(&sumA_local,&sumA,1,MPI::DOUBLE,MPI::SUM,0);
      MPI::COMM_WORLD.Reduce(&sumB_local,&sumB,1,MPI::DOUBLE,MPI::SUM,0);

      MPI::COMM_WORLD.Reduce(&rho_sum_local,&rho_sum,1,MPI::DOUBLE,MPI::SUM,0);

      MPI::COMM_WORLD.Reduce(&energy_sum,&energy,1,MPI::DOUBLE,MPI::SUM,0);

      if ( rank == 0)
      {
	L1 = sumA/(dt*sumB);
	L1_output.open(strBox+"L1.dat",std::ios_base::app); // append result
	assert(L1_output.is_open());
	L1_output << L1 << "\n";
	L1_output.close();

	rho_sum = rho_sum*dx*dx*dx;
	mass_output.open(strBox+"mass.dat",std::ios_base::app); // append result
	assert(mass_output.is_open());
	mass_output << rho_sum << "\n";
	mass_output.close();	

	energy = energy*dx*dx*dx;
	energy_output.open(strBox+"energy.dat",std::ios_base::app); // append result
	assert(energy_output.is_open());
	energy_output << energy << "\n";
	energy_output.close();	
      }

      MPI::COMM_WORLD.Bcast(&L1,1,MPI::DOUBLE,0);

      countL1 = 0;

      countSave++;

      /* SAVE PSI & OBTAIN SURFACE INFO (under count condition) */

      if ( countSave == stepSave ) // 4
      { 

	// Surface Track

	for ( i_local = 0; i_local < local_n0; i_local++ ) {
	for ( j = 0; j < Ny; j++ ) 
	{	
	  track = 0;
	  index2 = i_local*Ny + j;

	  for ( k = Nz-1; k > -1; k-- ) 
	  {

	    index = (i_local*Ny + j) * Nz + k;

	    if ( rho_local[index] > rho_0 + 0.2 )
	    {
	      surfZ_local[index2] = k;
	      break;
	    }

	    // 0.7 : results are better when looking for this 0
	    // if ( psi_local[index] > 0.7 & track == 0 ) 
	    // {
	    //   track = 1;
	    // }
	    // if ( psi_local[index] < 0.0 & track == 1 ) //std::abs(...) > 0.7
	    // {
	    //   k2 = k;

	    //   if( std::abs(psi_local[index]) > std::abs(psi_local[index+1]) ) // >
	    //   {
	    // 	index = index + 1;
	    // 	k2 = k + 1;
	    //   }
		    
	    //   surfZ_local[index2] = k2;
				
	    //   track = 2;
	    // }
	  }}
	}

	MPI::COMM_WORLD.Gather(surfZ_local.data(),alloc_surf,MPI::DOUBLE,
			       surfZ.data(),alloc_surf, MPI::DOUBLE,0);

	// end surface track

			 			 	
	j = Ny/2;
	for( k = 0; k < Nz ; k++ ){
	for( i_local = 0; i_local < local_n0 ; i_local++ ){
	  index  = (i_local*Ny +j)*Nz + k;
	  index2 = i_local*Nz + k;
	  psiSlice_local[index2] = psi_local[index];
	}}

	MPI::COMM_WORLD.Gather(psiSlice_local.data(),alloc_slice,MPI::DOUBLE,
			       psiSlice.data(),alloc_slice, MPI::DOUBLE,0);

	std::ofstream psi_output(strPsi.c_str());
	assert(psi_output.is_open());

	for ( i_local = 0; i_local < local_n0; i_local++ ){
	for ( j = 0; j < Ny; j++ ) {
	for ( k = 0; k < Nz; k++ ) 
	{
	  index  = (i_local*Ny +j)*Nz + k;
	  psi_output << psi_local[index] << "\n";
	}}}

	psi_output.close();

	/** rank 0 outputs **/

	if (rank == 0 )
	{	  	  	        
	  psiMid_output.open(strBox+"psiMid.dat",std::ios_base::app);					
	  assert(psiMid_output.is_open());

	  for ( i = 0; i < Nx; i++ ) {
	  for ( k = 0; k < Nz; k++ ) 
	  {
	    index = i*Nz + k;

	    psiMid_output << psiSlice[index] << "\n";
	  }}

	  psiMid_output.close();	


	  surf_output.open(strBox+"surfPsi.dat",std::ios_base::app);					
	  assert(surf_output.is_open());

	  for ( i = 0; i < Nx; i++ ) {
	  for ( j = 0; j < Ny; j++ ) 
	  {
	    index = i*Ny + j;

	    surf_output << surfZ[index] << "\n";
	  }}

	  surf_output.close();

	  /** Inform date and time after each save psi **/

	  time_t now = time(0);
	  char* dNow = ctime(&now);    		   

	  std::cout << "The loop " << nLoop 
		    << " local date and time is: " << dNow << std::endl;
					
	} // ends rank 0 psiMid output




	// Save velocity for mid x-section

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		velx_local, vx_output, strBox, "vx.dat");

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		vely_local, vy_output, strBox, "vy.dat");

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		velz_local, vz_output, strBox, "vz.dat");


	// solenoidal part of the velocity

	trans_local = vsolx_local;
	fftw_execute(iPlanSTx);
	vsolx_local = trans_local;
		 
	trans_local = vsoly_local;
	fftw_execute(iPlanSTy);
	vsoly_local = trans_local;

	trans_local = vsolz_local;
	fftw_execute(iPlanSTz);
	vsolz_local = trans_local;


	// force projection filtered by pressure

	trans_local = fx_local;
	fftw_execute(iPlanSTx);
	fx_local = trans_local;
		 
	trans_local = fy_local;
	fftw_execute(iPlanSTy);
	fy_local = trans_local;

	trans_local = fz_local;
	fftw_execute(iPlanSTz);
	fz_local = trans_local;


	// Just for saving divv

	for ( i_local = 0; i_local < local_n0; i_local++ ){
	for ( j = 0; j < Ny; j++ ) {
	for ( k = 0; k < Nz; k++ ) 
	{
	  index =  (i_local*Ny + j)*Nz + k;
	   
	  divv_local[index] = scale*divv_local[index];

	}}}

	trans_local = divv_local;
	fftw_execute(iPlanCT);
	divv_local = trans_local;

	// Save solenoidal velocity

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		vsolx_local, vsolx_output, strBox, "vsolx.dat");

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		vsoly_local, vsoly_output, strBox, "vsoly.dat");

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		vsolz_local, vsolz_output, strBox, "vsolz.dat");

	// Save projected force

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		fx_local, fx_output, strBox, "fx.dat");

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		fy_local, fy_output, strBox, "fy.dat");

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		fz_local, fz_output, strBox, "fz.dat");


	// Save actual force

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		Sx2_local, sx_output, strBox, "sx.dat");

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		Sy2_local, sy_output, strBox, "sy.dat");

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		Sz2_local, sz_output, strBox, "sz.dat");

	// Save divergence of the  velocity

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		divv_local, divv_output, strBox, "divv.dat");

	// Save pressure

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		p_local, p_output, strBox, "p.dat");

	// Save pressure

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		rho_local, rho_output, strBox, "rho.dat");

	// Save energy

	saveMid(Nx, Ny, Nz, local_n0, rank, alloc_slice, psiSlice_local, psiSlice,
		energy_local, energyMid_output, strBox, "energyMid.dat");


	MPI::COMM_WORLD.Barrier();
		
	countSave = 0;

      } // End: countSave block
		
    } // End: countL1 block

  } // End: time loop

/********************************************
 *                                          *
 *         Post Time Loop routine           *
 *                                          *
 *******************************************/

  if ( rank == 0 )
  {
    time_t now = time(0);
    char* dNow = ctime(&now);	   
    std::cout << "The post loop local date and time is: " 
	      << dNow << std::endl;
    double cpu_duration = (std::clock() - startcputime) / (double)CLOCKS_PER_SEC;
    std::cout << "Finished in " << cpu_duration <<
      " seconds [CPU Clock] " << std::endl;
    std::chrono::duration<double> wctduration =
      (std::chrono::system_clock::now() - wcts);
    std::cout << "Finished in " << wctduration.count()
	      << " seconds [Wall Clock] " << std::endl;
  }

	
  /** Destroy FFTW plans, cleanup **/

  fftw_destroy_plan(planCT);
  fftw_destroy_plan(iPlanCT);
  fftw_destroy_plan(planSTx);
  fftw_destroy_plan(iPlanSTx);
  fftw_destroy_plan(planSTy);
  fftw_destroy_plan(iPlanSTy);
  fftw_destroy_plan(planSTz);
  fftw_destroy_plan(iPlanSTz);
	
  fftw_cleanup();
	
/* Finalize MPI */

  MPI::Finalize();

} // END
