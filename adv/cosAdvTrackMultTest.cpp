/********************************************
 *                                          *
 *         cosAdvTrackMultTest.cpp          *
 *                                          *
 *     SmecticA 3D Phase Field              *
 *     FFTW in parallel                     *
 *     cos: DCT (and also DST!)             *
 *     Adv: Advection is on                 *
 *     Curv: Save curvatures after X steps  *
 *     Mult: Better parallelization         *
 *     in terms of memory usage             *                         
 *     Test: this one has the correct       *
 *     operations for sin/cos transforms.   *
 *     It is only for testing and           *
 *     reference, don't mess with it!       *
 *                                          *
 *     Last mod: 12/07/2018                 *
 *     Author: Eduardo Vitral               *
 *                                          *
 ********************************************/

/* General */

#include <vector>
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
module load intel/2016.3.210 mvapich2_ib 

mpicxx -I /opt/fftw/3.3.4/intel/mvapich2_ib/include -O2 -o code code.cpp 
-L /opt/fftw/3.3.4/intel/mvapich2_ib/lib -lfftw3_mpi -lfftw3 -lm -std=c++11 

********************************************/

/********************************************
 *                                          *
 *               FUNCTIONS                  *
 *                                          *
 *******************************************/    

// If the wall potential for the substrate is on, go 
// to the Time Loop and change the 3 sections with (S)

double wallPotential(int z)
{
	double wP;
	double sigma = 0.0; // 1
	double z0 = 0.0001; //0.0001;
	wP = sigma*exp(-z/z0);
	return wP;
}


/********************************************
 *                                          *
 *                 MAIN                     *
 *                                          *
 *******************************************/    

int main(int argc, char* argv[]) {

/* FFTW plans */

	fftw_plan planPsi, iPlanPsi, planN,
	  planSTx, planSTy, planSTz, iPlanSTx, iPlanSTy, iPlanSTz, iPlanCT;
		// iPlanPsiDxx, iPlanPsiDyy, iPlanPsiDzz, iPlanDTpsi, iPlanSp2,
		// planSx, planSy, planSz, iPlanSx, iPlanSy, iPlanSz;

/* Indices and mpi related numbers */

	int i, j, k, index, i_local, size;

	long long rank;

/* Fourier space doubles */

	double mq2, opSH, dotSqVq;

	double Sx, Sy, Sz;

/* L1 related doubles + output */

	double L1, limL1, sumA, sumA_local, sumB, sumB_local;

	std::ofstream L1_output;

/* Ints and doubles for surface info */

	int index1, index2, index3, index4, index5, track, k2;

	double psiDxy, psiDxz, psiDyz, gradVal;

	double psiDxxx, psiDxxy, psiDxxz, psiDyyx, psiDyyy, psiDyyz, psiDzzx,
		psiDzzy, psiDzzz;
	
/* Load/save parameters */

	int load = atof(argv[5]);  // (load YES == 1)

	int swtPsi = 0;  // (switch: psi.dat/psiB.dat)

	std::string strPsi = "psi";
	
	std::string strLoad = "/home/vinals/vitra002/smectic/results/adv-nu";
	
	strLoad += argv[1] + std::string("-e0d") + argv[2] 
	  + std::string("-r") + argv[3] + std::string("/save/");

	std::ofstream psiMid_output, surf_output, velS_output, 
	  curvH_output, curvK_output, sx_output, sy_output, sz_output;

	std::string strBox = "/home/vinals/vitra002/smectic/results/adv-nu";

	strBox += argv[1] + std::string("-e0d") + argv[2] 
	  + std::string("-r") + argv[3] + std::string("/");

	
/* ptrdiff_t: integer type, optimizes large transforms 64bit machines */

	const ptrdiff_t Nx = 512, Ny = 512, Nz = 512;
	const ptrdiff_t NG = Nx*Ny*Nz;
	const ptrdiff_t Nslice = Ny*Nz;
	
	ptrdiff_t alloc_local, local_n0, local_0_start;

	const int Nym = Ny-1, Nzm = Nz-1;

/* Constants and variables for morphologies (Nx = Ny = Nz) */

	const double mid = Nx/2; 
	const double aE = atof(argv[3]); // 270 (FC) // 80 // 312 // 432 // 248 // 810
	const double bE = atof(argv[3]); // 270 (FC) // 86 // 376 // 520 // 248 // 810

	double xs, ys, zs, ds;

/* Phase Field parameters */

	const double gamma =  1.0;
	const double beta  =  2.0;
	const double alpha =  1.0;
	double ep_arg    = atof(argv[2]); // -0.7 CHANGED !!!!!!!!!!!!!!
	const double ep = -0.01*ep_arg;
	const double q0    =  1.0;
	const double q02   = q0*q0;

/* Balance of Linear Momentum parameters */

	double nu = atof(argv[1]);
	
/* Points per wavelength, time step */
	
	const int    Nw = 16;
	const double dt = 0.0005; // 0.0005 (nw 16)	
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

	// REMOVEEEEEE
//	std::vector<double> psi(size*alloc_local);

/* Local data containers */

	std::vector <double> Vqx(local_n0), Vqy(Ny), Vqz(Nz);

	std::vector<double> aLin(alloc_local);
	std::vector<double> C1(alloc_local);
	std::vector<double> C2(alloc_local);
	std::vector<double> psi_local(alloc_local);
	std::vector<double> psiq_local(alloc_local);
	std::vector<double> psiNew_local(alloc_local);
	std::vector<double> Nr_local(alloc_local);
	std::vector<double> Nq_local(alloc_local);
	std::vector<double> NqPast_local(alloc_local);

	std::vector<double> Sp2_local(alloc_local);
	
/* Local data containers (wall potential) */
	
//	std::vector<double> wall(alloc_local);
//	std::vector<double> substrate(alloc_local);

/* Local data containers (advection) */

	std::vector <double> Vsx(local_n0), Vsy(Ny), Vsz(Nz);
	
	std::vector<double> Sx_local(alloc_local);
	std::vector<double> Sy_local(alloc_local);
	std::vector<double> Sz_local(alloc_local);

	std::vector<double> velx_local(alloc_local);
	std::vector<double> vely_local(alloc_local);
	std::vector<double> velz_local(alloc_local);
	
	std::vector<double> psi_temp(Nslice);
	std::vector<double> psi_front(Nslice);
	std::vector<double> psi_back(Nslice);
	std::vector<double> psi_front2(Nslice);
	std::vector<double> psi_back2(Nslice);

	std::vector<double> CM1x(alloc_local);
	std::vector<double> CM1y(alloc_local);
	std::vector<double> CM1z(alloc_local);
	std::vector<double> CM2x(alloc_local);
	std::vector<double> CM2y(alloc_local);
	std::vector<double> CM2z(alloc_local);

	std::vector<double> trans_local(alloc_local);
	
/* Local data containers (surface info) */

	std::vector<double> psiGradx_local(alloc_local);
	std::vector<double> psiGrady_local(alloc_local);
	std::vector<double> psiGradz_local(alloc_local);
	std::vector<double> dTpsi_local(alloc_local);
	std::vector<double> psiDxx_local(alloc_local);
	std::vector<double> psiDyy_local(alloc_local);
	std::vector<double> psiDzz_local(alloc_local);

	std::vector<double> psiSlice_local(alloc_slice);
	
	std::vector<double> surfZ_local(alloc_surf);
	std::vector<double> velSurf_local(alloc_surf);
	std::vector<double> curvH_local(alloc_surf);
	std::vector<double> curvK_local(alloc_surf);
	
/* Global data containers (surface info)*/

	std::vector<double> psiSlice(size*alloc_slice);
	
	std::vector<double> surfZ(size*alloc_surf);
	std::vector<double> velSurf(size*alloc_surf);
	std::vector<double> curvH(size*alloc_surf);
	std::vector<double> curvK(size*alloc_surf);
	

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

	planPsi = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		  	  psi_local.data(),psiq_local.data(),MPI::COMM_WORLD,
		  	  FFTW_REDFT10,FFTW_REDFT10,FFTW_REDFT10,
		  	  FFTW_MEASURE);

	iPlanPsi = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	   psiq_local.data(),psiNew_local.data(),MPI::COMM_WORLD,
		   	   FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,
		           FFTW_MEASURE);

	planN = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
			Nr_local.data(),Nq_local.data(),MPI::COMM_WORLD,
			FFTW_REDFT10,FFTW_REDFT10,FFTW_REDFT10,
	                FFTW_MEASURE);

	iPlanCT = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
			  trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
			  FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,
			  FFTW_MEASURE);

	planSTx = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
			 trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
			 FFTW_RODFT10,FFTW_REDFT10,FFTW_REDFT10,
	                 FFTW_MEASURE);

	planSTy = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
			 trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
			 FFTW_REDFT10,FFTW_RODFT10,FFTW_REDFT10,
	                 FFTW_MEASURE);

	planSTz = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
			 trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
			 FFTW_REDFT10,FFTW_REDFT10,FFTW_RODFT10,
	                 FFTW_MEASURE);
	
	iPlanSTx = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	  trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
		   	  FFTW_RODFT01,FFTW_REDFT01,FFTW_REDFT01,
		          FFTW_MEASURE);

	iPlanSTy = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	  trans_local.data(),trans_local.data(),MPI::COMM_WORLD,
		   	  FFTW_REDFT01,FFTW_RODFT01,FFTW_REDFT01,
		          FFTW_MEASURE);

	iPlanSTz = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
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

	double Amp = 1.328; 

/*************** Not in use ****************

	double Qi  = 2.0; // Perturbation wavelength

	if ( (k > Nx/5) && ( k < 4*Nx/5))
	psi_local[index] = Amp*cos(q0*k*dz);
					 + Amp*0.5*sin(q0*k*dz)*(cos(Qi*i*dx)+cos(Qi*j*dy)); 
					 + Amp*0.5*(cos(Qi*i*dx)+cos(Qi*j*dy));
	
********************************************/


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


/* Output IC to file and create L1 output */

	/** Create Psi output **/

	strPsi += std::to_string(rank);
	strPsi += ".dat";
	strPsi = strBox + strPsi;
	
	std::ofstream psi_output(strPsi.c_str());
	assert(psi_output.is_open());
	psi_output.close();
		
	if (rank == 0 )
	{	
		
	/** Create L1 output **/

	  std::ofstream L1_output(strBox+"L1.dat");
	  assert(L1_output.is_open());
	  L1_output.close();

	/** Create psiMid output **/

	  std::ofstream psiMid_output(strBox+"psiMid.dat");
	  assert(psiMid_output.is_open());
	  psiMid_output.close();

	/** Crete velocity outputs **/

	  std::ofstream sx_output(strBox+"vx.dat");
	  std::ofstream sy_output(strBox+"vy.dat");
	  std::ofstream sz_output(strBox+"vz.dat");
	  assert(sx_output.is_open());
	  assert(sy_output.is_open());
	  assert(sz_output.is_open());
	  sx_output.close();
	  sy_output.close();
	  sz_output.close();

	/** Create surf info outputs **/
		
	  std::ofstream surf_output(strBox+"surfPsi.dat");
	  std::ofstream velS_output(strBox+"velSurf.dat");
	  std::ofstream curvH_output(strBox+"curvH.dat");
	  std::ofstream curvK_output(strBox+"curvK.dat");
	  assert(surf_output.is_open());
	  assert(velS_output.is_open());
	  assert(curvH_output.is_open());
	  assert(curvK_output.is_open());
	  surf_output.close();
	  velS_output.close();
	  curvH_output.close();
	  curvK_output.close();	
		
	}

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
	  }
	  }}

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

	if ( rank == 0 )
	{

	/** Create L1 output **/

	  std::ofstream L1_output(strBox+"L1.dat");
	  assert(L1_output.is_open());
	  L1_output.close();

	/** Create psiMid output **/

	  std::ofstream psiMid_output(strBox+"psiMid.dat");
	  assert(psiMid_output.is_open());
	  psiMid_output.close();

	/** Create surf info outputs **/
		
	  std::ofstream surf_output(strBox+"surfPsi.dat");
	  std::ofstream velS_output(strBox+"velSurf.dat");
	  std::ofstream curvH_output(strBox+"curvH.dat");
	  std::ofstream curvK_output(strBox+"curvK.dat");
	  assert(surf_output.is_open());
	  assert(velS_output.is_open());
	  assert(curvH_output.is_open());
	  assert(curvK_output.is_open());
	  surf_output.close();
	  velS_output.close();
	  curvH_output.close();
	  curvK_output.close();	
		
	}

	} // End: load psi (B)

	
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

	 // Generate psiq for substrate gradient penalty
	 //fftw_execute(planPsi);
	 //

	 for ( i_local = 0; i_local < local_n0; i_local++ ){

	 i = i_local + local_0_start;

	 for ( j = 0; j < Ny; j++ ) {
	 for ( k = 0; k < Nz; k++ )
	 {
	   index =  (i_local*Ny + j)*Nz + k;
	   mq2 = pow(Vqx[i_local],2)+pow(Vqy[j],2)+pow(Vqz[k],2);
	   opSH = alpha*pow(mq2-q02,2);
	   aLin[index] = ep - opSH; ;
	   C1[index] = (1.0+dtd2*aLin[index]);
	   C2[index] = (1.0-dtd2*aLin[index]);
	   
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

	 // psiNew is the gradient penalty in real space
	 //fftw_execute(iPlanPsi);

	 // Compute Nr adding the substrate penalty * wall potential

	 for ( i_local = 0; i_local < local_n0; i_local++ ){

	 i = i_local + local_0_start;

	 for ( j = 0; j < Ny; j++ ) {
	 for ( k = 0; k < Nz; k++ )
	 {
	   index =  (i_local*Ny + j)*Nz + k;

	   Nr_local[index] = beta*pow(psi_local[index],3)
	     - gamma*pow(psi_local[index],5); // + psiNew_local[index]*wall[index];
	 }}}
	 //


 /* Move Nr_local to Fourier Space */

	 fftw_execute(planN);


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


 /********************************************
  *                                          *
  *   Time Loop (L1 as dynamics criterion)   *
  *                                          *
  *******************************************/

 //	while ( countL1 < 0 )
//	 while (L1 > limL1)
	 for (int tst = 0; tst < 10; tst++)
	 {

	   countL1++;

	   /** Empty out containers **/

	   std::fill(psiGradx_local.begin(),psiGradx_local.end(),0);
	   std::fill(psiGrady_local.begin(),psiGrady_local.end(),0);
	   std::fill(psiGradz_local.begin(),psiGradz_local.end(),0);

	   std::fill(velx_local.begin(),velx_local.end(),0);
	   std::fill(vely_local.begin(),vely_local.end(),0);
	   std::fill(velz_local.begin(),velz_local.end(),0);


	   /** Previous Nq_local is now NqPast_local  **/

	   NqPast_local = Nq_local;


	   /** Move psi to FS and scale it **/

	   fftw_execute(planPsi);

	   for ( i_local = 0; i_local < local_n0; i_local++ ){
	   for ( j = 0; j < Ny; j++ ) {
	   for ( k = 0; k < Nz; k++ )
	   {

	     i = i_local + local_0_start;

	     index =  (i_local*(Ny) + j)*(Nz) + k;

	     psiq_local[index] = scale*psiq_local[index];
	   
	   }}}	


	   /** COMPUTE: gradient of psi **/
	   // partial_x psi (parallelized direction)

	   i_local = 0;
	 
	   for( j = 0; j < Ny; j++ ){
	   for( k = 0; k < Nz; k++ ){

	     index2 = (i_local*Ny + j) * Nz + k;
	     index = j*Nz + k;

	     psi_back[index] = psiq_local[index2];
	   }}

	   if (rank == size-1){
	   
	     MPI::COMM_WORLD.Send(psi_back.data(),Nslice,
				  MPI::DOUBLE,rank-1,0);
	   
	   } else if (rank % 2 != 0){

	     MPI::COMM_WORLD.Send(psi_back.data(),Nslice,
				  MPI::DOUBLE,rank-1,0);

	     MPI::COMM_WORLD.Recv(psi_front.data(),Nslice,
				  MPI::DOUBLE,rank+1,0);

	   } else if (rank != 0){

	     MPI::COMM_WORLD.Recv(psi_front.data(),Nslice,
				  MPI::DOUBLE,rank+1,0);

	     MPI::COMM_WORLD.Send(psi_back.data(),Nslice,
				  MPI::DOUBLE,rank-1,0);

	   } else {

	     MPI::COMM_WORLD.Recv(psi_front.data(),Nslice,
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

	     }}
	   }
	 
	   for ( i_local = 0; i_local < local_n0-1; i_local++ ) {
	   for ( j = 0; j < Ny; j++ ) {
	   for ( k = 0; k < Nz; k++ )
	   {

	     index = (i_local*Ny + j) * Nz + k;
	     index2 = ((i_local+1)*Ny + j) * Nz + k;
	   
	     psiGradx_local[index] = -Vsx[i_local]*psiq_local[index2];

	   }}}

	   // partial_y psi

	   for ( i_local = 0; i_local < local_n0; i_local++ ){
	   for ( j = 0; j < Ny-1; j++ ) {
	   for ( k = 0; k < Nz; k++ )
	   {

	     index =  (i_local*(Ny) + j)*(Nz) + k;
	     index2 =  (i_local*(Ny) + j+1)*(Nz) + k;
	 
	     psiGrady_local[index] = -Vsy[j]*psiq_local[index2];
	   
	   }}}	

	   // partial_z psi

	   for ( i_local = 0; i_local < local_n0; i_local++ ){
	   for ( j = 0; j < Ny; j++ ) {
	   for ( k = 0; k < Nz-1; k++ )
	   {

	     index =  (i_local*(Ny) + j)*(Nz) + k;	 
	     index2 =  (i_local*(Ny) + j)*(Nz) + k+1;	 
	    
	     psiGradz_local[index] = -Vsz[k]*psiq_local[index2];
	   
	   }}}	

	   trans_local = psiq_local;
	   fftw_execute(iPlanCT);
	   psi_local = trans_local;

	   trans_local = psiGradx_local;
	   fftw_execute(iPlanSTx);
	   psiGradx_local = trans_local;

	   trans_local = psiGrady_local;
	   fftw_execute(iPlanSTy);
	   psiGrady_local = trans_local;

	   trans_local = psiGradz_local;
	   fftw_execute(iPlanSTz);
	   psiGradz_local = trans_local;


	   /** Sp2: linear part of mu, psiq is already scaled **/

	   for ( i_local = 0; i_local < local_n0; i_local++ ){
	   for ( j = 0; j < Ny; j++ ) {
	   for ( k = 0; k < Nz; k++ ) 
	   {
	     index =  (i_local*Ny + j)*Nz + k;

	     Sp2_local[index] = -aLin[index]*psiq_local[index];	    	    	    
	   }}}

	   trans_local = Sp2_local;
	   fftw_execute(iPlanCT);
	   Sp2_local = trans_local;


	   /** Compute mu grad psi and move it to Fourier Space **/
	   
	   for ( i_local = 0; i_local < local_n0; i_local++ ){
	   for ( j = 0; j < Ny; j++ ) {
	   for ( k = 0; k < Nz; k++ ) 
	   {

	     index =  (i_local*Ny + j)*Nz + k;

	     Sx_local[index] =
	       (Sp2_local[index] - beta*pow(psi_local[index],3)
		+ gamma*pow(psi_local[index],5))*psiGradx_local[index];

	     Sy_local[index] =
	       (Sp2_local[index] - beta*pow(psi_local[index],3)
		+ gamma*pow(psi_local[index],5))*psiGrady_local[index];

	     Sz_local[index] =
	       (Sp2_local[index] - beta*pow(psi_local[index],3)
		+ gamma*pow(psi_local[index],5))*psiGradz_local[index];

	   }}}

	   trans_local = Sx_local;
	   fftw_execute(planSTx);
	   Sx_local = trans_local;

	   trans_local = Sy_local;
	   fftw_execute(planSTy);
	   Sy_local = trans_local;

	   trans_local = Sz_local;
	   fftw_execute(planSTz);
	   Sz_local = trans_local;


	   /** COMPUTE: Velocity field **/
	   // vel_x (parallelized direction)

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

	   // Use front data to compute vx at i_local=local_n0-1

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
	     
	       velx_local[index] = CM1x[index]*(Sx_local[index]
					       - CM2x[index]*dotSqVq);

	     }}
	   } 	 


	   // Compute vel_x for the rest

	   for ( i_local = 0; i_local < local_n0-1; i_local++ ) {
	   for ( j = 0; j < Ny; j++ ) {
	   for ( k = 0; k < Nz; k++ )
	   {
	     index = (i_local*Ny + j) * Nz + k;

	     if ( j !=0) {
	       index2 = ((i_local+1)*Ny + j-1) * Nz + k;
	       Sy = Sy_local[index2];
	     } else {
	       Sy = 0;
	     }

	     if ( k !=0) {
	       index2 = ((i_local+1)*Ny + j) * Nz + k-1;
	       Sz = Sz_local[index2];
	     } else {
	       Sz = 0;
	     }
	     
	     dotSqVq = Vsx[i_local]*Sx_local[index] + Vqy[j]*Sy + Vqz[k]*Sz;
	   	     
	     velx_local[index] = CM1x[index]*(Sx_local[index]
					      - CM2x[index]*dotSqVq);
	   }}}


	   // Send i_local=local_n0-1 Sx data to back rank 

	   std::fill(psi_back.begin(),psi_back.end(),0);

	   i_local = local_n0-1;
	 
	   for( j = 0; j < Ny; j++ ){
	   for( k = 0; k < Nz; k++ ){

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

	   // Compute vel_y and vel_z

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
	   	     
	     vely_local[index] = CM1y[index]*(Sy_local[index]
					      - CM2y[index]*dotSqVq);
	   
	     index2 = j*Nz+k+1;
	     Sx = psi_back[index2];	     	     

	     if ( j !=0) {
	       index2 = ((i_local)*Ny + j-1) * Nz + k+1;
	       Sy = Sy_local[index2];
	     } else {
	       Sy = 0;
	     }
	     
	     dotSqVq = Vqx[i_local]*Sx + Vqy[j]*Sy + Vsz[k]*Sz_local[index];
	   	     
	     velz_local[index] = CM1z[index]*(Sz_local[index]
					      - CM2z[index]*dotSqVq);
	   }}


	   for ( i_local = 1; i_local < local_n0; i_local++ ) {
	   for ( j = 0; j < Ny-1; j++ ) {
	   for ( k = 0; k < Nz-1; k++ )
	   {
	     index = (i_local*Ny + j) * Nz + k;

	     index2 = ((i_local-1)*Ny + j+1) * Nz + k;
	     Sx = Sx_local[index2];

	     if ( k !=0) {
	       index2 = ((i_local)*Ny + j+1) * Nz + k-1;
	       Sz = Sz_local[index2];
	     } else {
	       Sz = 0;
	     }
	     
	     dotSqVq = Vqx[i_local]*Sx + Vsy[j]*Sy_local[index] + Vqz[k]*Sz;	   
	     
	     vely_local[index] = CM1y[index]*(Sy_local[index]
					      - CM2y[index]*dotSqVq);

	     index2 = ((i_local-1)*Ny + j) * Nz + k+1;
	     Sx = Sx_local[index2];

	     if ( j !=0) {
	       index2 = ((i_local)*Ny + j-1) * Nz + k+1;
	       Sy = Sy_local[index2];
	     } else {
	       Sy = 0;
	     }
	     
	     dotSqVq = Vqx[i_local]*Sx + Vqy[j]*Sy + Vsz[k]*Sz_local[index];	   
	     
	     velz_local[index] = CM1z[index]*(Sz_local[index]
					      - CM2z[index]*dotSqVq);
	   }}}

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
	     Nr_local[index] = beta*pow(psi_local[index],3)
	       - gamma*pow(psi_local[index],5)
	       - Sx_local[index]*psiGradx_local[index]
	       - Sy_local[index]*psiGrady_local[index]
	       - Sz_local[index]*psiGradz_local[index];
	   }}}

	   /* Obtain current Nq_local */

	   fftw_execute(planN);	

	   /* COMPUTE: SECOND DERIVATIVES AND PF VELOCITY */

	   if (countL1 == 1 & countSave == 0) {

	     for ( i_local = 0; i_local < local_n0; i_local++ ){

	     for ( j = 0; j < Ny; j++ ) {
	     for ( k = 0; k < Nz; k++ )
	     {
	       index =  (i_local*Ny + j)*Nz + k;

	       dTpsi_local[index] = (aLin[index]*psiq_local[index]
				     +scale*Nq_local[index]);

	       psiDxx_local[index] = -Vqx[i_local]*Vqx[i_local]*psiq_local[index];

	       psiDyy_local[index] = -Vqy[j]*Vqy[j]*psiq_local[index];

	       psiDzz_local[index] = -Vqz[k]*Vqz[k]*psiq_local[index];

	     }}}	

	     trans_local = psiDxx_local;
	     fftw_execute(iPlanCT);
	     psiDxx_local = trans_local;

	     trans_local = psiDyy_local;
	     fftw_execute(iPlanCT);
	     psiDyy_local = trans_local;

	     trans_local = psiDzz_local;
	     fftw_execute(iPlanCT);
	     psiDzz_local = trans_local;

	     trans_local = dTpsi_local;
	     fftw_execute(iPlanCT);
	     dTpsi_local = trans_local;

	   }

		 if (tst == 0){
		   NqPast_local = Nq_local;
		 }

		 /* COMPUTE: NEW PSI IN FOURIER SPACE (CN/AB scheme) */

		 for ( i_local = 0; i_local < local_n0; i_local++ ){

		 for ( j = 0; j < Ny; j++ ) {
		 for ( k = 0; k < Nz; k++ )
		 {
		   index =  (i_local*Ny + j)*Nz + k;

		   psiq_local[index] = 
		     (C1[index]*psiq_local[index]
		      + dtd2*scale*(3.0*Nq_local[index]-NqPast_local[index]))/C2[index];
		 }}}	
		 
		 /** Obtain new psi in real space **/

		 fftw_execute(iPlanPsi);

		 
		 /* COMPUTE: L1 (under count condition) */
		 
		 if ( countL1 == 1 )
		 {

		 sumA_local = 0.0; sumB_local = 0.0;
		 sumA = 0.0;       sumB = 0.0;

		 for ( i_local = 0; i_local < local_n0; i_local++ ) {
		 for ( j = 0; j < Ny; j++ ) {
		 for ( k = 0; k < Nz; k++ )
		 {
		   index = (i_local*Ny + j) * Nz + k;
		   
		   sumA_local = sumA_local  
		     + fabs(psiNew_local[index] - psi_local[index]);
		     
		   sumB_local = sumB_local + fabs(psiNew_local[index]);
		 }}}

		 MPI::COMM_WORLD.Reduce(&sumA_local,&sumA,1,MPI::DOUBLE,MPI::SUM,0);
		 MPI::COMM_WORLD.Reduce(&sumB_local,&sumB,1,MPI::DOUBLE,MPI::SUM,0);

		 if ( rank == 0)
		 {
		   L1 = sumA/(dt*sumB);
		   L1_output.open(strBox+"L1.dat",std::ios_base::app); // append result
		   assert(L1_output.is_open());
		   L1_output << L1 << "\n";
		   L1_output.close();
		 }

		 MPI::COMM_WORLD.Bcast(&L1,1,MPI::DOUBLE,0);

		 countL1 = 0;

		 countSave++;

		 /* SAVE PSI & OBTAIN SURFACE INFO (under count condition) */

		 if ( countSave == 1 )
		 { 
			 			 
		 /** Appropriate way to compute curvatures for a TFCD **/

		 for ( i_local = 0; i_local < local_n0; i_local++ ) {
		 for ( j = 0; j < Ny; j++ ) 
		 {	
		   track = 0;
		   index2 = i_local*Ny + j;

		 for ( k = Nz-1; k > -1; k-- ) 
		 {

		   index = (i_local*Ny + j) * Nz + k;

		   // 0.7 : results are better when looking for this 0
		   if ( psi_local[index] > 0.7 & track == 0 ) 
		   {
		     track = 1;
		   }
		   if ( psi_local[index] < 0.0 & track == 1 ) //std::abs(...) > 0.7
		   {
		     k2 = k;

		     if( std::abs(psi_local[index]) > std::abs(psi_local[index+1]) ) // >
		     {
		       index = index + 1;
		       k2 = k + 1;
		     }
		    
		    surfZ_local[index2] = k2;
				
		    gradVal = sqrt(psiGradx_local[index]*psiGradx_local[index]
				   + psiGrady_local[index]*psiGrady_local[index]
				   + psiGradz_local[index]*psiGradz_local[index]);
				
		    velSurf_local[index2] = dTpsi_local[index]/gradVal;

		    // Mixed second order derivates do not work with the DCT, so...
				
		    if ( j > 0 & j < (Ny-1) )
		    {
		      psiDxy = (psiGradx_local[(i_local*Ny + j+1) * Nz + k2]
				- psiGradx_local[(i_local*Ny + j-1) * Nz + k2])/tdy;
		    }			
		    
		    if ( j == 0 )
		    {
		      psiDxy = 2*(psiGradx_local[(i_local*Ny + 1) * Nz + k2] 
				  - psiGradx_local[(i_local*Ny) * Nz + k2])/tdy;
		    }
		    
		    if( j== Ny-1 )
		    {
		      psiDxy = 2*(psiGradx_local[(i_local*Ny + j) * Nz + k2] 
				  - psiGradx_local[(i_local*Ny+j-1) * Nz + k2])/tdy;
		    }

		    if( k > 0 & k < (Nz-1) )
		    {
		      psiDxz = (psiGradx_local[index+1]-psiGradx_local[index-1])/tdz;
		      psiDyz = (psiGrady_local[index+1]-psiGrady_local[index-1])/tdz;
		    }			
		    
		    if( k == 0  )
		    {
		      psiDxz = 2*(psiGradx_local[index+1]-psiGradx_local[index])/tdz;
		      psiDyz = 2*(psiGrady_local[index+1]-psiGrady_local[index])/tdz;
		    }
		
		    if(k == (Nz-1) )
		    {
		      psiDxz = 2*(psiGradx_local[index] - psiGradx_local[index-1])/tdz;
		      psiDyz = 2*(psiGrady_local[index] - psiGrady_local[index-1])/tdz;
		    }
				
		    // Proper way to numerically compute H and K
		    // (Megrabov 2014, On divergence representations ..)
		    // Note: I'm obtaining 2H instead of H, but K is okay

		    curvH_local[index2] =
		      ((pow(psiGrady_local[index],2)+pow(psiGradz_local[index],2))*psiDxx_local[index]
		       +(pow(psiGradx_local[index],2)+pow(psiGradz_local[index],2))*psiDyy_local[index]
		       +(pow(psiGradx_local[index],2)+pow(psiGrady_local[index],2))*psiDzz_local[index]
		       -2*(psiGradx_local[index]*psiGrady_local[index]*psiDxy
			   +psiGradx_local[index]*psiGradz_local[index]*psiDxz
			   +psiGrady_local[index]*psiGradz_local[index]*psiDyz))
		      / pow(gradVal,3);
				
		    curvK_local[index2] =
		      (pow(psiGradz_local[index],2)
		       *(psiDxx_local[index]*psiDyy_local[index]-pow(psiDxy,2))
		       + pow(psiGradx_local[index],2)
		       *(psiDyy_local[index]*psiDzz_local[index]-pow(psiDyz,2))
		       + pow(psiGrady_local[index],2)
		       *(psiDxx_local[index]*psiDzz_local[index]-pow(psiDxz,2))
		       + 2*(psiGrady_local[index]*psiDxy
			    *(psiGradz_local[index]*psiDxz-psiGradx_local[index]*psiDzz_local[index])
			    +psiGradx_local[index]*psiDxz
			    *(psiGrady_local[index]*psiDyz-psiGradz_local[index]*psiDyy_local[index])
			   +psiGradz_local[index]*psiDyz
			    *(psiGradx_local[index]*psiDxy-psiGrady_local[index]*psiDxx_local[index])
			    ))/pow(gradVal,4);

		    track = 2;
		  }
		}}
		}

		MPI::COMM_WORLD.Gather(surfZ_local.data(),alloc_surf,MPI::DOUBLE,
							   surfZ.data(),alloc_surf, MPI::DOUBLE,0);

		MPI::COMM_WORLD.Gather(velSurf_local.data(),alloc_surf,MPI::DOUBLE,
							   velSurf.data(),alloc_surf, MPI::DOUBLE,0);

		MPI::COMM_WORLD.Gather(curvH_local.data(),alloc_surf,MPI::DOUBLE,
							   curvH.data(),alloc_surf, MPI::DOUBLE,0);

		MPI::COMM_WORLD.Gather(curvK_local.data(),alloc_surf,MPI::DOUBLE,
							   curvK.data(),alloc_surf, MPI::DOUBLE,0);


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

		  surf_output.open(strBox+"surfPsi.dat",std::ios_base::app);
		  velS_output.open(strBox+"velSurf.dat",std::ios_base::app);
		  curvH_output.open(strBox+"curvH.dat",std::ios_base::app);
		  curvK_output.open(strBox+"curvK.dat",std::ios_base::app);
		  psiMid_output.open(strBox+"psiMid.dat",std::ios_base::app);
					
		  assert(surf_output.is_open());
		  assert(velS_output.is_open());
		  assert(curvH_output.is_open());
		  assert(curvK_output.is_open());
		  assert(psiMid_output.is_open());

		  for ( i = 0; i < Nx; i++ ) {
		  for ( j = 0; j < Ny; j++ ) 
		  {
		    index = i*Ny + j;

		    surf_output << surfZ[index] << "\n";
		    velS_output << velSurf[index] << "\n ";
		    curvH_output << curvH[index] << "\n ";
		    curvK_output << curvK[index] << "\n ";			
		  }}

		  for ( i = 0; i < Nx; i++ ) {
		  for ( k = 0; k < Nz; k++ ) 
		  {
		    index = i*Nz + k;

		    psiMid_output << psiSlice[index] << "\n";
		  }}

		  surf_output.close();
		  velS_output.close();
		  curvH_output.close();
		  curvK_output.close();	
		  psiMid_output.close();
	
	/** Switch between two save files **/
		  /*
		    if ( swtPsi == 0) {
		    strPsi = "psi.dat";
		    swtPsi = 1;
		    }
		    else {
		    strPsi = "psiB.dat";
		    swtPsi = 0;						
		    }
		    
		    std::ofstream psi_output(strPsi.c_str());
		    assert(psi_output.is_open());
	
		    for ( i = 0; i < Nx; i++ ) {
		    for ( j = 0; j < Ny; j++ ) {
		    for ( k = 0; k < Nz; k++ ) 
		    {
		    index = (i*Ny + j) * Nz + k;
		    psi_output << psi[index] << "\n ";
		    }}}
		    
		    psi_output.close();
		  */		

	/** Inform date and time after each save psi **/

		  time_t now = time(0);
		  char* dNow = ctime(&now);
		  nLoop++;    		   

		  std::cout << "The loop " << 1000*nLoop 
			    << " local date and time is: " << dNow << std::endl;
					
		} // ends rank 0 outputs

		// Save vel_x for mid cross section

		j = Ny/2;
		for( k = 0; k < Nz ; k++ ){
		for( i_local = 0; i_local < local_n0 ; i_local++ ){
			index  = (i_local*Ny +j)*Nz + k;
			index2 = i_local*Nz + k;
			psiSlice_local[index2] = velx_local[index];
		}}

		MPI::COMM_WORLD.Gather(psiSlice_local.data(),alloc_slice,MPI::DOUBLE,
					   psiSlice.data(),alloc_slice, MPI::DOUBLE,0);

		if (rank == 0 )
		  {
		    sx_output.open(strBox+"vx.dat",std::ios_base::app);
									
		    assert(sx_output.is_open());


		    for ( i = 0; i < Nx; i++ ) {
		      for ( k = 0; k < Nz; k++ ) {
			
			index = i*Nz + k;
			
			sx_output << psiSlice[index] << "\n";
		      }}

		    sx_output.close();

		  }

		// Save vel_y for mid cross section

		j = Ny/2;
		for( k = 0; k < Nz ; k++ ){
		for( i_local = 0; i_local < local_n0 ; i_local++ ){
			index  = (i_local*Ny +j)*Nz + k;
			index2 = i_local*Nz + k;
			psiSlice_local[index2] = vely_local[index];
		}}

		MPI::COMM_WORLD.Gather(psiSlice_local.data(),alloc_slice,MPI::DOUBLE,
					   psiSlice.data(),alloc_slice, MPI::DOUBLE,0);

		if (rank == 0 )
		  {
		    sy_output.open(strBox+"vy.dat",std::ios_base::app);
									
		    assert(sy_output.is_open());


		    for ( i = 0; i < Nx; i++ ) {
		      for ( k = 0; k < Nz; k++ ) {
			
			index = i*Nz + k;
			
			sy_output << psiSlice[index] << "\n";
		      }}

		    sy_output.close();

		  }

		// Save vel_z for mid cross section

		j = Ny/2;
		for( k = 0; k < Nz ; k++ ){
		for( i_local = 0; i_local < local_n0 ; i_local++ ){
			index  = (i_local*Ny +j)*Nz + k;
			index2 = i_local*Nz + k;
			psiSlice_local[index2] = velz_local[index];
		}}

		MPI::COMM_WORLD.Gather(psiSlice_local.data(),alloc_slice,MPI::DOUBLE,
					   psiSlice.data(),alloc_slice, MPI::DOUBLE,0);

		if (rank == 0 )
		  {
		    sz_output.open(strBox+"vz.dat",std::ios_base::app);
									
		    assert(sz_output.is_open());


		    for ( i = 0; i < Nx; i++ ) {
		      for ( k = 0; k < Nz; k++ ) {
			
			index = i*Nz + k;
			
			sz_output << psiSlice[index] << "\n";
		      }}

		    sz_output.close();

		  }


		MPI::COMM_WORLD.Barrier();
		
		countSave = 0;

		 } // End: countSave block
		
		 } // End: countL1 block

		 psi_local = psiNew_local;

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

  	fftw_destroy_plan(planPsi);
  	fftw_destroy_plan(iPlanPsi);
  	fftw_destroy_plan(planN);

  	fftw_destroy_plan(planSTx);
  	fftw_destroy_plan(iPlanSTx);
  	fftw_destroy_plan(planSTy);
  	fftw_destroy_plan(iPlanSTy);
  	fftw_destroy_plan(planSTz);
  	fftw_destroy_plan(iPlanSTz);
  	fftw_destroy_plan(iPlanCT);
	
  	fftw_cleanup();
	
/* Finalize MPI */

	MPI::Finalize();

} // END
