# sma-dft

The purpose of the sma-dft project is to study the evolution of an interface between a smectic-A liquid crystal and a disordered phase. The present files contain an implementation of the models proposed in:
(1) E. Vitral, P.H. Leo, J. Viñals, Physical Review E 100.3 (2019)
(2) E. Vitral, P.H. Leo, J. Viñals, in preparation

These are custom C++ codes based on the parallel FFTW library and the standard MPI passing interface. Part A was the initial study of a smectic-A in contat with its isotropic phase of same density, describing a diffusive evolution of the interface without advection, whose results are found in Ref. (1). Part B is similar to A, but introduces advection to a uniform density system. Part C contains implementations of a quasi-incompressible smectic-isotropic system, presenting a varying density field between phases. The derivation, discussion and numerical results associated to this model will be published in Ref. (2).


## A. no-adv: smectic-iso, advection is off, uniform density

A1. cosNoAdvConics.cpp

A2. cosNoAdvTrackBump.cpp

A3. cosNoAdvTrackMult.cpp

A4. cosNoAdvTrackSing.cpp

## B. adv: smectic-iso, advection is on, uniform density

B1. cosAdvTrackMult.cpp

B2. cosAdvConSingle.cpp

B3. cosAdvConConics.cpp

B4. cosAdvConSameConics.cpp

B5. cosAdvConDifConics.cpp

## C. quasi: smectic-iso, 

C1. quasi.cpp : smectic-iso, quasi-incompressible model, varying density

C2. quasi-stb.cpp : stability analysis