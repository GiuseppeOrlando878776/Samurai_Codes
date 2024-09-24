/* ------------------------------------------------------------------ */
/*       Flux CONSERVATIF associe au systeme homogene                 */
/* ------------------------------------------------------------------ */

Etat Flux(Etat U, int direction)
{
	if(direction == 0) {
		return Etat(
					0.,
					U.alrhou1,
					U.alrhou1*U.alrhou1/U.alrho1+U.al1*P1(U),
					U.alrhov1*U.alrhou1/U.alrho1,
					(U.alrhoE1+U.al1*P1(U))*U.alrhou1/U.alrho1,
					U.alrhou2,
					U.alrhou2*U.alrhou2/U.alrho2+(1.-U.al1)*P2(U),
					(U.alrhoE2+(1.-U.al1)*P2(U))*U.alrhou2/U.alrho2
				);
	}
	else if(direction == 1) {
		return Etat(
					0.,
					U.alrhov1,
					U.alrhou1*U.alrhov1/U.alrho1,
					U.alrhov1*U.alrhov1/U.alrho1+U.al1*P1(U),
					(U.alrhoE1+U.al1*P1(U))*U.alrhov1/U.alrho1,
					U.alrhov2,
					U.alrhov2*U.alrhov2/U.alrho2+(1.-U.al1)*P2(U),
					(U.alrhoE2+(1.-U.al1)*P2(U))*U.alrhov2/U.alrho2
				);
	}
}

/* ------------------------------------------------------------------ 	*/
/*       Calcul du flux numerique CONSERVATIF pour Rusanov	 	*/
/* ------------------------------------------------------------------ 	*/


//  Flux de RUSANOV entre deux etats gauche et droit ///////////

double fluxNumRs(Etat Ug, Etat Ud, Etat &f, int direction)
{
	double lmax;
	if(direction == 0) {
		lmax = max(
							 max(
							 		 max(fabs(Ug.alrhou1/Ug.alrho1+c1(Ug)),
									 		 fabs(Ug.alrhou1/Ug.alrho1-c1(Ug))),
							 		 max(fabs(Ug.alrhou2/Ug.alrho2+c2(Ug)),
									 		 fabs(Ug.alrhou2/Ug.alrho2-c2(Ug)))
									 ),
						 	 max(
							 		 max(fabs(Ud.alrhou1/Ud.alrho1+c1(Ud)),
									 		 fabs(Ud.alrhou1/Ud.alrho1-c1(Ud))),
							 		 max(fabs(Ud.alrhou2/Ud.alrho2+c2(Ud)),
									 		 fabs(Ud.alrhou2/Ud.alrho2-c2(Ud)))
									 )
						 );
	}
	else if(direction == 1) {
		lmax = max(
							 max(
							 		 max(fabs(Ug.alrhov1/Ug.alrho1+c1(Ug)),
									 		 fabs(Ug.alrhov1/Ug.alrho1-c1(Ug))),
							 		 max(fabs(Ug.alrhov2/Ug.alrho2+c2(Ug)),
									 		 fabs(Ug.alrhov2/Ug.alrho2-c2(Ug)))
									 ),
						 	 max(
							 		 max(fabs(Ud.alrhov1/Ud.alrho1+c1(Ud)),
									 		 fabs(Ud.alrhov1/Ud.alrho1-c1(Ud))),
							 		 max(fabs(Ud.alrhov2/Ud.alrho2+c2(Ud)),
									 		 fabs(Ud.alrhov2/Ud.alrho2-c2(Ud)))
									 )
						 );
	}

	f = (Flux(Ug,direction)+Flux(Ud,direction))/2. - lmax*(Ud-Ug)/2.;

  return lmax;
}



//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//		CALCUL DES FLUX POUR LA RELAXATION
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


double flux_relax(Etat UL, Etat UR, Etat &fWmoins, Etat &fWplus, int &ptfix, int &dissip, double eps, int direction)
{
	double alpha1L	= UL.al1;
	double rho1L	= UL.alrho1/alpha1L;
	double u1L	= UL.alrhou1/UL.alrho1;
	double v1L = UL.arhov1/UL.arho1;
        double E1L	= UL.alrhoE1/UL.alrho1;
        double e1L      = E1L-u1L*u1L/2.-v1L*v1L/2.;
	double pi1L	= p1(rho1L,e1L);

	double alpha1R	= UR.al1;
	double rho1R	= UR.alrho1/alpha1R;
	double u1R	= UR.alrhou1/UR.alrho1;
	double v1R	= UR.alrhov1/UR.alrho1;
        double E1R	= UR.alrhoE1/UR.alrho1;
        double e1R      = E1R-u1R*u1R/2.-v1R*v1R/2.;
	double pi1R	= p1(rho1R,e1R);

	double alpha2L	= 1.-alpha1L;
	double rho2L	= UL.alrho2/alpha2L;
	double u2L	= UL.alrhou2/UL.alrho2;
	double v2L = UL.arhov2/UL.alrho2;
	double E2L	= UL.alrhoE2/UL.alrho2;
        double e2L      = E2L-u2L*u2L/2.-v2L*v2L/2.;
	double pi2L	= p2(rho2L,e2L);

	double alpha2R	= 1.-alpha1R;
	double rho2R	= UR.alrho2/alpha2R;
	double u2R	= UR.alrhou2/UR.alrho2;
	double v2R  = UR.alrhov2/UR.alrho2;
	double E2R	= UR.alrhoE2/UR.alrho2;
        double e2R      = E2R-u2R*u2R/2.-v2R*v2R/2.;
	double pi2R	= p2(rho2R,e2R);

//         cout << "E2L = " << E2L << endl ;
//         cout << "E2R = " << E2R << endl ;
//         cout << "e2L = " << e2L << endl ;
//         cout << "e2R = " << e2R << endl ;

	// Calcul de a1, a2 (non definitif) avec Whitham approche
	double a1	= max(cc1(rho1L,e1L)*rho1L,cc1(rho1R,e1R)*rho1R);
	double a2	= max(cc2(rho2L,e2L)*rho2L,cc2(rho2R,e2R)*rho2R);

//         double a1	= max(c1(UL)*rho1L,c1(UR)*rho1R);
// 	double a2	= max(c2(UL)*rho2L,c2(UR)*rho2R);


        double u1e, pidxalpha2;

	double un1d, pi1d, t1Ld, t1Rd;
	double un2d, pi2d, t2Ld, t2Rd;

	double alpha1m, tau1m, u1m, /*w1m,*/ pi1m, E1m;
	double alpha1p, tau1p, u1p, /*w1p,*/ pi1p, E1p;

	double alpha2m, tau2m, u2m, w2m, pi2m, E2m;
	double alpha2p, tau2p, u2p, w2p, pi2p, E2p;

	double cLmax, cRmin, borninf, bornsup, cible;



	// Calcul de a1 (non definitif) pour que les tau diese soit > 0
	do
	{
		a1	= 1.01*a1;
		if(direction == 0) {
			un1d	= 0.5*(u1L+u1R)-0.5/a1*(pi1R-pi1L);
			pi1d	= 0.5*(pi1R+pi1L)-0.5*a1*(u1R-u1L);
			t1Ld	= 1./rho1L + 1./a1*(un1d-u1L);
			t1Rd	= 1./rho1R - 1./a1*(un1d-u1R);
		}
		else if(direction == 1) {
			un1d = 0.5*(v1L+v1R)-0.5/a1*(pi1R-pi1L);
			pi1d	= 0.5*(pi1R+pi1L)-0.5*a1*(v1R-v1L);
			t1Ld	= 1./rho1L + 1./a1*(un1d-v1L);
			t1Rd	= 1./rho1R - 1./a1*(un1d-v1R);
		}

	} while (t1Ld <= 0. || t1Rd <=0.);

	// Calcul de a2 (non definitif) pour que les tau diese soit > 0
	do
	{
		a2	= 1.01*a2;
		if(direction == 0) {
			un2d	= 0.5*(u2L+u2R)-0.5/a2*(pi2R-pi2L);
			pi2d	= 0.5*(pi2R+pi2L)-0.5*a2*(u2R-u2L);
			t2Ld	= 1./rho2L + 1./a2*(un2d-u2L);
			t2Rd	= 1./rho2R - 1./a2*(un2d-u2R);
		}
		else if(direction == 1) {
			un2d = 0.5*(v2L+v2R)-0.5/a2*(pi2R-pi2L);
			pi2d	= 0.5*(pi2R+pi2L)-0.5*a2*(v2R-v2L);
			t2Ld	= 1./rho2L + 1./a2*(un2d-v2L);
			t2Rd	= 1./rho2R - 1./a2*(un2d-v2R);
		}

	} while (t2Ld <= 0. || t2Rd <=0.);



        //Mise a jour de a1 et a2 pour assurer l'existence d'une solution au point fixe.

        do
        {
                if (un1d-a1*t1Ld > un2d-a2*t2Ld && un1d+a1*t1Rd < un2d+a2*t2Rd )
                {
                        a1	= 1.01*a1;
												if(direction == 0) {
                        	un1d	= 0.5*(u1L+u1R)-0.5/a1*(pi1R-pi1L);
													pi1d	= 0.5*(pi1R+pi1L)-0.5*a1*(u1R-u1L);
													t1Ld	= 1./rho1L + 1./a1*(un1d-u1L);
													t1Rd	= 1./rho1R - 1./a1*(un1d-u1R);
												}
												else if(direction == 1) {
                        	un1d	= 0.5*(v1L+v1R)-0.5/a1*(pi1R-pi1L);
													pi1d	= 0.5*(pi1R+pi1L)-0.5*a1*(v1R-v1L);
													t1Ld	= 1./rho1L + 1./a1*(un1d-v1L);
													t1Rd	= 1./rho1R - 1./a1*(un1d-v1R);
												}
                }
                else
                {
                        if (un2d-a2*t2Ld > un1d-a1*t1Ld && un2d+a2*t2Rd < un1d+a1*t1Rd )
                        {
                            a2	= 1.01*a2;
														if(direction == 0) {
                            	un2d	= 0.5*(u2L+u2R)-0.5/a2*(pi2R-pi2L);
															pi2d	= 0.5*(pi2R+pi2L)-0.5*a2*(u2R-u2L);
															t2Ld	= 1./rho2L + 1./a2*(un2d-u2L);
															t2Rd	= 1./rho2R - 1./a2*(un2d-u2R);
														}
														else if(direction == 1) {
															un2d	= 0.5*(v2L+v2R)-0.5/a2*(pi2R-pi2L);
															pi2d	= 0.5*(pi2R+pi2L)-0.5*a2*(v2R-v2L);
															t2Ld	= 1./rho2L + 1./a2*(un2d-v2L);
															t2Rd	= 1./rho2R - 1./a2*(un2d-v2R);
														}
                        }
                        else
                        {
                            a1	= 1.01*a1;
														if(direction == 0) {
                            	un1d	= 0.5*(u1L+u1R)-0.5/a1*(pi1R-pi1L);
															pi1d	= 0.5*(pi1R+pi1L)-0.5*a1*(u1R-u1L);
															t1Ld	= 1./rho1L + 1./a1*(un1d-u1L);
															t1Rd	= 1./rho1R - 1./a1*(un1d-u1R);
														}
														else if(direction == 1) {
															un1d	= 0.5*(v1L+v1R)-0.5/a1*(pi1R-pi1L);
															pi1d	= 0.5*(pi1R+pi1L)-0.5*a1*(v1R-v1L);
															t1Ld	= 1./rho1L + 1./a1*(un1d-v1L);
															t1Rd	= 1./rho1R - 1./a1*(un1d-v1R);
														}

                            a2	= 1.01*a2;
														if(direction == 0) {
                            	un2d	= 0.5*(u2L+u2R)-0.5/a2*(pi2R-pi2L);
															pi2d	= 0.5*(pi2R+pi2L)-0.5*a2*(u2R-u2L);
															t2Ld	= 1./rho2L + 1./a2*(un2d-u2L);
															t2Rd	= 1./rho2R - 1./a2*(un2d-u2R);
														}
														else if(direction == 1) {
															un2d	= 0.5*(v2L+v2R)-0.5/a2*(pi2R-pi2L);
															pi2d	= 0.5*(pi2R+pi2L)-0.5*a2*(v2R-v2L);
															t2Ld	= 1./rho2L + 1./a2*(un2d-v2L);
															t2Rd	= 1./rho2R - 1./a2*(un2d-v2R);
														}

                        }

                }

                // Second membre
                cible   = -pi1d*(alpha1R-alpha1L) -pi2d*(alpha2R-alpha2L);

//                 cout << "cible = " << cible << endl;

                // Bornes pour la recherche de la vitesse u1e telles que l'ecoulement soit subsonique en vitesses relatives
                cLmax   = max(un1d-a1*t1Ld,un2d-a2*t2Ld);
                cRmin   = min(un1d+a1*t1Rd,un2d+a2*t2Rd);

//                 cout << "cLmax= " << cLmax << endl;
//                 cout << "cRmin= " << cRmin << endl;


                // Bornes de la fonctions Psi
                borninf = Psi(cLmax, a1, alpha1L, alpha1R, un1d, t1Ld, t1Rd, a2, alpha2L, alpha2R, un2d, t2Ld, t2Rd);

                bornsup = Psi(cRmin, a1, alpha1L, alpha1R, un1d, t1Ld, t1Rd, a2, alpha2L, alpha2R, un2d, t2Ld, t2Rd);


//                 cout << "borninf= " << borninf << endl;
//                 cout << "bornsup= " << bornsup << endl;


        } while (cible-borninf <= 0.02*(bornsup-borninf) || bornsup-cible <=0.02*(bornsup-borninf));


        // LE POINT FIXE: On cherche u1e dans l'intervalle [cLmax,cRmin] tel que Psi(u1e)=cible

	u1e	= Newton (cible, a1, alpha1L, alpha1R, un1d, t1Ld, t1Rd, a2, alpha2L, alpha2R, un2d, t2Ld, t2Rd, eps, ptfix);

        //u1e	= Dichotomie(cible, a1, alpha1L, alpha1R, un1d, t1Ld, t1Rd, a2, alpha2L, alpha2R, un2d, t2Ld, t2Rd, eps, ptfix);



	// Enfin les flux !

	// Phase 2: solveur de Riemann decale en vitesse de u1e:
	if(direction == 0) {
		Riemannsoldec (-u1e,
			 	alpha2L, alpha2R, 1./rho2L, 1./rho2R, u2L-u1e, u2R-u1e, pi2L, pi2R, E2L-(u2L-u1e)*u1e-u1e*u1e/2., E2R-(u2R-u1e)*u1e-u1e*u1e/2., un2d-u1e, pi2d, t2Ld, t2Rd,
			 	a2, alpha2m, tau2m, w2m, pi2m, E2m, alpha2p, tau2p, w2p, pi2p, E2p, dissip
				);
	}
	else if(direction == 1) {
		Riemannsoldec (-u1e,
			 	alpha2L, alpha2R, 1./rho2L, 1./rho2R, v2L-u1e, v2R-u1e, pi2L, pi2R, E2L-(v2L-u1e)*u1e-u1e*u1e/2., E2R-(v2R-u1e)*u1e-u1e*u1e/2., un2d-u1e, pi2d, t2Ld, t2Rd,
			 	a2, alpha2m, tau2m, w2m, pi2m, E2m, alpha2p, tau2p, w2p, pi2p, E2p, dissip
				);
	}
	// recalage des vitesses:
	u2m	= w2m + u1e;
	u2p	= w2p + u1e;
	E2m	= E2m +(u2m-u1e)*u1e + u1e*u1e/2.;
	E2p	= E2p +(u2p-u1e)*u1e + u1e*u1e/2.;

	// Phase 1
	if(direction == 0) {
		RiemannsolP1 (0.,
		      	alpha1L, alpha1R, 1./rho1L, 1./rho1R, u1L, u1R, pi1L, pi1R, E1L, E1R,
		      	a1, u1e, alpha1m, tau1m, u1m, pi1m, E1m, alpha1p, tau1p, u1p, pi1p, E1p
		     	);
	}
	else if(direction == 1) {
		RiemannsolP1 (0.,
		      	alpha1L, alpha1R, 1./rho1L, 1./rho1R, v1L, v1R, pi1L, pi1R, E1L, E1R,
		      	a1, u1e, alpha1m, tau1m, u1m, pi1m, E1m, alpha1p, tau1p, u1p, pi1p, E1p
		     	);
	}


	// Les flux conservatifs
	if(direction == 0) {
  	fWmoins	= Etat(0.,alpha1m/tau1m*u1m,alpha1m/tau1m*u1m*u1m+alpha1m*pi1m,
									 0.5*u1m*(UL.alrhov1/UL.alrho1 + UR.alrhov1/UR.alrho1) -
									 0.5*std::abs(u1m)*(UR.alrhov1/UR.alrho1 - UL.alrhov1/UL.alrho1) - u1m*UL.alrhov1/UL.alrho1,
									 alpha1m/tau1m*E1m*u1m+alpha1m*pi1m*u1m,alpha2m/tau2m*u2m,alpha2m/tau2m*u2m*u2m+alpha2m*pi2m,
									 0.5*u2m*(UL.alrhov2/UL.alrho2 + UR.alrhov2/UR.alrho2) -
									 0.5*std::abs(u2m)*(UR.alrhov2/UR.alrho2 - UL.alrhov2/UL.alrho2) - u2m*UL.alrhov2/UL.alrho1,
									 alpha2m/tau2m*E2m*u2m+alpha2m*pi2m*u2m);

		fWplus	= Etat(0.,alpha1p/tau1p*u1p,alpha1p/tau1p*u1p*u1p+alpha1p*pi1p,
									 0.5*u1p*(UL.alrhov1/UL.alrho1 + UR.alrhov1/UR.alrho1) -
									 0.5*std::abs(u1p)*(UR.alrhov1/UR.alrho1 - UL.alrhov1/UL.alrho1) - u1p*UR.alrhov1/UR.alrho1,
									 alpha1p/tau1p*E1p*u1p+alpha1p*pi1p*u1p,alpha2p/tau2p*u2p,alpha2p/tau2p*u2p*u2p+alpha2p*pi2p,
									 0.5*u2p*(UL.alrhov2/UL.alrho2 + UR.alrhov2/UR.alrho2) -
									 0.5*std::abs(u2p)*(UR.alrhov2/UR.alrho2 - UL.alrhov2/UL.alrho2) - u2p*UR.alrhov2/UR.alrho1,
									 alpha2p/tau2p*E2p*u2p+alpha2p*pi2p*u2p);
	}
	else if(direction == 1) {
		fWmoins	= Etat(0.,alpha1m/tau1m*u1m,
									 0.5*u1m*(UL.alrhou1/UL.alrho1 + UR.alrhou1/UR.alrho1) -
									 0.5*std::abs(u1m)*(UR.alrhou1/UR.alrho1 - UL.alrhou1/UL.alrho1) - u1m*UL.alrhou1/UL.alrho1,
									 alpha1m/tau1m*u1m*u1m+alpha1m*pi1m,alpha1m/tau1m*E1m*u1m+alpha1m*pi1m*u1m,alpha2m/tau2m*u2m,
									 0.5*u2m*(UL.alrhou2/UL.alrho2 + UR.alrhou2/UR.alrho2) -
									 0.5*std::abs(u2m)*(UR.alrhou2/UR.alrho2 - UL.alrhou2/UL.alrho2) - u2m*UL.alrhou2/UL.alrho1,
									 alpha2m/tau2m*u2m*u2m+alpha2m*pi2m,alpha2m/tau2m*E2m*u2m+alpha2m*pi2m*u2m);

		fWplus	= Etat(0.,alpha1p/tau1p*u1p,
									 0.5*u1p*(UL.alrhou1/UL.alrho1 + UR.alrhou1/UR.alrho1) -
									 0.5*std::abs(u1p)*(UR.alrhou1/UR.alrho1 - UL.alrhou1/UL.alrho1) - u1p*UR.alrhou1/UR.alrho1,
									 alpha1p/tau1p*u1p*u1p+alpha1p*pi1p,alpha1p/tau1p*E1p*u1p+alpha1p*pi1p*u1p,alpha2p/tau2p*u2p,
									 0.5*u2p*(UL.alrhou2/UL.alrho2 + UR.alrhou2/UR.alrho2) -
									 0.5*std::abs(u2p)*(UR.alrhou2/UR.alrho2 - UL.alrhou2/UL.alrho2) - u2p*UR.alrhou2/UR.alrho1,
									 alpha2p/tau2p*u2p*u2p+alpha2p*pi2p,alpha2p/tau2p*E2p*u2p+alpha2p*pi2p*u2p);
	}


  // Termes non conservatifs des flux
	pidxalpha2	= pi2d*(alpha2R-alpha2L)+psi(u1e, a2, alpha2L, alpha2R, un2d, t2Ld, t2Rd);

	Etat H;
	if(direction == 0) {
		H 	= Etat(-u1e*(alpha1R-alpha1L), 0., -pidxalpha2, 0., -u1e*pidxalpha2, 0., +pidxalpha2, 0., u1e*pidxalpha2);
	}
	else if(direction == 1) {
		H 	= Etat(-u1e*(alpha1R-alpha1L), 0., 0., -pidxalpha2, -u1e*pidxalpha2, 0., 0., +pidxalpha2, u1e*pidxalpha2);
	}

	fWmoins	= (u1e<0.) ? fWmoins-H : fWmoins;
	fWplus	= (u1e>0.) ? fWplus+H : fWplus;

	// Vitesse d'onde maximale pour le calcul de la CFL
	return max(max(fabs(u1L-a1/rho1L),fabs(u1R+a1/rho1R)),max(fabs(u2L-a2/rho2L),fabs(u2R+a2/rho2R)));
}
