/* -------------------------------------------------------------------------
    Contributing author and copyright for this file:
    Anmol Sikka
    University of Maryland College Park
    anmolsikka09@gmail.com
------------------------------------------------------------------------- */

// #include <iostream>
#include <cmath>
#include "spherical_harmonics.h" 
#include <Eigen/Dense>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
// #define EIGEN_DONT_PARALLELIZE

#define __STDCPP_WANT_MATH_SPEC_FUNCS__


// Intiator
spherical_harmonics::spherical_harmonics(double radius, double susceptibilty, Eigen::Vector3d H0_vec, Eigen::Vector3d SEP_vec, Eigen::Vector3d M_i_vec, Eigen::Vector3d M_j_vec){

    // variable assignment
    a=radius;
    susc=susceptibilty;
    H0=H0_vec;
    SEP=SEP_vec;
    M_i=M_i_vec;
    M_j=M_j_vec;

    // look up table for number of multipoles for a give susc
    if (susc<4){
        L=10;}
    else if (susc>=4 && susc<7){
        L=20;}
    else if (susc>=7 && susc<12){
        L=30;}
    else if (susc>=12 && susc<20){
        L=40;}
    else if (susc>=20 && susc<35){
        L=45;}
    else if (susc>=35 && susc<55){
        L=50;}
    else if (susc>=55 && susc<75){
        L=75;}
    else if (susc>=75 && susc<100){
        L=80;}

    double mu=(1+susc)*MU0;
    susc_eff=3*susc/(susc+3);
    sep = SEP.norm();

    double c = sep/a; // separation distance scaled with a

    // axis definition
    z_cap=SEP/sep;

    x_cap=(H0-H0.dot(z_cap)*z_cap);

    if (x_cap.norm()==0){
        x_cap<<z_cap[2], 0, -z_cap[0];
        y_cap=z_cap.cross(x_cap);
    }
    else{
        x_cap=x_cap/x_cap.norm();

        y_cap=z_cap.cross(x_cap);
    }


    H_prll=H0.dot(z_cap)/H0.norm();
    H_perp=H0.dot(x_cap)/H0.norm();

    //////////////////////////////////////////////////
    // MATRIX FORMULATION FOR SPHERICAL HARMONICS
    //////////////////////////////////////////////////

    for (int m= 0; m < 2; m++){
        Eigen::MatrixXd X(L,L), Delta_m(L,L), Gamma_m(L,L); 
        for (int i = 0; i < L; i++){
            for (int j = 0; j < L; j++){
                // X matrix
                if (i==j){   
                    X(i,j)=(i+1)*(mu/MU0) + (i+1) + 1;
                }
                else{   X(i,j)=0;}
                
                // Delta and Gamma matrix
                Delta_m(i,j)=std::pow((-1),((i+1)+m))*((i+1)*(mu/MU0)-(i+1))*nchoosek(i+1+j+1, j+1-m)/std::pow(c,(i+1+j+1+1));
                Gamma_m(i,j)=std::pow((-1), (i+1+j+1))*Delta_m(i,j);
            } 
        }
        // 2L X 2L Matrix
        Eigen::MatrixXd Am(2*L, 2*L);
        Am.block(0,0,L,L)=X;
        Am.block(0,L,L,L)=Delta_m;
        Am.block(L,0,L,L)=Gamma_m;
        Am.block(L,L,L,L)=X;

        //qm vector
        Eigen::VectorXd qm(L);
        qm=Eigen::VectorXd::Zero(L);
        if (m==0){
            qm(0)=-H_prll*(1-mu/MU0);
        }
        else if (m==1){
            qm(0)=H_perp*(1-mu/MU0);
        }
        
        //2L Q vector
        Eigen::VectorXd Qm(2*L);
        Qm.block(0,0,L,1)=qm;
        Qm.block(L,0,L,1)=qm;

        //solve linear system
        Eigen::VectorXd Beta_m(2*L);
        
        Beta_m=Am.colPivHouseholderQr().solve(Qm);
        if (m==0){
            Beta1_0=Beta_m.block(0,0,L,1);
            Beta2_0=Beta_m.block(L,0,L,1);
        }
        else if (m==1){
            Beta1_1=Beta_m.block(0,0,L,1);
            Beta2_1=Beta_m.block(L,0,L,1);
        }
    };

    //////////////////////////////////////////////////
    //adjust two-body dipole moments
    //////////////////////////////////////////////////

    double Beta_01_dip=  M_i.dot(z_cap)/(4*M_PI*a*a*a*H0.norm());
    double Beta_11_dip= -M_i.dot(x_cap)/(4*M_PI*a*a*a*H0.norm());

    double Beta_02_dip=  M_j.dot(z_cap)/(4*M_PI*a*a*a*H0.norm());
    double Beta_12_dip= -M_j.dot(x_cap)/(4*M_PI*a*a*a*H0.norm());

    Eigen::Matrix2d A0(2, 2), A1(2, 2);

    A0<<    mu/MU0 + 2 , (-1)*(mu/MU0-1)*nchoosek(2,1)/std::pow(c,3),
            (-1)*(mu/MU0-1)*nchoosek(2,1)/std::pow(c,3),  mu/MU0 + 2;

    A1<<    mu/MU0 + 2 , (mu/MU0-1)*nchoosek(2,2)/std::pow(c,3),
            (mu/MU0-1)*nchoosek(2,2)/std::pow(c,3),  mu/MU0 + 2;
    
    double Beta_01_2Bdip, Beta_02_2Bdip, Beta_11_2Bdip, Beta_12_2Bdip;
    
    Eigen::Vector2d Q0, Q1, Beta_0_2Bdip, Beta_1_2Bdip;
    Q0 <<  -H_prll*(1-mu/MU0), -H_prll*(1-mu/MU0);
    Q1 <<  H_perp*(1-mu/MU0), H_perp*(1-mu/MU0);
    
    Eigen::Matrix2d A0_inv(2, 2), A1_inv(2, 2);

    A0_inv<<     A0(1, 1), -A0(0, 1),
                -A0(1, 0),  A0(0, 0);

    A0_inv /= A0.determinant();
    
    A1_inv<<     A1(1, 1), -A1(0, 1),
                -A1(1, 0),  A1(0, 0);

    A1_inv /= A1.determinant();

    Beta_0_2Bdip=A0_inv*Q0;
    Beta_1_2Bdip=A1_inv*Q1;

    Beta_01_2Bdip=Beta_0_2Bdip(0);
    Beta_02_2Bdip=Beta_0_2Bdip(1);
    Beta_11_2Bdip=Beta_1_2Bdip(0);
    Beta_12_2Bdip=Beta_1_2Bdip(1);

    double mir, mjr, mumu;

    mir=M_i.dot(SEP)/sep;
    mjr=M_j.dot(SEP)/sep;
    mumu = M_i.dot(M_j);

    double K = 3e-7/sep/sep/sep/sep;

    F_dip2B[0] = K*(mir*M_j[0]+mjr*M_i[0]+(mumu-5*mjr*mir)*SEP[0]/sep);
    F_dip2B[1] = K*(mir*M_j[1]+mjr*M_i[1]+(mumu-5*mjr*mir)*SEP[1]/sep);
    F_dip2B[2] = K*(mir*M_j[2]+mjr*M_i[2]+(mumu-5*mjr*mir)*SEP[2]/sep);

    Beta1_0[0]=Beta1_0[0] + Beta_01_dip - Beta_01_2Bdip;
    Beta2_0[0]=Beta2_0[0] + Beta_02_dip - Beta_02_2Bdip;
    Beta1_1[0]=Beta1_1[0] + Beta_11_dip - Beta_11_2Bdip;
    Beta2_1[0]=Beta2_1[0] + Beta_12_dip - Beta_12_2Bdip;

    //////////////////////////////////////////////////
    // Quadrature implementation
    //////////////////////////////////////////////////

    double x_error, y_error, z_error;

    F = Eigen::Vector3d::Zero();

    auto fx_integrand = [&](double th) {
        return fx_int(th);
    };
    auto fy_integrand = [&](double th) {
        return fy_int(th);
    };
    auto fz_integrand = [&](double th) {
        return fz_int(th);
    };

    F[0] = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(fx_integrand, 0, M_PI, 5, 1e-4, &x_error)*MU0;
    // std::cout<<"x error "<<x_error<<std::endl<<std::endl;

    F[1] = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(fy_integrand, 0, M_PI, 5, 1e-4, &y_error)*MU0;
    // std::cout<<"z error "<<z_error<<std::endl<<std::endl;

    F[2] = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(fz_integrand, 0, M_PI, 5, 1e-4, &z_error)*MU0;
    // std::cout<<"z error "<<z_error<<std::endl<<std::endl;

    F_act_coord=F[0]*x_cap + F[1]*y_cap + F[2]*z_cap;
    F_act_coord=a*a*H0.norm()*H0.norm()*MU0*F_act_coord;
}

/////////////////////////////////////////
// Get functions to get force from the class
////////////////////////////////////////

Eigen::Vector3d spherical_harmonics::get_force(){
    return F;
}

Eigen::Vector3d spherical_harmonics::get_force_actual_coord(){
    return F_act_coord;
}

Eigen::Vector3d spherical_harmonics::get_force_2B_corrections(){
    return F_dip2B;
}

/////////////////////////////////////////
// Permutation function
////////////////////////////////////////

double spherical_harmonics::nchoosek(int n, int k){
    if (k > n) return 0;
    if (k * 2 > n) k = n-k;
    if (k == 0) return 1;

    int result = n;
    for( int i = 2; i <= k; ++i ) {
        result *= (n-i+1);
        result /= i;
    }
    return result;
}

/////////////////////////////////////////
// FORCE INTEGRANDS
////////////////////////////////////////

double spherical_harmonics::fx_int(double theta){
    
    Eigen::Vector3d magABC, magUVW;
    Eigen::Vector2d magPQ;
    double A,B,C,P,Q;
    magABC = mag_ABC(a,theta);
    A=magABC[0];
    B=magABC[1];
    C=magABC[2];
    magPQ = mag_PQ(a,theta);
    P=magPQ[0];
    Q=magPQ[1];

    double term1, term2, term3, res;
    term1 = C*P;
    term2 = (B*P + A*Q);
    term3 = (-A*P+B*Q);
    res = -M_PI*(term1 - term2*std::cos(theta) + term3*std::sin(theta))*std::sin(theta);
    return res/MU0;
}

double spherical_harmonics::fy_int(double theta){
    
    Eigen::Vector3d magABC, magUVW;
    Eigen::Vector2d magPQ;
    double P,Q,U,V,W;
    magPQ = mag_PQ(a,theta);
    P=magPQ[0];
    Q=magPQ[1];
    magUVW = mag_UVW(a,theta);
    U=magUVW[0];
    V=magUVW[1];
    W=magUVW[2];

    double term1, term2, term3, res;
    term1 = P*W;
    term2 = (Q*U + P*V);
    term3 = (-P*U + Q*V);
    res = -M_PI*(term1 - term2*std::cos(theta) - term3*std::sin(theta))*std::sin(theta);
    return res/MU0;
}

double spherical_harmonics::fz_int(double theta){
    
    Eigen::Vector3d magABC, magUVW;
    Eigen::Vector2d magPQ;
    double A,B,C,P,Q,U,V,W;
    magABC = mag_ABC(a,theta);
    A=magABC[0];
    B=magABC[1];
    C=magABC[2];
    magPQ = mag_PQ(a,theta);
    P=magPQ[0];
    Q=magPQ[1];
    magUVW = mag_UVW(a,theta);
    U=magUVW[0];
    V=magUVW[1];
    W=magUVW[2];
    
    double term2, term3, res;
    term2 = 4*A*A-4*B*B-4*C*C+8*P*P-8*Q*Q+8*P*U+3*U*U-8*Q*V-3*V*V-W*W;
    term3 = 2*(4*A*B+8*P*Q+4*Q*U+4*P*V+3*U*V);
    res = (M_PI/8)*(term2*std::cos(theta) - term3*std::sin(theta))*std::sin(theta);
    return res/MU0;
}


/////////////////////////////////////////
// MAXWELL STRESS TENSOR FUNCTIONS
////////////////////////////////////////

Eigen::Vector3d spherical_harmonics::mag_ABC(double r, double theta){
    Eigen::Vector3d mag_ABC_res;
    double mag_A_res=0;
    double mag_B_res=0;
    double mag_C_res=0;
    for (int l = 1; l < L+1; l++){
        double Hrs1=0;
        double Hths1=0;
        double Hphs1=0;
        for (int s = 1; s < L+1; s++){
            double Psm=lpmn_cos(1, s, theta);
            double dPsm=d_lpmn_cos(1, s, theta);
            double ls_choose_sm=nchoosek(l+s, s+1);
            double r_pow_s_1=std::pow(r/a, (s-1));
            double sep_pow=std::pow(sep/a, l+s+1);
            double minus_one_pow=std::pow(-1, s+1);   
            double additional=(minus_one_pow)*r_pow_s_1*Psm/(sep_pow);
            Hrs1=Hrs1 + additional*(ls_choose_sm)*s;
            Hths1=Hths1 + minus_one_pow*r_pow_s_1*dPsm*(ls_choose_sm)/(sep_pow);
            Hphs1=Hphs1 + additional*(ls_choose_sm)/std::sin(theta);
        }
        double Plm=lpmn_cos(1, l, theta);
        double dPlm=d_lpmn_cos(1, l, theta);
        double r_pow_l2=std::pow(r/a, l+2);
        mag_A_res=mag_A_res + ((l+1)*Beta1_1[l-1]*(Plm/r_pow_l2) - Beta2_1[l-1]*Hrs1);
        mag_B_res=mag_B_res - (Beta1_1[l-1]*(dPlm/r_pow_l2) + Beta2_1[l-1]*Hths1);
        mag_C_res=mag_C_res + (Beta1_1[l-1]*(Plm/(r_pow_l2*std::sin(theta))) + Beta2_1[l-1]*Hphs1);
    }
    mag_A_res=mag_A_res + H_perp*std::sin(theta);
    mag_B_res=mag_B_res + H_perp*std::cos(theta);
    mag_C_res=mag_C_res - H_perp;

    mag_ABC_res << mag_A_res, mag_B_res, mag_C_res;

    return mag_ABC_res;
}

Eigen::Vector2d spherical_harmonics::mag_PQ(double r, double theta){
    Eigen::Vector2d mag_PQ_res;
    double mag_P_res=0;
    double mag_Q_res=0;
    for (int l = 1; l < L+1; l++){
        double Hrs0=0;
        double Hths0=0;
        for (int s = 0; s < L+1; s++){
            double Psm=lpmn_cos(0, s, theta);
            double dPsm=d_lpmn_cos(0, s, theta);
            double ls_choose_sm=nchoosek(l+s, s);
            double r_pow_s_1=std::pow(r/a, (s-1));
            double sep_pow=std::pow(sep/a, l+s+1);
            double minus_one_pow=std::pow(-1, s);

            Hrs0=Hrs0 + (minus_one_pow)*s*r_pow_s_1*Psm*(ls_choose_sm)/(sep_pow);
            Hths0=Hths0 + (minus_one_pow)*r_pow_s_1*dPsm*(ls_choose_sm)/(sep_pow);
        }
        double Plm=lpmn_cos(0, l, theta);
        double dPlm=d_lpmn_cos(0, l, theta);
        double r_pow_l2=std::pow(r/a, l+2);
        mag_P_res=mag_P_res + ((l+1)*Beta1_0[l-1]*(Plm/r_pow_l2) - Beta2_0[l-1]*Hrs0);
        mag_Q_res=mag_Q_res - (Beta1_0[l-1]*(dPlm/r_pow_l2) + Beta2_0[l-1]*Hths0);
    }
    mag_P_res= mag_P_res + H_prll*std::cos(theta);
    mag_Q_res= mag_Q_res - H_prll*std::sin(theta);

    mag_PQ_res << mag_P_res, mag_Q_res;
    return mag_PQ_res;
}

Eigen::Vector3d spherical_harmonics::mag_UVW(double r, double theta){
    double mag_U, mag_V, mag_W;
    Eigen::Vector3d mag_UVW_res;
    double beta_1, beta_2;
    beta_1 = (M_i.dot(y_cap)/(4*M_PI*a*a*a*H0.norm()));
    beta_2 = (M_j.dot(y_cap)/(4*M_PI*a*a*a*H0.norm()));

    double r2 = std::sqrt(r*r + sep*sep - 2*sep*r*std::cos(theta));

    mag_U= 2*beta_1*std::sin(theta)/(r/a*r/a*r/a) + 2*beta_2*std::sin(theta)/(r2/a*r2/a*r2/a) - 3*sep/r2*std::sin(theta)*std::cos(theta);
    mag_V= - beta_1*std::cos(theta)/(r/a*r/a*r/a) - beta_2*std::cos(theta)/(r2/a*r2/a*r2/a) + 3*sep/r2*std::sin(theta)*std::sin(theta);
    mag_W= - beta_1/(r/a*r/a*r/a) - beta_2/(r2/a*r2/a*r2/a);
    
    mag_UVW_res << mag_U, mag_V, mag_W;
    return mag_UVW_res;
}

/////////////////////////////////////////////////////////
//define associate legendre functions for cos(theta)
/////////////////////////////////////////////////////////

double spherical_harmonics::lpmn_cos(int m, int n, double theta){
    return boost::math::legendre_p(n, m, std::cos(theta));
}

double spherical_harmonics::d_lpmn_cos(int m, int n, double theta){
    return ((m-n-1)*lpmn_cos(m,n+1,theta) + (n+1)*std::cos(theta)*lpmn_cos(m,n,theta))/(-std::sin(theta));
}