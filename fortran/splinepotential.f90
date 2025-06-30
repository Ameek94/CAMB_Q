module GaussianPotential
  ! Gaussian-process RBF Kernel potential implementation in Fortran
  ! Self-contained linear solver replacing external LAPACK
  ! use, intrinsic :: iso_fortran_env, only: dl => real64
  use constants
  ! use MathUtils, only :: Integrate_Trapz, Integrate_Romberg
  implicit none
  private
  public :: GP_logVprime_RBF_type

  integer, parameter :: N_grid = 100
  real(dl), parameter :: PHI_MIN = 0.0_dl, PHI_MAX = 0.5_dl
  real(dl), parameter :: DPHI = (PHI_MAX - PHI_MIN) / real(N_grid-1, dl)
  real(dl), parameter :: INV_DPHI = 1.0_dl / DPHI


  type :: GP_logVprime_RBF_type
    integer :: N = 0
    real(dl), allocatable :: phi_train(:)
    real(dl), allocatable :: alpha(:)
    real(dl) :: length_scale = 0.1_dl
    real(dl) :: prior_mean   = 0.0_dl
    real(dl) :: variance     = 1.0_dl
    real(dl) :: noise        = 1e-6_dl
    real(dl) :: phi_star = 0.25_dl ! Last phi value used for prediction
    real(dl) :: V_star = 0.8260998715675062_dl ! Last V value used for prediction 0.73256704_dl !
    real(dl), allocatable :: phi_grid(:), V_grid(:), Vd_grid(:), Vdd_grid(:)
  contains
    procedure :: init => GP_init
    procedure :: GP_predict
    procedure :: GP_predict_derivative
    procedure :: V    => GP_V_build
    procedure :: Vd   => GP_Vd_build
    procedure :: Vdd  => GP_Vdd_build
    procedure, private :: GP_V
    procedure, private :: GP_Vd
    procedure, private :: GP_Vdd
    ! procedure, private :: MLL => GP_MLL
  end type GP_logVprime_RBF_type

contains

  subroutine GP_init(this, phi_vals, V_vals, ls, prior_mean, var, noise_in)
    class(GP_logVprime_RBF_type), intent(inout) :: this
    real(dl), intent(in) :: phi_vals(:), V_vals(:)
    real(dl), intent(in), optional :: ls, prior_mean, var, noise_in
    integer :: i,j
    real(dl),allocatable :: K(:,:)
    real(dl) :: phi_local

    this%N = size(phi_vals)
    allocate(this%phi_train(this%N), this%alpha(this%N))
    if(present(ls))      this%length_scale = ls
    if(present(prior_mean)) this%prior_mean = prior_mean
    if(present(var))     this%variance   = var
    if(present(noise_in))this%noise        = noise_in

    this%phi_train = phi_vals
    this%alpha     = (V_vals - this%prior_mean)
    this%phi_star = this%phi_train(this%N) ! Initialize with last phi value


    allocate(K(this%N,this%N))
    do i=1,this%N
      do j=1,this%N
        K(i,j) = this%variance*exp(-0.5_dl*(phi_vals(i)-phi_vals(j))**2/this%length_scale**2)
      end do
      K(i,i) = K(i,i) + this%noise**2
    end do

    call solve_linear_system(K, this%alpha, this%N)
    deallocate(K)

    ! allocate(this%phi_grid(N_grid),this%V_grid(N_grid), this%Vd_grid(N_grid), this%Vdd_grid(N_grid))

    ! !!!!! $omp! parallel do default(shared) private(i)


    ! do i=1,N_grid
    !   phi_local = PHI_MIN + real((i-1),dl)*DPHI
    !   this%phi_grid(i) = phi_local
    !   this%V_grid(i)   = this%GP_V_build(phi_local)
    !   this%Vd_grid(i)  = this%GP_Vd_build(phi_local)
    !   this%Vdd_grid(i) = this%GP_Vdd_build(phi_local)
    !   write(*, '(A, F8.4, 3(F12.6))') 'GP_init: phi = ', phi_local, this%V_grid(i), this%Vd_grid(i), this%Vdd_grid(i)
    ! end do


    ! !!!!$omp! end parallel do

  end subroutine GP_init

  subroutine solve_linear_system(A, b, N)
    ! Simple Gaussian elimination with partial pivoting: A*x=b, overwrites b
    real(dl), intent(inout) :: A(N,N)
    real(dl), intent(inout) :: b(N)
    integer, intent(in) :: N
    integer :: i,j,k, piv
    real(dl) :: maxval, factor, tmp

    do k = 1, N-1
      ! pivot
      piv = k
      maxval = abs(A(k,k))
      do i = k+1, N
        if (abs(A(i,k)) > maxval) then
          maxval = abs(A(i,k)); piv = i
        end if
      end do
      if (piv /= k) then
        A([k,piv],:) = A([piv,k],:)
        tmp = b(k); b(k) = b(piv); b(piv) = tmp
      end if
      ! elimination
      do i = k+1, N
        factor = A(i,k)/A(k,k)
        A(i,k:N) = A(i,k:N) - factor*A(k,k:N)
        b(i)     = b(i)     - factor*b(k)
      end do
    end do
    ! back substitution
    do i = N, 1, -1
      do j = i+1, N
        b(i) = b(i) - A(i,j)*b(j)
      end do
      b(i) = b(i)/A(i,i)
    end do
  end subroutine solve_linear_system


! Linear interpolate the V, Vd, and Vdd grids to get values at arbitrary phi
  function GP_V(this,phi) result(vout)
    ! this predicts log(-V') at phi using the Gaussian process
    class(GP_logVprime_RBF_type), intent(in) :: this
    real(dl), intent(in) :: phi
    real(dl) :: vout, t
    integer :: j
    ! real(dl) :: k_i

    if (phi < PHI_MIN) then
      vout = this%V_grid(1)
      return
    elseif (phi > PHI_MAX) then
      vout = this%V_grid(N_grid)
      return
    end if

    ! find integer j such that phi is between phi_grid(j) and phi_grid(j+1)
    j = floor((phi - PHI_MIN) * INV_DPHI) + 1
    if (j<0) then
      write(*,*) 'GP_V: j < 0, phi = ', phi, ' PHI_MIN = ', PHI_MIN
    end if
    t = (phi - (PHI_MIN + (j-1)*DPHI)) * INV_DPHI
    vout =  (1.0_dl - t)*this%V_grid(j) + t*this%V_grid(j+1)
  end function GP_V

  function GP_Vd(this,phi) result(vout)
    class(GP_logVprime_RBF_type), intent(in) :: this
    real(dl), intent(in) :: phi
    real(dl) :: vout, t
    integer :: j

    if (phi < PHI_MIN) then
      vout = 0.0_dl !  this%Vd_grid(1)
      return
    elseif (phi > PHI_MAX) then
      vout = 0.0_dl !this%Vd_grid(N_grid)
      return
    end if

    j = floor((phi - PHI_MIN) * INV_DPHI) + 1
    t = (phi - (PHI_MIN + (j-1)*DPHI)) * INV_DPHI
    vout = (1.0_dl - t)*this%Vd_grid(j) + t*this%Vd_grid(j+1)
  end function GP_Vd

  function GP_Vdd(this,phi) result(vout)
    class(GP_logVprime_RBF_type), intent(in) :: this
    real(dl), intent(in) :: phi
    real(dl) :: vout, t
    integer :: j

    if (phi < PHI_MIN) then
      vout = 0.0_dl !this%Vdd_grid(1)
      return
    elseif (phi > PHI_MAX) then
      vout = 0.0_dl !this%Vdd_grid(N_grid)
      return
    end if

    j = floor((phi - PHI_MIN) * INV_DPHI) + 1
    t = (phi - (PHI_MIN + (j-1)*DPHI)) * INV_DPHI
    vout = (1.0_dl - t)*this%Vdd_grid(j) + t*this%Vdd_grid(j+1)

  end function GP_Vdd

  pure function GP_predict(this,phi) result(vout)
    ! this predicts log(-V') at phi using the Gaussian process
    class(GP_logVprime_RBF_type), intent(in) :: this
    real(dl), intent(in) :: phi
    real(dl) :: vout
    integer :: i
    real(dl) :: k_i

    vout = this%prior_mean !0.0_dl
    do i=1,this%N
      k_i = this%variance*exp(-0.5_dl*(phi-this%phi_train(i))**2/this%length_scale**2)
      vout = vout + k_i*this%alpha(i)
    end do

  end function GP_predict

  pure function GP_predict_derivative(this,phi) result(v1)
    class(GP_logVprime_RBF_type), intent(in) :: this
    real(dl), intent(in) :: phi
    real(dl) :: v1
    integer :: i
    real(dl) :: x,k_i,dk_i

    v1 = 0.0_dl
    do i=1,this%N
      x   = phi - this%phi_train(i)
      k_i = this%variance*exp(-0.5_dl*x**2/this%length_scale**2)
      dk_i= - x/this%length_scale**2 * k_i
      v1 = v1 + dk_i*this%alpha(i)
    end do
  end function GP_predict_derivative

  function GP_V_build(this, phi) result(vout)
    class(GP_logVprime_RBF_type), intent(in) :: this
    real(dl), intent(in) :: phi
    real(dl) :: vout
    integer :: i, n_points
    real(dl) :: range, dx, x_i, f_old, f_new, integral
    ! real(dl), allocatable :: phivals(:), Vvals(:)

    n_points = 40
    vout     = this%V_star
    range    = phi - this%phi_star
    if (abs(range) < 1.0e-6_dl) return

    dx       = range / real(n_points-1, dl)
    integral = 0.0_dl

    ! ! vectorized integral
    ! phivals = this%phi_star + dx * [(real(i, dl), i=0,n_points-1)]
    ! Vvals = this%Vd(phivals)

    ! integral = dx * (sum(Vvals) - 0.5_dl * (Vvals(1) - Vvals(n_points)) )

    x_i   = this%phi_star
    f_old = this%Vd(x_i)
    do i = 1, n_points-1
      x_i   = this%phi_star + dx * real(i, dl)
      f_new = this%Vd(x_i)
      integral = integral + 0.5_dl * dx * (f_old + f_new)
      f_old = f_new
    end do

    vout = this%V_star + integral

  end function GP_V_build

  pure function GP_Vd_build(this, phi) result(vout)
    class(GP_logVprime_RBF_type), intent(in) :: this
    real(dl), intent(in) :: phi
    real(dl) :: vout

    ! square-link: GP_predict returns h = sqrt(-V'), so V' = -h^2
    vout = - (this%GP_predict(phi))**2
  end function GP_Vd_build

  pure function GP_Vdd_build(this, phi) result(v2)
    class(GP_logVprime_RBF_type), intent(in) :: this
    real(dl), intent(in) :: phi
    real(dl) :: v2

    ! derivative of Vd: V'' = -2*h * dh/dφ
    v2 = -2.0_dl * this%GP_predict(phi) * this%GP_predict_derivative(phi)
  end function GP_Vdd_build

  ! function GP_V(this, phi) result(vout)
  !   class(GP_logVprime_RBF_type), intent(in) :: this
  !   real(dl), intent(in) :: phi
  !   real(dl) :: vout
  !   integer :: i, n_points
  !   real(dl) :: range, dx, x_i, f_old, f_new, integral

  !   n_points = 200

  !   ! Initialize
  !   vout  = this%V_star
  !   range = phi - this%phi_star
  !   if (abs(range) < 1.0e-6_dl) return

  !   ! Precompute step size and integration direction
  !   dx        = range / real(n_points-1, dl)
  !   integral  = 0.0_dl

  !   ! First point
  !   x_i   = this%phi_star
  !   f_old = this%Vd(x_i)

  !   ! Loop over interior & endpoint points:
  !   do i = 1, n_points-1
  !     x_i   = this%phi_star + dx * real(i, dl) !* sign(1.0_dl, range)
  !     f_new = this%Vd(x_i)

  !     ! Trapezoid between (i-1)->i
  !     integral = integral + 0.5_dl * dx * (f_old + f_new)

  !     f_old = f_new
  !   end do

  !   vout = this%V_star + integral
  ! end function GP_V

  ! function GP_Vd(this,phi) result(vout)
  !   class(GP_logVprime_RBF_type), intent(in) :: this
  !   real(dl), intent(in) :: phi
  !   real(dl) :: vout
  !   integer :: i
  !   real(dl) :: x,k_i,dk_i

  !   vout = -exp(this%GP_predict(phi))

  ! end function GP_Vd

  ! function GP_Vdd(this,phi) result(v2)
  !   class(GP_logVprime_RBF_type), intent(in) :: this
  !   real(dl), intent(in) :: phi
  !   real(dl) :: v2
  !   integer :: i
  !   real(dl) :: x,k_i,d2k_i

  !   v2 = -this%GP_predict_derivative(phi) * exp(this%GP_predict(phi))

  ! end function GP_Vdd

end module GaussianPotential

! !=======================================================================
! program test_GP
!   use GaussianPotential
!   use, intrinsic :: iso_fortran_env, only: dl => real64
!   implicit none
!   type(GP_logVprime_RBF_type) :: gp
!   real(dl), allocatable :: phi_vals(:), V_vals(:)
!   real(dl), dimension(:), allocatable :: phi_test
!   real(dl) :: Vval, Vdval, Vddval
!   integer :: i

!   ! Example training data
!   allocate(phi_vals(6), V_vals(6))
!   phi_vals = [0.0_dl,   0.05_dl, 0.1_dl,  0.15_dl, 0.2_dl,  0.25_dl]
!   V_vals   = [0.0_dl, 1.16014591_dl, 1.4744805_dl,  1.47873107_dl, 1.18081867_dl, 0.19975953_dl] ![-5.0_dl,  0.29709156_dl,  0.77661145_dl,  0.78236866_dl,  0.33241596_dl, -3.22128196_dl]

!   call gp%init(phi_vals, V_vals, 0.0735_dl)

!   ! Test points in [0, 0.25]
!   allocate(phi_test(6))
!   phi_test = [0.03_dl, 0.09_dl, 0.12_dl, 0.15_dl, 0.18_dl, 0.21_dl]

!   print '(A10, 3(A12))', 'phi', 'V', 'Vd', 'Vdd'
!   do i = 1, size(phi_test)
!     Vval   = gp%V(phi_test(i))
!     Vdval  = gp%Vd(phi_test(i))
!     Vddval = gp%Vdd(phi_test(i))
!     print '(F8.4, 3(F12.6))', phi_test(i), Vval, Vdval, Vddval
!   end do

! end program test_GP
