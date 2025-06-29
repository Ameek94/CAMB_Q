module GaussianPotential
  ! Gaussian-process RBF Kernel potential implementation in Fortran
  ! Self-contained linear solver replacing external LAPACK
  use, intrinsic :: iso_fortran_env, only: wp => real64
  ! use MathUtils, only :: Integrate_Trapz, Integrate_Romberg
  implicit none
  private
  public :: GP_logVprime_RBF_type

  type :: GP_logVprime_RBF_type
    integer :: N = 0
    real(wp), allocatable :: phi_train(:)
    real(wp), allocatable :: alpha(:)
    real(wp) :: length_scale = 0.1_wp
    real(wp) :: prior_mean   = 0.0_wp
    real(wp) :: variance     = 1.0_wp
    real(wp) :: noise        = 1e-6_wp
    real(wp) :: phi_star = 0.25_wp ! Last phi value used for prediction
    real(wp) :: V_star = 0.8260998715675062_wp ! Last V value used for prediction 0.73256704_wp !
  contains
    procedure :: init => GP_init
    procedure :: GP_predict
    procedure :: GP_predict_derivative
    procedure :: V    => GP_V
    procedure :: Vd   => GP_Vd
    procedure :: Vdd  => GP_Vdd
    ! procedure, private :: MLL => GP_MLL
  end type GP_logVprime_RBF_type

contains

  subroutine GP_init(this, phi_vals, V_vals, ls, prior_mean, var, noise_in)
    class(GP_logVprime_RBF_type), intent(inout) :: this
    real(wp), intent(in) :: phi_vals(:), V_vals(:)
    real(wp), intent(in), optional :: ls, prior_mean, var, noise_in
    integer :: i,j
    real(wp),allocatable :: K(:,:)
    real(wp) :: phistar

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
        K(i,j) = this%variance*exp(-0.5_wp*(phi_vals(i)-phi_vals(j))**2/this%length_scale**2)
      end do
      K(i,i) = K(i,i) + this%noise**2
    end do

    call solve_linear_system(K, this%alpha, this%N)
    deallocate(K)
  end subroutine GP_init

  subroutine solve_linear_system(A, b, N)
    ! Simple Gaussian elimination with partial pivoting: A*x=b, overwrites b
    real(wp), intent(inout) :: A(N,N)
    real(wp), intent(inout) :: b(N)
    integer, intent(in) :: N
    integer :: i,j,k, piv
    real(wp) :: maxval, factor, tmp

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

  function GP_predict(this,phi) result(vout)
    ! this predicts log(-V') at phi using the Gaussian process
    class(GP_logVprime_RBF_type), intent(in) :: this
    real(wp), intent(in) :: phi
    real(wp) :: vout
    integer :: i
    real(wp) :: k_i

    vout = this%prior_mean !0.0_wp
    do i=1,this%N
      k_i = this%variance*exp(-0.5_wp*(phi-this%phi_train(i))**2/this%length_scale**2)
      vout = vout + k_i*this%alpha(i)
    end do

  end function GP_predict

  function GP_predict_derivative(this,phi) result(v1)
    class(GP_logVprime_RBF_type), intent(in) :: this
    real(wp), intent(in) :: phi
    real(wp) :: v1
    integer :: i
    real(wp) :: x,k_i,dk_i

    v1 = 0.0_wp
    do i=1,this%N
      x   = phi - this%phi_train(i)
      k_i = this%variance*exp(-0.5_wp*x**2/this%length_scale**2)
      dk_i= - x/this%length_scale**2 * k_i
      v1 = v1 + dk_i*this%alpha(i)
    end do
  end function GP_predict_derivative

  function GP_V(this, phi) result(vout)
    class(GP_logVprime_RBF_type), intent(in) :: this
    real(wp), intent(in) :: phi
    real(wp) :: vout
    integer :: i, n_points
    real(wp) :: range, dx, x_i, f_old, f_new, integral

    n_points = 50
    vout     = this%V_star
    range    = phi - this%phi_star
    if (abs(range) < 1.0e-6_wp) return

    dx       = range / real(n_points-1, wp)
    integral = 0.0_wp

    x_i   = this%phi_star
    f_old = this%Vd(x_i)
    do i = 1, n_points-1
      x_i   = this%phi_star + dx * real(i, wp)
      f_new = this%Vd(x_i)
      integral = integral + 0.5_wp * dx * (f_old + f_new)
      f_old = f_new
    end do

    vout = this%V_star + integral
  end function GP_V

  function GP_Vd(this, phi) result(vout)
    class(GP_logVprime_RBF_type), intent(in) :: this
    real(wp), intent(in) :: phi
    real(wp) :: vout

    ! square-link: GP_predict returns h = sqrt(-V'), so V' = -h^2
    vout = - (this%GP_predict(phi))**2
  end function GP_Vd

  function GP_Vdd(this, phi) result(v2)
    class(GP_logVprime_RBF_type), intent(in) :: this
    real(wp), intent(in) :: phi
    real(wp) :: v2

    ! derivative of Vd: V'' = -2*h * dh/dφ
    v2 = -2.0_wp * this%GP_predict(phi) * this%GP_predict_derivative(phi)
  end function GP_Vdd

  ! function GP_V(this, phi) result(vout)
  !   class(GP_logVprime_RBF_type), intent(in) :: this
  !   real(wp), intent(in) :: phi
  !   real(wp) :: vout
  !   integer :: i, n_points
  !   real(wp) :: range, dx, x_i, f_old, f_new, integral

  !   n_points = 200

  !   ! Initialize
  !   vout  = this%V_star
  !   range = phi - this%phi_star
  !   if (abs(range) < 1.0e-6_wp) return

  !   ! Precompute step size and integration direction
  !   dx        = range / real(n_points-1, wp)
  !   integral  = 0.0_wp

  !   ! First point
  !   x_i   = this%phi_star
  !   f_old = this%Vd(x_i)

  !   ! Loop over interior & endpoint points:
  !   do i = 1, n_points-1
  !     x_i   = this%phi_star + dx * real(i, wp) !* sign(1.0_wp, range)
  !     f_new = this%Vd(x_i)

  !     ! Trapezoid between (i-1)->i
  !     integral = integral + 0.5_wp * dx * (f_old + f_new)

  !     f_old = f_new
  !   end do

  !   vout = this%V_star + integral
  ! end function GP_V

  ! function GP_Vd(this,phi) result(vout)
  !   class(GP_logVprime_RBF_type), intent(in) :: this
  !   real(wp), intent(in) :: phi
  !   real(wp) :: vout
  !   integer :: i
  !   real(wp) :: x,k_i,dk_i

  !   vout = -exp(this%GP_predict(phi))

  ! end function GP_Vd

  ! function GP_Vdd(this,phi) result(v2)
  !   class(GP_logVprime_RBF_type), intent(in) :: this
  !   real(wp), intent(in) :: phi
  !   real(wp) :: v2
  !   integer :: i
  !   real(wp) :: x,k_i,d2k_i

  !   v2 = -this%GP_predict_derivative(phi) * exp(this%GP_predict(phi))

  ! end function GP_Vdd

end module GaussianPotential

! !=======================================================================
! program test_GP
!   use GaussianPotential
!   use, intrinsic :: iso_fortran_env, only: wp => real64
!   implicit none
!   type(GP_logVprime_RBF_type) :: gp
!   real(wp), allocatable :: phi_vals(:), V_vals(:)
!   real(wp), dimension(:), allocatable :: phi_test
!   real(wp) :: Vval, Vdval, Vddval
!   integer :: i

!   ! Example training data
!   allocate(phi_vals(6), V_vals(6))
!   phi_vals = [0.0_wp,   0.05_wp, 0.1_wp,  0.15_wp, 0.2_wp,  0.25_wp]
!   V_vals   = [0.0_wp, 1.16014591_wp, 1.4744805_wp,  1.47873107_wp, 1.18081867_wp, 0.19975953_wp] ![-5.0_wp,  0.29709156_wp,  0.77661145_wp,  0.78236866_wp,  0.33241596_wp, -3.22128196_wp]

!   call gp%init(phi_vals, V_vals, 0.0735_wp)

!   ! Test points in [0, 0.25]
!   allocate(phi_test(6))
!   phi_test = [0.03_wp, 0.09_wp, 0.12_wp, 0.15_wp, 0.18_wp, 0.21_wp]

!   print '(A10, 3(A12))', 'phi', 'V', 'Vd', 'Vdd'
!   do i = 1, size(phi_test)
!     Vval   = gp%V(phi_test(i))
!     Vdval  = gp%Vd(phi_test(i))
!     Vddval = gp%Vdd(phi_test(i))
!     print '(F8.4, 3(F12.6))', phi_test(i), Vval, Vdval, Vddval
!   end do

! end program test_GP
