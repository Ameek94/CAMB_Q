module GaussianPotential
  ! Gaussian-process RBF Kernel potential implementation in Fortran
  ! Self-contained linear solver replacing external LAPACK
  use, intrinsic :: iso_fortran_env, only: wp => real64
  implicit none
  private
  public :: GPotentialRBF_type

  type :: GPotentialRBF_type
    integer :: N = 0
    real(wp), allocatable :: phi_train(:)
    real(wp), allocatable :: alpha(:)
    real(wp) :: length_scale = 0.5_wp
    real(wp) :: prior_mean   = 0.0_wp
    real(wp) :: variance     = 1.0_wp
    real(wp) :: noise        = 1e-6_wp
  contains
    procedure :: init => GP_init
    procedure :: V    => GP_V
    procedure :: Vd   => GP_Vd
    procedure :: Vdd  => GP_Vdd
  end type GPotentialRBF_type

contains

  subroutine GP_init(this, phi_vals, V_vals, ls, prior_mean, var, noise_in)
    class(GPotentialRBF_type), intent(inout) :: this
    real(wp), intent(in) :: phi_vals(:), V_vals(:)
    real(wp), intent(in), optional :: ls, prior_mean, var, noise_in
    integer :: i,j
    real(wp),allocatable :: K(:,:)

    this%N = size(phi_vals)
    allocate(this%phi_train(this%N), this%alpha(this%N))
    if(present(ls))      this%length_scale = ls
    if(present(prior_mean)) this%prior_mean = prior_mean
    if(present(var))     this%variance   = var
    if(present(noise_in))this%noise        = noise_in

    this%phi_train = phi_vals
    this%alpha     = (V_vals - this%prior_mean)

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

  function GP_V(this,phi_star) result(vout)
    class(GPotentialRBF_type), intent(in) :: this
    real(wp), intent(in) :: phi_star
    real(wp) :: vout
    integer :: i
    real(wp) :: k_i

    vout = this%prior_mean !0.0_wp
    do i=1,this%N
      k_i = this%variance*exp(-0.5_wp*(phi_star-this%phi_train(i))**2/this%length_scale**2)
      vout = vout + k_i*this%alpha(i)
    end do
  end function GP_V

  function GP_Vd(this,phi_star) result(v1)
    class(GPotentialRBF_type), intent(in) :: this
    real(wp), intent(in) :: phi_star
    real(wp) :: v1
    integer :: i
    real(wp) :: x,k_i,dk_i

    v1 = 0.0_wp
    do i=1,this%N
      x   = phi_star - this%phi_train(i)
      k_i = this%variance*exp(-0.5_wp*x**2/this%length_scale**2)
      dk_i= - x/this%length_scale**2 * k_i
      v1 = v1 + dk_i*this%alpha(i)
    end do
  end function GP_Vd

  function GP_Vdd(this,phi_star) result(v2)
    class(GPotentialRBF_type), intent(in) :: this
    real(wp), intent(in) :: phi_star
    real(wp) :: v2
    integer :: i
    real(wp) :: x,k_i,d2k_i

    v2 = 0.0_wp
    do i=1,this%N
      x    = phi_star - this%phi_train(i)
      k_i  = this%variance*exp(-0.5_wp*x**2/this%length_scale**2)
      d2k_i= (x**2/this%length_scale**4 - 1.0_wp/this%length_scale**2)*k_i
      v2 = v2 + d2k_i*this%alpha(i)
    end do
  end function GP_Vdd

end module GaussianPotential

!=======================================================================
! program test_gp
!   use GaussianPotential
!   use, intrinsic :: iso_fortran_env, only: wp => real64
!   implicit none
!   type(GPotentialRBF_type) :: gp
!   real(wp), allocatable :: phi_train(:), V_train(:)
!   real(wp) :: phi_star,v,vd,vdd
!   integer :: N,i

!   ! Sample: V(phi)=sin(phi)
!   N=5; allocate(phi_train(N),V_train(N))
!   do i=1,N
!     phi_train(i) = (i-1)*0.25_wp
!     V_train(i)   = sin(phi_train(i))
!   end do



!   call gp%init(phi_train,V_train)

!   write(*,*) 'Training data:'
!   write(*,*) 'phi_train = ', phi_train
!   write(*,*) 'V_train   = ', V_train
!   write(*,*) 'Length scale = ', gp%length_scale
!   write(*,*) 'Variance     = ', gp%variance
!   write(*,*) 'Noise        = ', gp%noise
!   write(*,*) '----------------------------------'

!   phi_star = 0.33_wp
!   v   = gp%V(phi_star)
!   vd  = gp%Vd(phi_star)
!   vdd = gp%Vdd(phi_star)

!   write(*,*) 'phi* =', phi_star
!   write(*,*) 'GP V   =', v,   ' exact =', sin(phi_star)
!   write(*,*) 'GP Vd  =', vd,  ' exact =', cos(phi_star)
!   write(*,*) 'GP Vdd =', vdd, ' exact =', -sin(phi_star)

!   phi_star = 0.66_wp
!   v   = gp%V(phi_star)
!   vd  = gp%Vd(phi_star)
!   vdd = gp%Vdd(phi_star)

!   write(*,*) 'phi* =', phi_star
!   write(*,*) 'GP V   =', v,   ' exact =', sin(phi_star)
!   write(*,*) 'GP Vd  =', vd,  ' exact =', cos(phi_star)
!   write(*,*) 'GP Vdd =', vdd, ' exact =', -sin(phi_star)


!   phi_star = 0.99_wp
!   v   = gp%V(phi_star)
!   vd  = gp%Vd(phi_star)
!   vdd = gp%Vdd(phi_star)

!   write(*,*) 'phi* =', phi_star
!   write(*,*) 'GP V   =', v,   ' exact =', sin(phi_star)
!   write(*,*) 'GP Vd  =', vd,  ' exact =', cos(phi_star)
!   write(*,*) 'GP Vdd =', vdd, ' exact =', -sin(phi_star)

! end program test_gp
