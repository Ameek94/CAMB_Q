! =============================================================================
! A module for a 1D linear interpolator class.
! =============================================================================
module PotentialInterpolator
  use precision, only: dl
  implicit none
  private ! Make module contents private by default

  ! Make the type public. Its methods are available through the type.
  public :: PotentialInterpolator1D !

  ! Define the "class" for the 1D interpolator
  type :: PotentialInterpolator1D
    private ! Encapsulate the data
    real(dl), allocatable :: x_data(:)
    real(dl), allocatable :: y_data(:)
  contains
    ! MODIFIED: Bind the 'init' method to the 'init_method' subroutine
    procedure :: init => init_method
    ! Bind the 'interpolate' method to the 'interpolate_method' function
    procedure :: interpolate => interpolate_method
  end type PotentialInterpolator1D

contains

  ! ---------------------------------------------------------------------------
  ! "Constructor" method to initialize the interpolator object.
  ! ---------------------------------------------------------------------------
  subroutine init_method(this, x_in, y_in)
    ! MODIFIED: Intent for 'this' is now INOUT
    class(PotentialInterpolator1D), intent(inout) :: this
    real(dl), intent(in), target :: x_in(:)
    real(dl), intent(in), target :: y_in(:)
    integer :: n


    n = size(x_in)
    ! Basic error checking
    if (size(y_in) /= n) then
      write(*,*) 'FATAL ERROR (Interpolator1D init): x and y arrays must have the same size.'
      stop 1
    end if
    if (n < 2) then
        write(*,*) 'FATAL ERROR (Interpolator1D init): Interpolation requires at least 2 points.'
        stop 1
    end if

    ! If the object is being re-initialized, clean up old data first.
    if (allocated(this%x_data)) deallocate(this%x_data)
    if (allocated(this%y_data)) deallocate(this%y_data)

    ! Allocate internal arrays and copy data
    allocate(this%x_data(n))
    allocate(this%y_data(n))
    this%x_data = x_in
    this%y_data = y_in

    ! write (*,*) 'Interpolator initialized with ', n, ' points from ', this%x_data(1), ' to ', this%x_data(n)
  end subroutine init_method

  ! ---------------------------------------------------------------------------
  ! The interpolation method. This implementation is unchanged.
  ! ---------------------------------------------------------------------------
  function interpolate_method(this, x0) result(y0)
    class(PotentialInterpolator1D), intent(in) :: this
    real(dl), intent(in) :: x0
    real(dl) :: y0
    integer :: n, i, l, u

    n = size(this%x_data)

    ! --- Handle points outside the interpolation interval ---
    if (x0 <= this%x_data(1)) then
      y0 = this%y_data(1)
      return
    else if (x0 >= this%x_data(n)) then
      y0 = this%y_data(n)
      return
    end if

    ! --- Use binary search to find the correct interval ---
    l = 1
    u = n
    do while (u - l > 1)
      i = (l + u) / 2
      if (this%x_data(i) > x0) then
        u = i
      else
        l = i
      end if
    end do
    i = u

    ! --- Perform linear interpolation ---
    y0 = this%y_data(i-1) + (this%y_data(i) - this%y_data(i-1)) * &
         (x0 - this%x_data(i-1)) / (this%x_data(i) - this%x_data(i-1))

  end function interpolate_method

end module PotentialInterpolator


! =============================================================================
! test_program.f90
!
! A main program to demonstrate how to use the Interpolator1D class
! with the init method.
! =============================================================================
! =============================================================================
! program test_interpolator
!   ! MODIFIED: The 'use' statement no longer needs to import 'init'
!   use interpolator_1d_mod, only: Interpolator1D
!   use precision, only: dl
!   implicit none

!   ! Declare an object of our interpolator class
!   type(Interpolator1D) :: my_interpolator

!   ! Define the data points for interpolation (must be sorted by x)
!   real(dl), dimension(5) :: x_values = [1.0_dl, 2.0_dl, 3.0_dl, 4.0_dl, 5.0_dl]
!   real(dl), dimension(5) :: y_values = [10.0_dl, 25.0_dl, 30.0_dl, 20.0_dl, 15.0_dl]

!   real(dl) :: x_test, y_result

!   ! 1. Initialize the interpolator object using its init method
!   ! MODIFIED: The call syntax is now object-oriented
!   call my_interpolator%init(x_values, y_values)

!   write(*,*) 'Interpolator initialized with data:'
!   write(*,'(A, 5F8.2)') 'X:', x_values
!   write(*,'(A, 5F8.2)') 'Y:', y_values
!   write(*,*) '-----------------------------------------------'
!   write(*,*) 'Testing interpolation...'
!   write(*,*) '-----------------------------------------------'

!   ! 2. Perform interpolation tests (this part is unchanged)

!   ! Test 1: A point within an interval
!   x_test = 2.5_dl
!   y_result = my_interpolator%interpolate(x_test)
!   write(*, '(A, F6.2, A, F8.4)') 'interpolate at x = ', x_test, ' -> y = ', y_result

!   ! Test 2: A point exactly on a node
!   x_test = 4.0_dl
!   y_result = my_interpolator%interpolate(x_test)
!   write(*, '(A, F6.2, A, F8.4)') 'interpolate at x = ', x_test, ' -> y = ', y_result

!   ! Test 3: A point outside the interval (below the minimum)
!   x_test = 0.5_dl
!   y_result = my_interpolator%interpolate(x_test)
!   write(*, '(A, F6.2, A, F8.4)') 'interpolate at x = ', x_test, ' -> y = ', y_result

!   ! Test 4: A point outside the interval (above the maximum)
!   x_test = 6.0_dl
!   y_result = my_interpolator%interpolate(x_test)
!   write(*, '(A, F6.2, A, F8.4)') 'interpolate at x = ', x_test, ' -> y = ', y_result

!   ! Test 5: A point at the lower boundary
!   x_test = 1.0_dl
!   y_result = my_interpolator%interpolate(x_test)
!   write(*, '(A, F6.2, A, F8.4)') 'interpolate at x = ', x_test, ' -> y = ', y_result

! end program test_interpolator
! ! =============================================================================