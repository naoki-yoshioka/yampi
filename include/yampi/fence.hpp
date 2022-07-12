#ifndef YAMPI_FENCE_HPP
# define YAMPI_FENCE_HPP

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/assertion_mode.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  namespace yampi_fence_detail
  {
    template <typename Derived>
    inline void fence(int const assertion, ::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
    {
      int const error_code = MPI_Win_fence(assertion, window.mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::yampi_fence_detail::fence", environment);
    }
  }

  template <typename Derived>
  inline void fence(::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
  { ::yampi::yampi_fence_detail::fence(0, window, environment); }

  template <typename Derived>
  inline void fence(::yampi::assertion_mode const assertion, ::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
  { ::yampi::yampi_fence_detail::fence(static_cast<int>(assertion), window, environment); }
}


#endif

