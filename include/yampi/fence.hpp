#ifndef YAMPI_FENCE_HPP
# define YAMPI_FENCE_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/mode.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
#   define YAMPI_MODE ::yampi::mode
# else // BOOST_NO_CXX11_SCOPED_ENUMS
#   define YAMPI_MODE ::yampi::mode::mode_
# endif // BOOST_NO_CXX11_SCOPED_ENUMS


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
  inline void fence(YAMPI_MODE const assertion, ::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
  { ::yampi::yampi_fence_detail::fence(static_cast<int>(assertion), window, environment); }
}


# undef YAMPI_MODE

#endif

