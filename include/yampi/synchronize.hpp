#ifndef YAMPI_SYNCHRONIZE_HPP
# define YAMPI_SYNCHRONIZE_HPP

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


# if MPI_VERSION >= 3
namespace yampi
{
  template <typename Derived>
  inline void synchronize(::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
  {
    int const error_code = MPI_Win_synchronize(window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::synchronize", environment);
  }
}
# endif // MPI_VERSION >= 3


#endif

