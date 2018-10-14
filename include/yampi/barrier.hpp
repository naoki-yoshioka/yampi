#ifndef YAMPI_BARRIER_HPP
# define YAMPI_BARRIER_HPP

# include <boost/config.hpp>

# include <mpi.h>

# if MPI_VERSION >= 3
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     include <memory>
#   else
#     include <boost/core/addressof.hpp>
#   endif
# endif

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# if MPI_VERSION >= 3
#   include <yampi/request.hpp>
# endif

# if MPI_VERSION >= 3
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     define YAMPI_addressof std::addressof
#   else
#     define YAMPI_addressof boost::addressof
#   endif
# endif


namespace yampi
{
  inline void barrier(::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    int const error_code = MPI_Barrier(communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::barrier", environment);
  }
# if MPI_VERSION >= 3

  inline void barrier(
    ::yampi::request& request, ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    int const error_code
      = MPI_Ibarrier(communicator.mpi_comm(), YAMPI_addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::barrier", environment);

    request.reset(mpi_request, environment);
  }
# endif
}


# if MPI_VERSION >= 3
#   undef YAMPI_addressof
# endif

#endif

