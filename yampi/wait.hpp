#ifndef YAMPI_WAIT_HPP
# define YAMPI_WAIT_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/request.hpp>
# include <yampi/status.hpp>
# include <yampi/eror.hpp>


namespace yampi
{
  template <typename Value>
  inline ::yampi::status<Value> wait(::yampi::request const& request)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto stat = MPI_Status{};
#   else
    auto stat = MPI_Status();
#   endif

    auto const error_code = MPI_Wait(&request.mpi_request(), &stat);
# else
    MPI_Status stat;

    int const error_code = MPI_Wait(&request.mpi_request(), &stat);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::wait"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::wait");
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    return ::yampi::status<Value>{stat};
# else
    return ::yampi::status<Value>(stat);
# endif
  }

  inline void wait(::yampi::request const& request, ::yampi::ignore_status_t const)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto stat = MPI_Status{};
#   else
    auto stat = MPI_Status();
#   endif

    auto const error_code = MPI_Wait(&request.mpi_request(), MPI_STATUS_IGNORE);
# else
    MPI_Status stat;

    int const error_code = MPI_Wait(&request.mpi_request(), MPI_STATUS_IGNORE);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::wait"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::wait");
# endif
  }
}


#endif

