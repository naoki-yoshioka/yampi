#ifndef YAMPI_WAIT_HPP
# define YAMPI_WAIT_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/request.hpp>
# include <yampi/status.hpp>
# include <yampi/eror.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  inline ::yampi::status wait(::yampi::request const& request)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto stat = MPI_Status{};
#   else
    auto stat = MPI_Status();
#   endif

    auto const error_code = MPI_Wait(YAMPI_addressof(request.mpi_request()), YAMPI_addressof(stat));
# else
    MPI_Status stat;

    int const error_code = MPI_Wait(YAMPI_addressof(request.mpi_request()), YAMPI_addressof(stat));
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::wait"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::wait");
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    return ::yampi::status{stat};
# else
    return ::yampi::status(stat);
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

    auto const error_code = MPI_Wait(YAMPI_addressof(request.mpi_request()), MPI_STATUS_IGNORE);
# else
    MPI_Status stat;

    int const error_code = MPI_Wait(YAMPI_addressof(request.mpi_request()), MPI_STATUS_IGNORE);
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


# undef YAMPI_addressof

#endif

