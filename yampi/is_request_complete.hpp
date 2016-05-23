#ifndef YAMPI_IS_REQUEST_COMPLETE_HPP
# define YAMPI_IS_REQUEST_COMPLETE_HPP

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
  template <typename Value>
  inline std::pair<bool, ::yampi::status> is_request_complete(::yampi::request const& request)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto flag = int{};
      auto stat = MPI_Status{};
#   else
      auto flag = int();
      auto stat = MPI_Status();
#   endif

    auto const error_code = MPI_Test(YAMPI_addressof(request.mpi_request()), &flag, YAMPI_addressof(stat));
# else
    int flag;
    MPI_Status stat;

    int const error_code = MPI_Test(YAMPI_addressof(request.mpi_request()), &flag, YAMPI_addressof(stat));
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::is_request_complete"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::is_request_complete");
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    return std::make_pair(static_cast<bool>(flag), ::yampi::status{stat});
# else
    return std::make_pair(static_cast<bool>(flag), ::yampi::status(stat));
# endif
  }

  inline bool is_request_complete(::yampi::request const& request, ::yampi::ignore_status_t const)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto flag = int{};
#   else
      auto flag = int();
#   endif
# else
    int flag;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
    auto const error_code = MPI_Test(YAMPI_addressof(request.mpi_request()), &flag, MPI_STATUS_IGNORE);
# else
    int const error_code = MPI_Test(YAMPI_addressof(request.mpi_request()), &flag, MPI_STATUS_IGNORE);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::is_request_complete"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::is_request_complete");
# endif

    return static_cast<bool>(flag);
  }
}


# undef YAMPI_addressof

#endif

