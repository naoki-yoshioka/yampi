#ifndef YAMPI_IS_REQUEST_COMPLETE_HPP
# define YAMPI_IS_REQUEST_COMPLETE_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/request.hpp>
# include <yampi/status.hpp>
# include <yampi/eror.hpp>


namespace yampi
{
  template <typename Value>
  inline std::pair<bool, ::yampi::status<Value> > is_request_complete(::yampi::request const& request)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto flag = int{};
      auto stat = MPI_Status{};
#   else
      auto flag = int();
      auto stat = MPI_Status();
#   endif

    auto const error_code = MPI_Test(&request.mpi_request(), &flag, &stat);
# else
    int flag;
    MPI_Status stat;

    int const error_code = MPI_Test(&request.mpi_request(), &flag, &stat);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::is_request_complete"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::is_request_complete");
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    return std::make_pair(static_cast<bool>(flag), ::yampi::status<Value>{stat});
# else
    return std::make_pair(static_cast<bool>(flag), ::yampi::status<Value>(stat));
# endif
  }

  inline bool is_request_complete(::yampi::request const& request, ::yampi::ignore_status_t const)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto flag = int{};
      auto stat = MPI_Status{};
#   else
      auto flag = int();
      auto stat = MPI_Status();
#   endif

    auto const error_code = MPI_Test(&request.mpi_request(), &flag, MPI_STATUS_IGNORE);
# else
    int flag;
    MPI_Status stat;

    int const error_code = MPI_Test(&request.mpi_request(), &flag, MPI_STATUS_IGNORE);
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


#endif

