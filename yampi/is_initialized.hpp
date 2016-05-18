#ifndef YAMPI_IS_INITIALIZED_HPP
# define YAMPI_IS_INITIALIZED_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/error.hpp>


namespace yampi
{
  inline bool is_initialized()
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto result = int{};
#   else
      auto result = int();
#   endif

    auto const error_code = MPI_Initialized(&result);
# else
    int result;

    int const error_code = MPI_Initialized(&result);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::is_initialized"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::is_initialized");
# endif

    return result;
  }
}


#endif

