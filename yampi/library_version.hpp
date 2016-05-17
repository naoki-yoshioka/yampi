#ifndef YAMPI_LIBRARY_VERSION_HPP
# define YAMPI_LIBRARY_VERSION_HPP

# include <boost/config.hpp>

# include <string>

# include <mpi.h>

# include <yampi/error.hpp>


namespace yampi
{
  inline std::string library_version() const
  {
    char version[MPI_MAX_LIBRARY_VERSION_STRING];
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto length = int{};
#   else
      auto length = int();
#   endif

    auto const error_code = MPI_Get_library_version(version, &length);
# else
    int length;

    int const error_code = MPI_Get_library_version(version, &length);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::library_version"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::library_version");
# endif

    return version;
  }
}


#endif

