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
    int length;
    int const error_code = MPI_Get_library_version(version, &length);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::library_version");
    return version;
  }
}


#endif

