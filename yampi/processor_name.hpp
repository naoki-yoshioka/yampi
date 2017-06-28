#ifndef YAMPI_PROCESSOR_NAME_HPP
# define YAMPI_PROCESSOR_NAME_HPP

# include <boost/config.hpp>

# include <string>

# include <mpi.h>

# include <yampi/error.hpp>


namespace yampi
{
  std::string processor_name()
  {
    char name[MPI_MAX_PROCESSOR_NAME];
    int length;
    int const error_code = MPI_Get_processor_name(name, &length);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::environment::processor_name");

    return name;
  }
}


#endif

