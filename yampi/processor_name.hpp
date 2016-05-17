#ifndef YAMPI_PROCESSOR_NAME_HPP
# define YAMPI_PROCESSOR_NAME_HPP

# include <boost/config.hpp>

# include <string>

# include <mpi.h>

# include <yampi/error.hpp>


namespace yampi
{
  class environment;

  inline std::string processor_name(::yampi::environment&)
  {
    char name[MPI_MAX_PROCESSOR_NAME];
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto length = int{};
#   else
      auto length = int();
#   endif

    auto const error_code = MPI_Get_processor_name(name, &length);
# else
    int length;

    int const error_code = MPI_Get_processor_name(name, &length);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::processor_name"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::processor_name");
# endif

    return name;
  }
}


#endif

