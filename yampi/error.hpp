#ifndef YAMPI_ERROR_HPP
# define YAMPI_ERROR_HPP

# include <boost/config.hpp>

# include <string>
# include <stdexcept>

# include <mpi.h>


namespace yampi
{
  class error_in_error
    : public std::runtime_error
  {
   public:
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    explicit error_in_error(std::string const& where)
      : std::runtime_error{std::string{"In "} + where + ": error occured in MPI_Error_class/string"}
    { }
# else
    explicit error_in_error(std::string const& where)
      : std::runtime_error(std::string("In ") + where + ": error occured in MPI_Error_class/string")
    { }
# endif
  };

  class error
    : public std::runtime_error
  {
    int error_class_;

   public:
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    error(int const error_code, std::string const& where)
     : std::runtime_error{generate_what_string(error_code, where).c_str()}, error_class_{generate_error_class(error_code, where)}
    { }
# else
    error(int const error_code, std::string const& where)
     : std::runtime_error(generate_what_string(error_code, where).c_str()), error_class_(generate_error_class(error_code, where))
    { }
# endif

    int error_class() const { return error_class_; }

   private:
    int generate_error_class(int const error_code, std::string const& where)
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto error_class = int{};
#   else
      auto error_class = int();
#   endif

      auto const error_class_error_code = MPI_Error_class(error_code, &error_class);
# else
      int error_class;

      int const error_class_error_code = MPI_Error_class(error_code, &error_class);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_class_error_code != MPI_SUCCESS)
        throw ::yampi::error_in_error{where};
# else
      if (error_class_error_code != MPI_SUCCESS)
        throw ::yampi::error_in_error(where);
# endif

      return error_class;
    }

    std::string generate_what_string(int const error_code, std::string const& where)
    {
      char error[MPI_MAX_ERROR_STRING];
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto length = int{};
#   else
      auto length = int();
#   endif

      auto const error_string_error_code = MPI_Error_string(error_code, error, &length);
# else
      int length;

      int const error_string_error_code = MPI_Error_string(error_code, error, &length);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_string_error_code != MPI_SUCCESS)
        throw ::yampi::error_in_error{where};
# else
      if (error_string_error_code != MPI_SUCCESS)
        throw ::yampi::error_in_error(where);
# endif

      return std::string{"In "} + where + ": " + std::string{error};
    }
  };
}


#endif

