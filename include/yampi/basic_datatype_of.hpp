#ifndef YAMPI_BASIC_DATATYPE_OF_HPP
# define YAMPI_BASIC_DATATYPE_OF_HPP

# include <boost/config.hpp>

# include <utility>

# include <mpi.h>

# if MPI_VERSION >= 2
#   include <complex>
# endif

# include <yampi/datatype.hpp>


namespace yampi
{
  template <typename T>
  struct basic_datatype_of;

# define YAMPI_MAKE_BASIC_DATATYPE_OF(type, mpi_datatype) \
  template <>\
  struct basic_datatype_of< type >\
  {\
    static ::yampi::datatype call()\
    { return ::yampi::datatype( mpi_datatype ); }\
  };


  YAMPI_MAKE_BASIC_DATATYPE_OF(char, MPI_CHAR)
  YAMPI_MAKE_BASIC_DATATYPE_OF(signed short int, MPI_SHORT)
  YAMPI_MAKE_BASIC_DATATYPE_OF(signed int, MPI_INT)
  YAMPI_MAKE_BASIC_DATATYPE_OF(signed long int, MPI_LONG)
# ifndef BOOST_NO_LONG_LONG
  YAMPI_MAKE_BASIC_DATATYPE_OF(signed long long int, MPI_LONG_LONG)
# endif
  YAMPI_MAKE_BASIC_DATATYPE_OF(signed char, MPI_SIGNED_CHAR)
  YAMPI_MAKE_BASIC_DATATYPE_OF(unsigned char, MPI_UNSIGNED_CHAR)
  YAMPI_MAKE_BASIC_DATATYPE_OF(unsigned short int, MPI_UNSIGNED_SHORT)
  YAMPI_MAKE_BASIC_DATATYPE_OF(unsigned int, MPI_UNSIGNED)
  YAMPI_MAKE_BASIC_DATATYPE_OF(unsigned long int, MPI_UNSIGNED_LONG)
# ifndef BOOST_NO_LONG_LONG
  YAMPI_MAKE_BASIC_DATATYPE_OF(unsigned long long int, MPI_UNSIGNED_LONG_LONG)
# endif
  YAMPI_MAKE_BASIC_DATATYPE_OF(float, MPI_FLOAT)
  YAMPI_MAKE_BASIC_DATATYPE_OF(double, MPI_DOUBLE)
  YAMPI_MAKE_BASIC_DATATYPE_OF(long double, MPI_LONG_DOUBLE)
  YAMPI_MAKE_BASIC_DATATYPE_OF(wchar_t, MPI_WCHAR)

# if MPI_VERSION >= 3 || defined(__FUJITSU)
  YAMPI_MAKE_BASIC_DATATYPE_OF(bool, MPI_CXX_BOOL)
  YAMPI_MAKE_BASIC_DATATYPE_OF(std::complex<float>, MPI_CXX_FLOAT_COMPLEX)
  YAMPI_MAKE_BASIC_DATATYPE_OF(std::complex<double>, MPI_CXX_DOUBLE_COMPLEX)
  YAMPI_MAKE_BASIC_DATATYPE_OF(std::complex<long double>, MPI_CXX_LONG_DOUBLE_COMPLEX)
# elif MPI_VERSION >= 2
  YAMPI_MAKE_BASIC_DATATYPE_OF(bool, MPI::BOOL)
  YAMPI_MAKE_BASIC_DATATYPE_OF(std::complex<float>, MPI::COMPLEX)
  YAMPI_MAKE_BASIC_DATATYPE_OF(std::complex<double>, MPI::DOUBLE_COMPLEX)
  YAMPI_MAKE_BASIC_DATATYPE_OF(std::complex<long double>, MPI::LONG_DOUBLE_COMPLEX)
# endif

  YAMPI_MAKE_BASIC_DATATYPE_OF((std::pair<short, int>), MPI_SHORT_INT)
  YAMPI_MAKE_BASIC_DATATYPE_OF((std::pair<int, int>), MPI_2INT)
  YAMPI_MAKE_BASIC_DATATYPE_OF((std::pair<long, int>), MPI_LONG_INT)
  YAMPI_MAKE_BASIC_DATATYPE_OF((std::pair<float, int>), MPI_FLOAT_INT)
  YAMPI_MAKE_BASIC_DATATYPE_OF((std::pair<double, int>), MPI_DOUBLE_INT)
  YAMPI_MAKE_BASIC_DATATYPE_OF((std::pair<long double, int)>, MPI_LONG_DOUBLE_INT)

# undef YAMPI_MAKE_BASIC_DATATYPE_OF
}


#endif
