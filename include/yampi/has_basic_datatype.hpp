#ifndef YAMPI_HAS_BASIC_DATATYPE_HPP
# define YAMPI_HAS_BASIC_DATATYPE_HPP

# include <boost/config.hpp>

# if MPI_VERSION >= 2
#   include <complex>
# endif
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/integral_constant.hpp>
# endif

# include <yampi/datatype.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_true_type std::true_type
#   define YAMPI_false_type std::false_type
# else
#   define YAMPI_true_type boost::true_type
#   define YAMPI_false_type boost::false_type
# endif


namespace yampi
{
  template <typename T>
  struct has_basic_datatype : YAMPI_false_type { };

# define YAMPI_MAKE_HAS_BASIC_DATATYPE(type) \
  template <>\
  struct has_basic_datatype< type > : YAMPI_true_type { };


  YAMPI_MAKE_HAS_BASIC_DATATYPE(char)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(signed short int)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(signed int)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(signed long int)
# ifndef BOOST_NO_LONG_LONG
  YAMPI_MAKE_HAS_BASIC_DATATYPE(signed long long int)
# endif
  YAMPI_MAKE_HAS_BASIC_DATATYPE(signed char)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(unsigned char)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(unsigned short int)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(unsigned int)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(unsigned long int)
# ifndef BOOST_NO_LONG_LONG
  YAMPI_MAKE_HAS_BASIC_DATATYPE(unsigned long long int)
# endif
  YAMPI_MAKE_HAS_BASIC_DATATYPE(float)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(double)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(long double)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(wchar_t)

# if MPI_VERSION >= 3 || defined(__FUJITSU)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(bool)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(std::complex<float>)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(std::complex<double>)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(std::complex<long double>)
# elif MPI_VERSION >= 2
  YAMPI_MAKE_HAS_BASIC_DATATYPE(bool)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(std::complex<float>)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(std::complex<double>)
  YAMPI_MAKE_HAS_BASIC_DATATYPE(std::complex<long double>)
# endif

# undef YAMPI_MAKE_HAS_BASIC_DATATYPE
}


# undef YAMPI_false_type
# undef YAMPI_true_type

#endif
