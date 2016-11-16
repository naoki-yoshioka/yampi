#ifndef YAMPI_HAS_CORRESPONDING_DATATYPE_HPP
# define YAMPI_HAS_CORRESPONDING_DATATYPE_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <complex>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/integral_constant.hpp>
# endif

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
  struct has_corresponding_datatype
    : public YAMPI_false_type
  { };


# define YAMPI_HAS_CORRESPONDING_DATATYPE(type) \
  template <>\
  struct has_corresponding_datatype< type >\
    : public YAMPI_true_type\
  { };\


  YAMPI_HAS_CORRESPONDING_DATATYPE(char)
  YAMPI_HAS_CORRESPONDING_DATATYPE(signed short int)
  YAMPI_HAS_CORRESPONDING_DATATYPE(signed int)
  YAMPI_HAS_CORRESPONDING_DATATYPE(signed long int)
# ifndef BOOST_NO_LONG_LONG
  YAMPI_HAS_CORRESPONDING_DATATYPE(signed long long int)
# endif
  YAMPI_HAS_CORRESPONDING_DATATYPE(signed char)
  YAMPI_HAS_CORRESPONDING_DATATYPE(unsigned char)
  YAMPI_HAS_CORRESPONDING_DATATYPE(unsigned short int)
  YAMPI_HAS_CORRESPONDING_DATATYPE(unsigned int)
  YAMPI_HAS_CORRESPONDING_DATATYPE(unsigned long int)
# ifndef BOOST_NO_LONG_LONG
  YAMPI_HAS_CORRESPONDING_DATATYPE(unsigned long long int)
# endif
  YAMPI_HAS_CORRESPONDING_DATATYPE(float)
  YAMPI_HAS_CORRESPONDING_DATATYPE(double)
  YAMPI_HAS_CORRESPONDING_DATATYPE(long double)
  YAMPI_HAS_CORRESPONDING_DATATYPE(wchar_t)

# if defined(__FUJITSU) || MPI_VERSION >= 2
  YAMPI_HAS_CORRESPONDING_DATATYPE(bool)
  YAMPI_HAS_CORRESPONDING_DATATYPE(std::complex<float>)
  YAMPI_HAS_CORRESPONDING_DATATYPE(std::complex<double>)
  YAMPI_HAS_CORRESPONDING_DATATYPE(std::complex<long double>)
# endif

# undef YAMPI_HAS_CORRESPONDING_DATATYPE
}


# undef YAMPI_true_type
# undef YAMPI_false_type

#endif

