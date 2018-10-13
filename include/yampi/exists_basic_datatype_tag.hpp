#ifndef YAMPI_EXISTS_BASIC_DATATYPE_TAG_HPP
# define YAMPI_EXISTS_BASIC_DATATYPE_TAG_HPP

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
  namespace exists_basic_datatype_tag_detail
  {
    typedef std::pair<short, int> short_int_type;
    typedef std::pair<int, int> int_int_type;
    typedef std::pair<long, int> long_int_type;
    typedef std::pair<float, int> float_int_type;
    typedef std::pair<double, int> double_int_type;
    typedef std::pair<long double, int> long_double_int_type;
  }

  template <typename T>
  struct exists_basic_datatype_tag : YAMPI_false_type { };

# define YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(type) \
  template <>\
  struct exists_basic_datatype_tag< type > : YAMPI_true_type { };


  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(char)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(signed short int)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(signed int)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(signed long int)
# ifndef BOOST_NO_LONG_LONG
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(signed long long int)
# endif
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(signed char)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(unsigned char)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(unsigned short int)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(unsigned int)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(unsigned long int)
# ifndef BOOST_NO_LONG_LONG
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(unsigned long long int)
# endif
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(float)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(double)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(long double)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(wchar_t)

# if MPI_VERSION >= 3 || defined(__FUJITSU)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(bool)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(std::complex<float>)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(std::complex<double>)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(std::complex<long double>)
# elif MPI_VERSION >= 2
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(bool)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(std::complex<float>)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(std::complex<double>)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(std::complex<long double>)
# endif

  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(::yampi::exists_basic_datatype_tag_detail::short_int_type)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(::yampi::exists_basic_datatype_tag_detail::int_int_type)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(::yampi::exists_basic_datatype_tag_detail::long_int_type)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(::yampi::exists_basic_datatype_tag_detail::float_int_type)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(::yampi::exists_basic_datatype_tag_detail::double_int_type)
  YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF(::yampi::exists_basic_datatype_tag_detail::long_double_int_type)

# undef YAMPI_MAKE_EXISTS_BASIC_DATATYPE_OF
}


# undef YAMPI_false_type
# undef YAMPI_true_type

#endif
