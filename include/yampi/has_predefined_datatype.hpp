#ifndef YAMPI_HAS_PREDEFINED_DATATYPE_HPP
# define YAMPI_HAS_PREDEFINED_DATATYPE_HPP

# include <boost/config.hpp>

# if MPI_VERSION >= 2
#   include <complex>
# endif
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
  namespace has_predefined_datatype_detail
  {
    // These typedef's are required because the original types std::pair<foo, int>'s have commas
    typedef std::pair<short, int> short_int_type;
    typedef std::pair<int, int> int_int_type;
    typedef std::pair<long, int> long_int_type;
    typedef std::pair<float, int> float_int_type;
    typedef std::pair<double, int> double_int_type;
    typedef std::pair<long double, int> long_double_int_type;
  } // namespace has_predefined_datatype_detail

  template <typename T>
  struct has_predefined_datatype : YAMPI_false_type { };

# define YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(type) \
  template <>\
  struct has_predefined_datatype< type > : YAMPI_true_type { };


  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(char)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(signed short int)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(signed int)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(signed long int)
# ifndef BOOST_NO_LONG_LONG
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(signed long long int)
# endif
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(signed char)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(unsigned char)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(unsigned short int)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(unsigned int)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(unsigned long int)
# ifndef BOOST_NO_LONG_LONG
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(unsigned long long int)
# endif
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(float)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(double)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(long double)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(wchar_t)

# if MPI_VERSION >= 2
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(bool)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(std::complex<float>)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(std::complex<double>)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(std::complex<long double>)
# endif

  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(has_predefined_datatype_detail::short_int_type)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(has_predefined_datatype_detail::int_int_type)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(has_predefined_datatype_detail::long_int_type)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(has_predefined_datatype_detail::float_int_type)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(has_predefined_datatype_detail::double_int_type)
  YAMPI_MAKE_HAS_PREDEFINED_DATATYPE(has_predefined_datatype_detail::long_double_int_type)

# undef YAMPI_MAKE_HAS_PREDEFINED_DATATYPE
}


# undef YAMPI_false_type
# undef YAMPI_true_type

#endif // YAMPI_HAS_PREDEFINED_DATATYPE_HPP
