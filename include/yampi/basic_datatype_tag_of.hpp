#ifndef YAMPI_BASIC_DATATYPE_TAG_OF_HPP
# define YAMPI_BASIC_DATATYPE_TAG_OF_HPP

# include <boost/config.hpp>

# include <utility>

# include <mpi.h>

# if MPI_VERSION >= 2
#   include <complex>
# endif

# include <yampi/datatype.hpp>


namespace yampi
{
  namespace basic_datatype_tag_of_detail
  {
    typedef std::pair<short, int> short_int_type;
    typedef std::pair<int, int> int_int_type;
    typedef std::pair<long, int> long_int_type;
    typedef std::pair<float, int> float_int_type;
    typedef std::pair<double, int> double_int_type;
    typedef std::pair<long double, int> long_double_int_type;
  }

  template <typename T>
  struct basic_datatype_tag_of;

# define YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(type, basic_datatype_name) \
  template <>\
  struct basic_datatype_tag_of< type >\
  {\
    static basic_datatype_name ## _datatype_t call()\
    { return basic_datatype_name ## _datatype_t(); }\
  };


  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(char, char)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(signed short int, short)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(signed int, int)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(signed long int, long)
# ifndef BOOST_NO_LONG_LONG
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(signed long long int, long_long)
# endif
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(signed char, signed_char)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(unsigned char, unsigned_char)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(unsigned short int, unsigned_short)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(unsigned int, unsigned)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(unsigned long int, unsigned_long)
# ifndef BOOST_NO_LONG_LONG
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(unsigned long long int, unsigned_long_long)
# endif
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(float, float)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(double, double)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(long double, long_double)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(wchar_t, wchar)

# if MPI_VERSION >= 2
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(bool, bool)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(std::complex<float>, float_complex)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(std::complex<double>, double_complex)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(std::complex<long double>, long_double_complex)
# endif

  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(::yampi::basic_datatype_tag_of_detail::short_int_type, short_int)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(::yampi::basic_datatype_tag_of_detail::int_int_type, int_int)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(::yampi::basic_datatype_tag_of_detail::long_int_type, long_int)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(::yampi::basic_datatype_tag_of_detail::float_int_type, float_int)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(::yampi::basic_datatype_tag_of_detail::double_int_type, double_int)
  YAMPI_MAKE_BASIC_DATATYPE_TAG_OF(::yampi::basic_datatype_tag_of_detail::long_double_int_type, long_double_int)

# undef YAMPI_MAKE_BASIC_DATATYPE_TAG_OF
}


#endif
