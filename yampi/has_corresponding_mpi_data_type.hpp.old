#ifndef YAMPI_HAS_CORRESPONDING_MPI_DATA_TYPE_HPP
# define YAMPI_HAS_CORRESPONDING_MPI_DATA_TYPE_HPP

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
  struct has_corresponding_mpi_data_type
    : public YAMPI_false_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<char>
    : public YAMPI_true_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<signed short int>
    : public YAMPI_true_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<signed int>
    : public YAMPI_true_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<signed long int>
    : public YAMPI_true_type
  { };

# ifndef BOOST_NO_LONG_LONG
  template <>
  struct has_corresponding_mpi_data_type<signed long long int>
    : public YAMPI_true_type
  { };
# endif

  template <>
  struct has_corresponding_mpi_data_type<signed char>
    : public YAMPI_true_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<unsigned char>
    : public YAMPI_true_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<unsigned short int>
    : public YAMPI_true_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<unsigned int>
    : public YAMPI_true_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<unsigned long int>
    : public YAMPI_true_type
  { };

# ifndef BOOST_NO_LONG_LONG
  template <>
  struct has_corresponding_mpi_data_type<unsigned long long int>
    : public YAMPI_true_type
  { };
# endif

  template <>
  struct has_corresponding_mpi_data_type<float>
    : public YAMPI_true_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<double>
    : public YAMPI_true_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<long double>
    : public YAMPI_true_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<wchar_t>
    : public YAMPI_true_type
  { };

# if defined(__FUJITSU) || MPI_VERSION >= 3
  template <>
  struct has_corresponding_mpi_data_type<bool>
    : public YAMPI_true_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<std::complex<float> >
    : public YAMPI_true_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<std::complex<double> >
    : public YAMPI_true_type
  { };

  template <>
  struct has_corresponding_mpi_data_type<std::complex<long double> >
    : public YAMPI_true_type
  { };
# endif // defined(__FUJITSU) || MPI_VERSION >= 3
}


# undef YAMPI_true_type
# undef YAMPI_false_type

#endif

