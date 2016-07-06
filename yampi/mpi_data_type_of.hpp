#ifndef YAMPI_MPI_DATA_TYPE_OF_HPP
# define YAMPI_MPI_DATA_TYPE_OF_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <complex>

# include <mpi.h>


namespace yampi
{
# if !defined(__FUJITSU) && !defined(BOOST_NO_CXX11_CONSTEXPR)
  template <typename T>
  struct mpi_data_type_of;

  template <>
  struct mpi_data_type_of<char>
  { static constexpr MPI_Datatype value = MPI_CHAR; };

  template <>
  struct mpi_data_type_of<signed short int>
  { static constexpr MPI_Datatype value = MPI_SHORT; };

  template <>
  struct mpi_data_type_of<signed int>
  { static constexpr MPI_Datatype value = MPI_INT; };

  template <>
  struct mpi_data_type_of<signed long int>
  { static constexpr MPI_Datatype value = MPI_LONG; };

#   ifndef BOOST_NO_LONG_LONG
  template <>
  struct mpi_data_type_of<signed long long int>
  { static constexpr MPI_Datatype value = MPI_LONG_LONG; };
#   endif

  template <>
  struct mpi_data_type_of<signed char>
  { static constexpr MPI_Datatype value = MPI_SIGNED_CHAR; };

  template <>
  struct mpi_data_type_of<unsigned char>
  { static constexpr MPI_Datatype value = MPI_UNSIGNED_CHAR; };

  template <>
  struct mpi_data_type_of<unsigned short int>
  { static constexpr MPI_Datatype value = MPI_UNSIGNED_SHORT; };

  template <>
  struct mpi_data_type_of<unsigned int>
  { static constexpr MPI_Datatype value = MPI_UNSIGNED; };

  template <>
  struct mpi_data_type_of<unsigned long int>
  { static constexpr MPI_Datatype value = MPI_UNSIGNED_LONG; };

#   ifndef BOOST_NO_LONG_LONG
  template <>
  struct mpi_data_type_of<unsigned long long int>
  { static constexpr MPI_Datatype value = MPI_UNSIGNED_LONG_LONG; };
#   endif

  template <>
  struct mpi_data_type_of<float>
  { static constexpr MPI_Datatype value = MPI_FLOAT; };

  template <>
  struct mpi_data_type_of<double>
  { static constexpr MPI_Datatype value = MPI_DOUBLE; };

  template <>
  struct mpi_data_type_of<long double>
  { static constexpr MPI_Datatype value = MPI_LONG_DOUBLE; };

  template <>
  struct mpi_data_type_of<wchar_t>
  { static constexpr MPI_Datatype value = MPI_WCHAR; };

#   if MPI_VERSION >= 3
  template <>
  struct mpi_data_type_of<bool>
  { static constexpr MPI_Datatype value = MPI_CXX_BOOL; };

  template <>
  struct mpi_data_type_of<std::complex<float> >
  { static constexpr MPI_Datatype value = MPI_CXX_FLOAT_COMPLEX; };

  template <>
  struct mpi_data_type_of<std::complex<double> >
  { static constexpr MPI_Datatype value = MPI_CXX_DOUBLE_COMPLEX; };

  template <>
  struct mpi_data_type_of<std::complex<long double> >
  { static constexpr MPI_Datatype value = MPI_CXX_LONG_DOUBLE_COMPLEX; };
#   endif // MPI_VERSION >= 3
# else // !defined(__FUJITSU) && !defined(BOOST_NO_CXX11_CONSTEXPR)
  template <typename T>
  struct mpi_data_type_of
  { static MPI_Datatype const value; };

  template <>
  MPI_Datatype const mpi_data_type_of<char>::value = MPI_CHAR;

  template <>
  MPI_Datatype const mpi_data_type_of<signed short int>::value = MPI_SHORT;

  template <>
  MPI_Datatype const mpi_data_type_of<signed int>::value = MPI_INT;

  template <>
  MPI_Datatype const mpi_data_type_of<signed long int>::value = MPI_LONG;

#   ifndef BOOST_NO_LONG_LONG
  template <>
  MPI_Datatype const mpi_data_type_of<signed long long int>::value = MPI_LONG_LONG;
#   endif

  template <>
  MPI_Datatype const mpi_data_type_of<signed char>::value = MPI_SIGNED_CHAR;

  template <>
  MPI_Datatype const mpi_data_type_of<unsigned char>::value = MPI_UNSIGNED_CHAR;

  template <>
  MPI_Datatype const mpi_data_type_of<unsigned short int>::value = MPI_UNSIGNED_SHORT;

  template <>
  MPI_Datatype const mpi_data_type_of<unsigned int>::value = MPI_UNSIGNED;

  template <>
  MPI_Datatype const mpi_data_type_of<unsigned long int>::value = MPI_UNSIGNED_LONG;

#   ifndef BOOST_NO_LONG_LONG
  template <>
  MPI_Datatype const mpi_data_type_of<unsigned long long int>::value = MPI_UNSIGNED_LONG_LONG;
#   endif

  template <>
  MPI_Datatype const mpi_data_type_of<float>::value = MPI_FLOAT;

  template <>
  MPI_Datatype const mpi_data_type_of<double>::value = MPI_DOUBLE;

  template <>
  MPI_Datatype const mpi_data_type_of<long double>::value = MPI_LONG_DOUBLE;

  template <>
  MPI_Datatype const mpi_data_type_of<wchar_t>::value = MPI_WCHAR;

#   if defined(__FUJITSU) || MPI_VERSION >= 3
  template <>
  MPI_Datatype const mpi_data_type_of<bool>::value = MPI_CXX_BOOL;

  template <>
  MPI_Datatype const mpi_data_type_of<std::complex<float> >::value = MPI_CXX_FLOAT_COMPLEX;

  template <>
  MPI_Datatype const mpi_data_type_of<std::complex<double> >::value = MPI_CXX_DOUBLE_COMPLEX;

  template <>
  MPI_Datatype const mpi_data_type_of<std::complex<long double> >::value = MPI_CXX_LONG_DOUBLE_COMPLEX;
#   endif
#endif // BOOST_NO_CXX11_CONSTEXPR
}


#endif

