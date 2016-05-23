#ifndef YAMPI_MPI_DATA_TYPE_OF_HPP
# define YAMPI_MPI_DATA_TYPE_OF_HPP

# include <boost/config.hpp>

# include <cstddef>
# ifdef BOOST_HAS_STDINT_H
#   include <cstdint>
# endif
# include <complex>

# include <boost/cstdint.hpp>

# include <mpi.h>


namespace yampi
{
  template <typename T>
  struct mpi_data_type_of;

  template <>
  struct mpi_data_type_of<char>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_CHAR; };

  template <>
  struct mpi_data_type_of<signed short int>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_SHORT; };

  template <>
  struct mpi_data_type_of<signed int>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_INT; };

  template <>
  struct mpi_data_type_of<signed long int>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_LONG; };

# ifndef BOOST_NO_LONG_LONG
  template <>
  struct mpi_data_type_of<signed long long int>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_LONG_LONG; };
# endif

  template <>
  struct mpi_data_type_of<signed char>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_SIGNED_CHAR; };

  template <>
  struct mpi_data_type_of<unsigned char>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_UNSIGNED_CHAR; };

  template <>
  struct mpi_data_type_of<unsigned short int>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_UNSIGNED_SHORT; };

  template <>
  struct mpi_data_type_of<unsigned int>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_UNSIGNED; };

  template <>
  struct mpi_data_type_of<unsigned long int>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_UNSIGNED_LONG; };

# ifndef BOOST_NO_LONG_LONG
  template <>
  struct mpi_data_type_of<unsigned long long int>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_UNSIGNED_LONG_LONG; };
# endif

  template <>
  struct mpi_data_type_of<float>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_FLOAT; };

  template <>
  struct mpi_data_type_of<double>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_DOUBLE; };

  template <>
  struct mpi_data_type_of<long double>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_LONG_DOUBLE; };

  template <>
  struct mpi_data_type_of<wchar_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_WCHAR; };

/*
# ifdef BOOST_HAS_STDINT_H
  template <>
  struct mpi_data_type_of<std::int8_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_INT8_T; };

  template <>
  struct mpi_data_type_of<std::int16_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_INT16_T; };

  template <>
  struct mpi_data_type_of<std::int32_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_INT32_T; };

  template <>
  struct mpi_data_type_of<std::int64_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_INT64_T; };

  template <>
  struct mpi_data_type_of<std::uint8_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_UINT8_T; };

  template <>
  struct mpi_data_type_of<std::uint16_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_UINT16_T; };

  template <>
  struct mpi_data_type_of<std::uint32_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_UINT32_T; };

  template <>
  struct mpi_data_type_of<std::uint64_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_UINT64_T; };
# endif

  template <>
  struct mpi_data_type_of<boost::int8_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_INT8_T; };

  template <>
  struct mpi_data_type_of<boost::int16_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_INT16_T; };

  template <>
  struct mpi_data_type_of<boost::int32_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_INT32_T; };

  template <>
  struct mpi_data_type_of<boost::int64_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_INT64_T; };

  template <>
  struct mpi_data_type_of<boost::uint8_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_UINT8_T; };

  template <>
  struct mpi_data_type_of<boost::uint16_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_UINT16_T; };

  template <>
  struct mpi_data_type_of<boost::uint32_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_UINT32_T; };

  template <>
  struct mpi_data_type_of<boost::uint64_t>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_UINT64_T; };
*/

  template <>
  struct mpi_data_type_of<bool>
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_CXX_BOOL; };

  template <>
  struct mpi_data_type_of<std::complex<float> >
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_CXX_FLOAT_COMPLEX; };

  template <>
  struct mpi_data_type_of<std::complex<double> >
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_CXX_DOUBLE_COMPLEX; };

  template <>
  struct mpi_data_type_of<std::complex<long double> >
  { BOOST_STATIC_CONSTEXPR MPI_Datatype value = MPI_CXX_LONG_DOUBLE_COMPLEX; };
}


#endif

