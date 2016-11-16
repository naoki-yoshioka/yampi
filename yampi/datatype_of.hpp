#ifndef YAMPI_DATATYPE_OF_HPP
# define YAMPI_DATATYPE_OF_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <complex>

# include <mpi.h>

# include <yampi/datatype.hpp>


namespace yampi
{
  template <typename T>
  class datatype_of
  {
    static ::yampi::datatype datatype_;

   public:
    static void set(::yampi::datatype const& datatype)
    { datatype_ = datatype; }

    static ::yampi::datatype const& call()
    { return datatype_; }
  };

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
  template <typename T>
  ::yampi::datatype datatype_of<T>::datatype_ = ::yampi::datatype{};
# else
  template <typename T>
  ::yampi::datatype datatype_of<T>::datatype_;
# endif


# define YAMPI_NEW_DATATYPE(type, datatype_) \
  template <>\
  class datatype_of< type >\
  {\
   public:\
    static ::yampi::datatype call()\
    { return datatype_; }\
  };\


# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
  YAMPI_NEW_DATATYPE(char, ::yampi::datatype{MPI_CHAR})
  YAMPI_NEW_DATATYPE(signed short int, ::yampi::datatype{MPI_SHORT})
  YAMPI_NEW_DATATYPE(signed int, ::yampi::datatype{MPI_INT})
  YAMPI_NEW_DATATYPE(signed long int, ::yampi::datatype{MPI_LONG})
#   ifndef BOOST_NO_LONG_LONG
  YAMPI_NEW_DATATYPE(signed long long int, ::yampi::datatype{MPI_LONG_LONG})
#   endif
  YAMPI_NEW_DATATYPE(signed char, ::yampi::datatype{MPI_SIGNED_CHAR})
  YAMPI_NEW_DATATYPE(unsigned char, ::yampi::datatype{MPI_UNSIGNED_CHAR})
  YAMPI_NEW_DATATYPE(unsigned short int, ::yampi::datatype{MPI_UNSIGNED_SHORT})
  YAMPI_NEW_DATATYPE(unsigned int, ::yampi::datatype{MPI_UNSIGNED})
  YAMPI_NEW_DATATYPE(unsigned long int, ::yampi::datatype{MPI_UNSIGNED_LONG})
#   ifndef BOOST_NO_LONG_LONG
  YAMPI_NEW_DATATYPE(unsigned long long int, ::yampi::datatype{MPI_UNSIGNED_LONG_LONG})
#   endif
  YAMPI_NEW_DATATYPE(float, ::yampi::datatype{MPI_FLOAT})
  YAMPI_NEW_DATATYPE(double, ::yampi::datatype{MPI_DOUBLE})
  YAMPI_NEW_DATATYPE(long double, ::yampi::datatype{MPI_LONG_DOUBLE})
  YAMPI_NEW_DATATYPE(wchar_t, ::yampi::datatype{MPI_WCHAR})

#   if defined(YAMPI_ENABLE_CXX_DATATYPES) || defined(__FUJITSU) || MPI_VERSION >= 3
  YAMPI_NEW_DATATYPE(bool, ::yampi::datatype{MPI_CXX_BOOL})
  YAMPI_NEW_DATATYPE(std::complex<float>, ::yampi::datatype{MPI_CXX_FLOAT_COMPLEX})
  YAMPI_NEW_DATATYPE(std::complex<double>, ::yampi::datatype{MPI_CXX_DOUBLE_COMPLEX})
  YAMPI_NEW_DATATYPE(std::complex<long double>, ::yampi::datatype{MPI_CXX_LONG_DOUBLE_COMPLEX})
#   endif
# else // BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
  YAMPI_NEW_DATATYPE(char, ::yampi::datatype(MPI_CHAR))
  YAMPI_NEW_DATATYPE(signed short int, ::yampi::datatype(MPI_SHORT))
  YAMPI_NEW_DATATYPE(signed int, ::yampi::datatype(MPI_INT))
  YAMPI_NEW_DATATYPE(signed long int, ::yampi::datatype(MPI_LONG))
#   ifndef BOOST_NO_LONG_LONG
  YAMPI_NEW_DATATYPE(signed long long int, ::yampi::datatype(MPI_LONG_LONG))
#   endif
  YAMPI_NEW_DATATYPE(signed char, ::yampi::datatype(MPI_SIGNED_CHAR))
  YAMPI_NEW_DATATYPE(unsigned char, ::yampi::datatype(MPI_UNSIGNED_CHAR))
  YAMPI_NEW_DATATYPE(unsigned short int, ::yampi::datatype(MPI_UNSIGNED_SHORT))
  YAMPI_NEW_DATATYPE(unsigned int, ::yampi::datatype(MPI_UNSIGNED))
  YAMPI_NEW_DATATYPE(unsigned long int, ::yampi::datatype(MPI_UNSIGNED_LONG))
#   ifndef BOOST_NO_LONG_LONG
  YAMPI_NEW_DATATYPE(unsigned long long int, ::yampi::datatype(MPI_UNSIGNED_LONG_LONG))
#   endif
  YAMPI_NEW_DATATYPE(float, ::yampi::datatype(MPI_FLOAT))
  YAMPI_NEW_DATATYPE(double, ::yampi::datatype(MPI_DOUBLE))
  YAMPI_NEW_DATATYPE(long double, ::yampi::datatype(MPI_LONG_DOUBLE))
  YAMPI_NEW_DATATYPE(wchar_t, ::yampi::datatype(MPI_WCHAR))

#   if defined(YAMPI_ENABLE_CXX_DATATYPES) || defined(__FUJITSU) || MPI_VERSION >= 3
  YAMPI_NEW_DATATYPE(bool, ::yampi::datatype(MPI_CXX_BOOL))
  YAMPI_NEW_DATATYPE(std::complex<float>, ::yampi::datatype(MPI_CXX_FLOAT_COMPLEX))
  YAMPI_NEW_DATATYPE(std::complex<double>, ::yampi::datatype(MPI_CXX_DOUBLE_COMPLEX))
  YAMPI_NEW_DATATYPE(std::complex<long double>, ::yampi::datatype(MPI_CXX_LONG_DOUBLE_COMPLEX))
#   endif
# endif // BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX

# undef YAMPI_NEW_DATATYPE
}


#endif

