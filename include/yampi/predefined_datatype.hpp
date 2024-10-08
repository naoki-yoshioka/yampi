#ifndef YAMPI_PREDEFINED_DATATYPE_HPP
# define YAMPI_PREDEFINED_DATATYPE_HPP

# include <utility>

# include <mpi.h>

# if MPI_VERSION >= 2
#   include <complex>
# endif

# include <yampi/datatype_base.hpp>
# include <yampi/address.hpp>
# include <yampi/byte_displacement.hpp>
# include <yampi/offset.hpp>
# include <yampi/count.hpp>
# include <yampi/extent.hpp>


namespace yampi
{
  namespace predefined_datatype_detail
  {
    // These typedef's are required because the original types std::pair<foo, int>'s have commas
    typedef std::pair<short, int> short_int_type;
    typedef std::pair<int, int> int_int_type;
    typedef std::pair<long, int> long_int_type;
    typedef std::pair<float, int> float_int_type;
    typedef std::pair<double, int> double_int_type;
    typedef std::pair<long double, int> long_double_int_type;

    template <typename T>
    struct mpi_datatype_of;

    // Implemented by using a static member function, not by using a static constant expression member variable
    // This is because sometimes MPI_Datatype instances are not defined as a constant expression
# define YAMPI_MAKE_MPI_DATATYPE_OF(type, mpitype) \
    template <>\
    struct mpi_datatype_of< type >\
    {\
      static MPI_Datatype call() { return MPI_ ## mpitype; }\
    };

    YAMPI_MAKE_MPI_DATATYPE_OF(char, CHAR)
    YAMPI_MAKE_MPI_DATATYPE_OF(signed short int, SHORT)
    YAMPI_MAKE_MPI_DATATYPE_OF(signed int, INT)
    YAMPI_MAKE_MPI_DATATYPE_OF(signed long int, LONG)
    YAMPI_MAKE_MPI_DATATYPE_OF(signed long long int, LONG_LONG)
    YAMPI_MAKE_MPI_DATATYPE_OF(signed char, SIGNED_CHAR)
    YAMPI_MAKE_MPI_DATATYPE_OF(unsigned char, UNSIGNED_CHAR)
    YAMPI_MAKE_MPI_DATATYPE_OF(unsigned short int, UNSIGNED_SHORT)
    YAMPI_MAKE_MPI_DATATYPE_OF(unsigned int, UNSIGNED)
    YAMPI_MAKE_MPI_DATATYPE_OF(unsigned long int, UNSIGNED_LONG)
    YAMPI_MAKE_MPI_DATATYPE_OF(unsigned long long int, UNSIGNED_LONG_LONG)
    YAMPI_MAKE_MPI_DATATYPE_OF(float, FLOAT)
    YAMPI_MAKE_MPI_DATATYPE_OF(double, DOUBLE)
    YAMPI_MAKE_MPI_DATATYPE_OF(long double, LONG_DOUBLE)
    YAMPI_MAKE_MPI_DATATYPE_OF(wchar_t, WCHAR)

    // No template specializations for types like std::int8_t because they are typedefs of fundamental types

    YAMPI_MAKE_MPI_DATATYPE_OF(yampi::address, AINT)
    YAMPI_MAKE_MPI_DATATYPE_OF(yampi::byte_displacement, AINT)
    YAMPI_MAKE_MPI_DATATYPE_OF(yampi::offset, OFFSET)
# if MPI_VERSION >= 3
    YAMPI_MAKE_MPI_DATATYPE_OF(yampi::count, COUNT)
    YAMPI_MAKE_MPI_DATATYPE_OF(yampi::extent, COUNT)
# else
    YAMPI_MAKE_MPI_DATATYPE_OF(yampi::count, INT)
    YAMPI_MAKE_MPI_DATATYPE_OF(yampi::extent, AINT)
# endif

# if MPI_VERSION >= 3
    YAMPI_MAKE_MPI_DATATYPE_OF(bool, CXX_BOOL)
    YAMPI_MAKE_MPI_DATATYPE_OF(std::complex<float>, CXX_FLOAT_COMPLEX)
    YAMPI_MAKE_MPI_DATATYPE_OF(std::complex<double>, CXX_DOUBLE_COMPLEX)
    YAMPI_MAKE_MPI_DATATYPE_OF(std::complex<long double>, CXX_LONG_DOUBLE_COMPLEX)
# elif MPI_VERSION >= 2
#   define YAMPI_MAKE_MPI_DATATYPE_OF_FOR_MPI2CXX(type, mpitype) \
    template <>\
    struct mpi_datatype_of< type >\
    {\
      static MPI_Datatype call() { return MPI:: mpitype; }\
    };
    YAMPI_MAKE_MPI_DATATYPE_OF_FOR_MPI2CXX(bool, BOOL)
    YAMPI_MAKE_MPI_DATATYPE_OF_FOR_MPI2CXX(std::complex<float>, COMPLEX)
    YAMPI_MAKE_MPI_DATATYPE_OF_FOR_MPI2CXX(std::complex<double>, DOUBLE_COMPLEX)
    YAMPI_MAKE_MPI_DATATYPE_OF_FOR_MPI2CXX(std::complex<long double>, LONG_DOUBLE_COMPLEX)
#   undef YAMPI_MAKE_MPI_DATATYPE_OF_FOR_MPI2CXX
# endif

    YAMPI_MAKE_MPI_DATATYPE_OF(short_int_type, SHORT_INT)
    YAMPI_MAKE_MPI_DATATYPE_OF(int_int_type, 2INT)
    YAMPI_MAKE_MPI_DATATYPE_OF(long_int_type, LONG_INT)
    YAMPI_MAKE_MPI_DATATYPE_OF(float_int_type, FLOAT_INT)
    YAMPI_MAKE_MPI_DATATYPE_OF(double_int_type, DOUBLE_INT)
    YAMPI_MAKE_MPI_DATATYPE_OF(long_double_int_type, LONG_DOUBLE_INT)

# undef YAMPI_MAKE_MPI_DATATYPE_OF
  } // namespace predefined_datatype_detail

  template <typename T>
  class predefined_datatype
    : public ::yampi::datatype_base< ::yampi::predefined_datatype<T> >
  {
    typedef ::yampi::datatype_base< ::yampi::predefined_datatype<T> > base_type;
    friend base_type;

    constexpr bool do_is_null() const noexcept { return false; }

    MPI_Datatype do_mpi_datatype() const noexcept { return ::yampi::predefined_datatype_detail::mpi_datatype_of<T>::call(); }
  }; // class predefined_datatype<T>
} // namespace yampi

#endif // YAMPI_PREDEFINED_DATATYPE_HPP
