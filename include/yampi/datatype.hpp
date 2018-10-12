#ifndef YAMPI_DATATYPE_HPP
# define YAMPI_DATATYPE_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/utility/is_nothrow_swappable.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/address.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  struct char_datatype_t { };
  struct short_datatype_t { };
  struct int_datatype_t { };
  struct long_datatype_t { };
# ifndef BOOST_NO_LONG_LONG
  struct long_long_datatype_t { };
# endif
  struct signed_char_datatype_t { };
  struct unsigned_char_datatype_t { };
  struct unsigned_short_datatype_t { };
  struct unsigned_datatype_t { };
  struct unsigned_long_datatype_t { };
# ifndef BOOST_NO_LONG_LONG
  struct unsigned_long_long_datatype_t { };
# endif
  struct float_datatype_t { };
  struct double_datatype_t { };
  struct long_double_datatype_t { };
  struct wchar_datatype_t { };
# if MPI_VERSION >= 2
  struct bool_datatype_t { };
  struct float_complex_datatype_t { };
  struct double_complex_datatype_t { };
  struct long_double_complex_datatype_t { };
# endif
  struct short_int_datatype_t { };
  struct int_int_datatype_t { };
  struct long_int_datatype_t { };
  struct float_int_datatype_t { };
  struct double_int_datatype_t { };
  struct long_double_int_datatype_t { };

  namespace datatype_detail
  {
    inline bool is_predefined_mpi_datatype(MPI_Datatype const& mpi_datatype)
    {
      return
        mpi_datatype == MPI_CHAR or mpi_datatype == MPI_SHORT
        or mpi_datatype == MPI_INT or mpi_datatype == MPI_LONG
# ifndef BOOST_NO_LONG_LONG
        or mpi_datatype == MPI_LONG_LONG
# endif
        or mpi_datatype == MPI_SIGNED_CHAR
        or mpi_datatype == MPI_UNSIGNED_CHAR or mpi_datatype == MPI_UNSIGNED_SHORT
        or mpi_datatype == MPI_UNSIGNED or mpi_datatype == MPI_UNSIGNED_LONG
# ifndef BOOST_NO_LONG_LONG
        or mpi_datatype == MPI_UNSIGNED_LONG_LONG
# endif
        or mpi_datatype == MPI_FLOAT or mpi_datatype == MPI_DOUBLE or mpi_datatype == MPI_LONG_DOUBLE
        or mpi_datatype == MPI_WCHAR
# if MPI_VERSION >= 3 || defined(__FUJITSU)
        or mpi_datatype == MPI_CXX_BOOL or mpi_datatype == MPI_CXX_FLOAT_COMPLEX
        or mpi_datatype == MPI_CXX_DOUBLE_COMPLEX or mpi_datatype == MPI_CXX_LONG_DOUBLE_COMPLEX
# elif MPI_VERSION >= 2
        or mpi_datatype == MPI::BOOL or mpi_datatype == MPI::COMPLEX
        or mpi_datatype == MPI::DOUBLE_COMPLEX or mpi_datatype == MPI::LONG_DOUBLE_COMPLEX
# endif
        or mpi_datatype == MPI_SHORT_INT or mpi_datatype == MPI_2INT or mpi_datatype == MPI_LONG_INT
        or mpi_datatype == MPI_FLOAT_INT or mpi_datatype == MPI_DOUBLE_INT or mpi_datatype == MPI_LONG_DOUBLE_INT;
    }
  }

  class datatype
  {
    MPI_Datatype mpi_datatype_;

   public:
    datatype() BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_datatype_(MPI_DATATYPE_NULL)
    { }
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    datatype(datatype const&) = delete
    datatype& operator=(datatype const&) = delete
# else
   private:
    datatype(datatype const&);
    datatype& operator=(datatype const&);

   public:
# endif

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    datatype(datatype&& other)
      : mpi_datatype_(std::move(other.mpi_datatype_))
    { other.mpi_datatype_ = MPI_DATATYPE_NULL; }

    datatype& operator=(datatype&& other)
    {
      if (this != YAMPI_addressof(other))
      {
        mpi_datatype_ = std::move(other.mpi_datatype_);
        other.mpi_datatype_ = MPI_DATATYPE_NULL;
      }
      return *this;
    }
# endif

    ~datatype() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_datatype_ == MPI_DATATYPE_NULL or ::yampi::datatype_detail::is_predefined_mpi_datatype(mpi_datatype_))
        return;

      MPI_Type_free(YAPMI_addressof(mpi_datatype_));
    }

    explicit datatype(MPI_Datatype const& mpi_datatype) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_datatype_(mpi_datatype)
    { }

# define YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(type, mpitype) \
    explicit binary_operation(::yampi:: type ## _datatype_t const) BOOST_NOEXCEPT_OR_NOTHROW\
      : mpi_op_(MPI_ ## mpitype )\
    { }

    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(char, CHAR)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(short, SHORT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(int, INT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(long, LONG)
# ifndef BOOST_NO_LONG_LONG
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(long_long, LONG_LONG)
# endif
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(signed_char, SIGNED_CHAR)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(unsigned_char, UNSIGNED_CHAR)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(unsigned_short, UNSIGNED_SHORT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(unsigned, UNSIGNED)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(unsigned_long, UNSIGNED_LONG)
# ifndef BOOST_NO_LONG_LONG
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(unsigned_long_long, UNSIGNED_LONG_LONG)
# endif
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(float, FLOAT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(double, DOUBLE)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(long_double, LONG_DOUBLE)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(wchar, WCHAR)
# if MPI_VERSION >= 3 || defined(__FUJITSU)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(bool, CXX_BOOL)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(float_complex, CXX_FLOAT_COMPLEX)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(double_complex, CXX_DOUBLE_COMPLEX)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(long_double_complex, CXX_LONG_DOUBLE_COMPLEX)
# elif MPI_VERSION >= 2
    explicit binary_operation(::yampi::bool_datatype_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_op_(MPI::BOOL)
    { }
    explicit binary_operation(::yampi::float_complex_datatype_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_op_(MPI::COMPLEX)
    { }
    explicit binary_operation(::yampi::double_complex_datatype_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_op_(MPI::DOUBLE_COMPLEX)
    { }
    explicit binary_operation(::yampi::long_double_complex_datatype_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_op_(MPI::LONG_DOUBLE_COMPLEX)
    { }
# endif
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(short_int SHORT_INT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(int_int 2INT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(long_int LONG_INT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(float_int FLOAT_INT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(double_int DOUBLE_INT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(long_double_int LONG_DOUBLE_INT)

# undef YAMPI_DEFINE_DATATYPE_CONSTRUCTOR

    void release(::yampi::environment const& environment)
    {
      if (mpi_datatype_ == MPI_DATATYPE_NULL or ::yampi::datatype_detail::is_predefined_mpi_datatype(mpi_datatype_))
        return;

      int const error_code = MPI_Type_free(YAMPI_addressof(mpi_datatype_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::datatype::release", environment);
    }

    // TODO: implement constructor using MPI_Type_create_resized

    int size(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Type_size(mpi_datatype_, YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::datatype::size", environment);
      return result;
    }

    std::pair< ::yampi::address, ::yampi::address >
    lower_bound_extent(::yampi::environment const& environment) const
    {
      MPI_Aint lower_bound, extent;
      int const error_code
        = MPI_Type_get_extent(
            mpi_datatype_, YAMPI_addressof(lower_bound), YAMPI_addressof(extent));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(
          error_code, "yampi::datatype::lower_bound_extent", environment);
      return std::make_pair(::yampi::address(lower_bound), ::yampi::address(extent));
    }

    bool is_null() const { return mpi_datatype_ == MPI_DATATYPE_NULL; }

    bool operator==(datatype const& other) const { return mpi_datatype_ == other.mpi_datatype_; }

    MPI_Datatype const& mpi_datatype() const { return mpi_datatype_; }

    void swap(datatype& other)
      BOOST_NOEXCEPT_IF(::yampi::utility::is_nothrow_swappable<MPI_Datatype>::value)
    {
      using std::swap;
      swap(mpi_datatype_, other.mpi_datatype_);
    }
  };

  inline bool operator!=(::yampi::datatype const& lhs, ::yampi::datatype const& rhs)
  { return not (lhs == rhs); }

  inline void swap(::yampi::datatype& lhs, ::yampi::datatype& rhs)
    BOOST_NOEXCEPT_IF(::yampi::utility::is_nothrow_swappable< ::yampi::datatype >::value)
  { lhs.swap(rhs); }
}


# undef YAMPI_addressof

#endif

