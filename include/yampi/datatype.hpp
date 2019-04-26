#ifndef YAMPI_DATATYPE_HPP
# define YAMPI_DATATYPE_HPP

# include <boost/config.hpp>

# include <utility>
# include <stdexcept>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/is_same.hpp>
#   include <boost/type_traits/is_convertible.hpp>
#   include <boost/type_traits/has_nothrow_copy.hpp>
#   include <boost/type_traits/has_nothrow_assign.hpp>
#   include <boost/type_traits/is_nothrow_move_constructible.hpp>
#   include <boost/type_traits/is_nothrow_move_assignable.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/address.hpp>
# include <yampi/uncommitted_datatype.hpp>
# include <yampi/bounds.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_same std::is_same
#   define YAMPI_is_convertible std::is_convertible
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   define YAMPI_is_nothrow_copy_assignable std::is_nothrow_copy_assignable
#   define YAMPI_is_nothrow_move_constructible std::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable std::is_nothrow_move_assignable
# else
#   define YAMPI_is_same boost::is_same
#   define YAMPI_is_convertible boost::is_convertible
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   define YAMPI_is_nothrow_copy_assignable boost::has_nothrow_assign
#   define YAMPI_is_nothrow_move_constructible boost::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable boost::is_nothrow_move_assignable
# endif

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert BOOST_STATIC_ASSERT_MSG
# endif

# if MPI_VERSION >= 3
#   define YAMPI_Type_size MPI_Type_size_x
#   define YAMPI_Type_get_extent MPI_Type_get_extent_x
#   define YAMPI_Type_get_true_extent MPI_Type_get_true_extent_x
# else
#   define YAMPI_Type_size MPI_Type_size
#   define YAMPI_Type_get_extent MPI_Type_get_extent
#   define YAMPI_Type_get_true_extent MPI_Type_get_true_extent
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
    inline bool is_basic_datatype(MPI_Datatype const& mpi_datatype)
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


  class uncommitted_datatype_has_been_committed_error
    : public std::logic_error
  {
   public:
    uncommitted_datatype_has_been_committed_error()
      : std::logic_error("MPI datatype given by uncommitted_datatype has already been committed by MPI_Type_commit")
    { }
  };


  class datatype
  {
    MPI_Datatype mpi_datatype_;

   public:
# if MPI_VERSION >= 3
    typedef MPI_Count size_type;
    typedef MPI_Count count_type;
    typedef ::yampi::bounds<count_type> bounds_type;
# else
    typedef int size_type;
    typedef MPI_Aint count_type;
    typedef ::yampi::bounds<count_type> bounds_type;
# endif

    datatype()
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Datatype>::value)
      : mpi_datatype_(MPI_DATATYPE_NULL)
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    datatype(datatype const&) = delete;
    datatype& operator=(datatype const&) = delete;
# else
   private:
    datatype(datatype const&);
    datatype& operator=(datatype const&);

   public:
# endif

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    datatype(datatype&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_constructible<MPI_Datatype>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Datatype>::value)
      : mpi_datatype_(std::move(other.mpi_datatype_))
    { other.mpi_datatype_ = MPI_DATATYPE_NULL; }

    datatype& operator=(datatype&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_assignable<MPI_Datatype>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Datatype>::value)
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
      if (mpi_datatype_ == MPI_DATATYPE_NULL or ::yampi::datatype_detail::is_basic_datatype(mpi_datatype_))
        return;

      MPI_Type_free(YAMPI_addressof(mpi_datatype_));
    }

    explicit datatype(MPI_Datatype const& mpi_datatype)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Datatype>::value)
      : mpi_datatype_(mpi_datatype)
    { }

# define YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(type, mpitype) \
    explicit datatype(::yampi:: type ## _datatype_t const)\
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Datatype>::value)\
      : mpi_datatype_(MPI_ ## mpitype )\
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
    explicit datatype(::yampi::bool_datatype_t const)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Datatype>::value)
      : mpi_datatype_(MPI::BOOL)
    { }
    explicit datatype(::yampi::float_complex_datatype_t const)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Datatype>::value)
      : mpi_datatype_(MPI::COMPLEX)
    { }
    explicit datatype(::yampi::double_complex_datatype_t const)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Datatype>::value)
      : mpi_datatype_(MPI::DOUBLE_COMPLEX)
    { }
    explicit datatype(::yampi::long_double_complex_datatype_t const)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Datatype>::value)
      : mpi_datatype_(MPI::LONG_DOUBLE_COMPLEX)
    { }
# endif
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(short_int, SHORT_INT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(int_int, 2INT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(long_int, LONG_INT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(float_int, FLOAT_INT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(double_int, DOUBLE_INT)
    YAMPI_DEFINE_DATATYPE_CONSTRUCTOR(long_double_int, LONG_DOUBLE_INT)

# undef YAMPI_DEFINE_DATATYPE_CONSTRUCTOR

    datatype(
      ::yampi::uncommitted_datatype& uncommitted_datatype,
      ::yampi::environment const& environment)
      : mpi_datatype_(commit(uncommitted_datatype, environment))
    { }

    datatype(
      datatype const& old_datatype,
      ::yampi::environment const& environment)
      : mpi_datatype_(duplicate(old_datatype, environment))
    { }

   private:
    MPI_Datatype commit(
      ::yampi::uncommitted_datatype& uncommitted_datatype,
      ::yampi::environment const& environment) const
    {
      if (uncommitted_datatype.is_committed_)
        throw ::yampi::uncommitted_datatype_has_been_committed_error();

      MPI_Datatype result = uncommitted_datatype.mpi_datatype();
      int const error_code = MPI_Type_commit(YAMPI_addressof(result));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::datatype::commit", environment);

      uncommitted_datatype.is_committed_ = true;
      return result;
    }

    MPI_Datatype duplicate(
      datatype const& old_datatype,
      ::yampi::environment const& environment)
    {
      MPI_Datatype result;
      int const error_code
        = MPI_Type_dup(old_datatype.mpi_datatype_, YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::datatype::duplicate", environment);
    }

   public:
    bool operator==(datatype const& other) const
      BOOST_NOEXCEPT_OR_NOTHROW/*BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_datatype_ == other.mpi_datatype_))*/
    { return mpi_datatype_ == other.mpi_datatype_; }

    bool is_null() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_datatype_ == MPI_DATATYPE_NULL))
    { return mpi_datatype_ == MPI_DATATYPE_NULL; }

    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(
      MPI_Datatype const& mpi_datatype, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = mpi_datatype;
    }

# define YAMPI_DEFINE_DATATYPE_RESET(type, mpitype) \
    void reset(\
      ::yampi:: type ## _datatype_t const, ::yampi::environment const& environment)\
    {\
      free(environment);\
      mpi_datatype_ = MPI_ ## mpitype ;\
    }

    YAMPI_DEFINE_DATATYPE_RESET(char, CHAR)
    YAMPI_DEFINE_DATATYPE_RESET(short, SHORT)
    YAMPI_DEFINE_DATATYPE_RESET(int, INT)
    YAMPI_DEFINE_DATATYPE_RESET(long, LONG)
# ifndef BOOST_NO_LONG_LONG
    YAMPI_DEFINE_DATATYPE_RESET(long_long, LONG_LONG)
# endif
    YAMPI_DEFINE_DATATYPE_RESET(signed_char, SIGNED_CHAR)
    YAMPI_DEFINE_DATATYPE_RESET(unsigned_char, UNSIGNED_CHAR)
    YAMPI_DEFINE_DATATYPE_RESET(unsigned_short, UNSIGNED_SHORT)
    YAMPI_DEFINE_DATATYPE_RESET(unsigned, UNSIGNED)
    YAMPI_DEFINE_DATATYPE_RESET(unsigned_long, UNSIGNED_LONG)
# ifndef BOOST_NO_LONG_LONG
    YAMPI_DEFINE_DATATYPE_RESET(unsigned_long_long, UNSIGNED_LONG_LONG)
# endif
    YAMPI_DEFINE_DATATYPE_RESET(float, FLOAT)
    YAMPI_DEFINE_DATATYPE_RESET(double, DOUBLE)
    YAMPI_DEFINE_DATATYPE_RESET(long_double, LONG_DOUBLE)
    YAMPI_DEFINE_DATATYPE_RESET(wchar, WCHAR)
# if MPI_VERSION >= 3 || defined(__FUJITSU)
    YAMPI_DEFINE_DATATYPE_RESET(bool, CXX_BOOL)
    YAMPI_DEFINE_DATATYPE_RESET(float_complex, CXX_FLOAT_COMPLEX)
    YAMPI_DEFINE_DATATYPE_RESET(double_complex, CXX_DOUBLE_COMPLEX)
    YAMPI_DEFINE_DATATYPE_RESET(long_double_complex, CXX_LONG_DOUBLE_COMPLEX)
# elif MPI_VERSION >= 2
    void reset(
      ::yampi::bool_datatype_t const, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = MPI::BOOL;
    }
    void reset(
      ::yampi::float_complex_datatype_t const, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = MPI::COMPLEX;
    }
    void reset(
      ::yampi::double_complex_datatype_t const, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = MPI::DOUBLE_COMPLEX;
    }
    void reset(
      ::yampi::long_double_complex_datatype_t const,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = MPI::LONG_DOUBLE_COMPLEX;
    }
# endif
    YAMPI_DEFINE_DATATYPE_RESET(short_int, SHORT_INT)
    YAMPI_DEFINE_DATATYPE_RESET(int_int, 2INT)
    YAMPI_DEFINE_DATATYPE_RESET(long_int, LONG_INT)
    YAMPI_DEFINE_DATATYPE_RESET(float_int, FLOAT_INT)
    YAMPI_DEFINE_DATATYPE_RESET(double_int, DOUBLE_INT)
    YAMPI_DEFINE_DATATYPE_RESET(long_double_int, LONG_DOUBLE_INT)

# undef YAMPI_DEFINE_DATATYPE_RESET

    void reset(
      ::yampi::uncommitted_datatype& uncommitted_datatype,
      ::yampi::environment const& environment)
    {
      if (uncommitted_datatype.is_committed_)
        throw ::yampi::uncommitted_datatype_has_been_committed_error();

      free(environment);

      mpi_datatype_ = uncommitted_datatype.mpi_datatype();
      int const error_code = MPI_Type_commit(YAMPI_addressof(mpi_datatype_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::datatype::reset", environment);

      uncommitted_datatype.is_committed_ = true;
    }

    void reset(
      datatype const& old_datatype,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = duplicate(old_datatype, environment);
    }

    void free(::yampi::environment const& environment)
    {
      if (mpi_datatype_ == MPI_DATATYPE_NULL
          or ::yampi::datatype_detail::is_basic_datatype(mpi_datatype_))
        return;

      int const error_code = MPI_Type_free(YAMPI_addressof(mpi_datatype_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::datatype::free", environment);
    }

    size_type size(::yampi::environment const& environment) const
    {
      size_type result;
      int const error_code = YAMPI_Type_size(mpi_datatype_, YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::datatype::size", environment);
    }

    bounds_type bounds(::yampi::environment const& environment) const
    {
      count_type lower_bound, extent;
      int const error_code
        = YAMPI_Type_get_extent(
            mpi_datatype_, YAMPI_addressof(lower_bound), YAMPI_addressof(extent));
      return error_code == MPI_SUCCESS
        ? ::yampi::make_bounds(lower_bound, extent)
        : throw ::yampi::error(
            error_code, "yampi::datatype::bounds", environment);
    }

    bounds_type true_bounds(::yampi::environment const& environment) const
    {
      count_type lower_bound, extent;
      int const error_code
        = YAMPI_Type_get_true_extent(
            mpi_datatype_, YAMPI_addressof(lower_bound), YAMPI_addressof(extent));
      return error_code == MPI_SUCCESS
        ? ::yampi::make_bounds(lower_bound, extent)
        : throw ::yampi::error(
            error_code, "yampi::datatype::true_bounds", environment);
    }

    ::yampi::uncommitted_datatype to_uncommitted_datatype() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(::yampi::uncommitted_datatype(mpi_datatype_, true)))
    { return ::yampi::uncommitted_datatype(mpi_datatype_, true); }
    MPI_Datatype const& mpi_datatype() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_datatype_; }

    void swap(datatype& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Datatype>::value)
    {
      using std::swap;
      swap(mpi_datatype_, other.mpi_datatype_);
    }
  };

  inline bool operator!=(::yampi::datatype const& lhs, ::yampi::datatype const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(::yampi::datatype& lhs, ::yampi::datatype& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_Type_get_true_extent
# undef YAMPI_Type_get_extent
# undef YAMPI_Type_size
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_move_assignable
# undef YAMPI_is_nothrow_move_constructible
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible
# undef YAMPI_is_convertible
# undef YAMPI_is_same

#endif

