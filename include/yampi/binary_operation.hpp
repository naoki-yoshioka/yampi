#ifndef YAMPI_BINARY_OPERATION_HPP
# define YAMPI_BINARY_OPERATION_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/has_nothrow_copy.hpp>
#   include <boost/type_traits/has_nothrow_assign.hpp>
#   include <boost/type_traits/is_nothrow_move_constructible.hpp>
#   include <boost/type_traits/is_nothrow_move_assignable.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# if __cplusplus >= 201703L
#   include <type_traits>
# else
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   define YAMPI_is_nothrow_copy_assignable std::is_nothrow_copy_assignable
#   define YAMPI_is_nothrow_move_constructible std::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable std::is_nothrow_move_assignable
# else
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


namespace yampi
{
  struct maximum_t { };
  struct minimum_t { };
  struct plus_t { };
  struct multiplies_t { };
  struct logical_and_t { };
  struct bit_and_t { };
  struct logical_or_t { };
  struct bit_or_t { };
  struct logical_xor_t { };
  struct bit_xor_t { };
  struct maximum_location_t { };
  struct minimum_location_t { };
  struct replace_t { }; // for yampi::accumulate, etc.
  struct no_op_t { }; // for yampi::fetch_accumulate, etc.

  class binary_operation
  {
    MPI_Op mpi_op_;

   public:
    binary_operation()
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Op>::value)
      : mpi_op_(MPI_OP_NULL)
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    binary_operation(binary_operation const&) = delete;
    binary_operation& operator=(binary_operation const&) = delete;
# else
   private:
    binary_operation(binary_operation const&);
    binary_operation& operator=(binary_operation const&);

   public:
# endif
# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    binary_operation(binary_operation&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_constructible<MPI_Op>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Op>::value)
      : mpi_op_(std::move(other.mpi_op_))
    { other.mpi_op_ = MPI_OP_NULL; }

    binary_operation& operator=(binary_operation&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_assignable<MPI_Op>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Op>::value)
    {
      if (this != YAMPI_addressof(other))
      {
        mpi_op_ = std::move(other.mpi_op_);
        other.mpi_op_ = MPI_OP_NULL;
      }
      return *this;
    }
# endif

    ~binary_operation() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_op_ == MPI_OP_NULL
          or mpi_op_ == MPI_MAX or mpi_op_ == MPI_MIN
          or mpi_op_ == MPI_SUM or mpi_op_ == MPI_PROD
          or mpi_op_ == MPI_LAND or mpi_op_ == MPI_BAND
          or mpi_op_ == MPI_LOR or mpi_op_ == MPI_BOR
          or mpi_op_ == MPI_LXOR or mpi_op_ == MPI_BXOR
          or mpi_op_ == MPI_MAXLOC or mpi_op_ == MPI_MINLOC
          or mpi_op_ == MPI_REPLACE or mpi_op_ == MPI_NO_OP)
        return;

      MPI_Op_free(YAMPI_addressof(mpi_op_));
    }

    explicit binary_operation(MPI_Op const& mpi_op)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Op>::value)
      : mpi_op_(mpi_op)
    { }

# define YAMPI_DEFINE_OPERATION_CONSTRUCTOR(op, mpiop) \
    explicit binary_operation(::yampi:: op ## _t const)\
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Op>::value)\
      : mpi_op_(MPI_ ## mpiop )\
    { }

    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(maximum, MAX)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(minimum, MIN)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(plus, SUM)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(multiplies, PROD)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(logical_and, LAND)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(bit_and, BAND)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(logical_or, LOR)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(bit_or, BOR)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(logical_xor, LXOR)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(bit_xor, BXOR)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(maximum_location, MAXLOC)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(minimum_location, MINLOC)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(replace, REPLACE)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(no_op, NO_OP)

# undef YAMPI_DEFINE_OPERATION_CONSTRUCTOR

    // TODO: Implement something like yampi::function and constructor using it
    binary_operation(
      MPI_User_function* mpi_user_function, bool const is_commutative,
      ::yampi::environment const& environment)
      : mpi_op_(create(mpi_user_function, is_commutative, environment))
    { }

   private:
    MPI_Op create(
      MPI_User_function* mpi_user_function, bool const is_commutative,
      ::yampi::environment const& environment) const
    {
      MPI_Op result;
      int const error_code
        = MPI_Op_create(
            mpi_user_function, static_cast<int>(is_commutative),
            YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::binary_operation::create", environment);
    }

   public:
    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(MPI_Op const& mpi_op, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_op_ = mpi_op;
    }

# define YAMPI_DEFINE_OPERATION_RESET(op, mpiop) \
    void reset(::yampi:: op ## _t const, ::yampi::environment const& environment)\
    {\
      free(environment);\
      mpi_op_ = MPI_ ## mpiop ;\
    }

    YAMPI_DEFINE_OPERATION_RESET(maximum, MAX)
    YAMPI_DEFINE_OPERATION_RESET(minimum, MIN)
    YAMPI_DEFINE_OPERATION_RESET(plus, SUM)
    YAMPI_DEFINE_OPERATION_RESET(multiplies, PROD)
    YAMPI_DEFINE_OPERATION_RESET(logical_and, LAND)
    YAMPI_DEFINE_OPERATION_RESET(bit_and, BAND)
    YAMPI_DEFINE_OPERATION_RESET(logical_or, LOR)
    YAMPI_DEFINE_OPERATION_RESET(bit_or, BOR)
    YAMPI_DEFINE_OPERATION_RESET(logical_xor, LXOR)
    YAMPI_DEFINE_OPERATION_RESET(bit_xor, BXOR)
    YAMPI_DEFINE_OPERATION_RESET(maximum_location, MAXLOC)
    YAMPI_DEFINE_OPERATION_RESET(minimum_location, MINLOC)
    YAMPI_DEFINE_OPERATION_RESET(replace, REPLACE)
    YAMPI_DEFINE_OPERATION_RESET(no_op, NO_OP)

# undef YAMPI_DEFINE_OPERATION_RESET

    // TODO: Implement something like yampi::function
    void reset(
      MPI_User_function* mpi_user_function, bool const is_commutative,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_op_ = create(mpi_user_function, is_commutative, environment);
    }

    void free(::yampi::environment const& environment)
    {
      if (mpi_op_ == MPI_OP_NULL
          or mpi_op_ == MPI_MAX or mpi_op_ == MPI_MIN
          or mpi_op_ == MPI_SUM or mpi_op_ == MPI_PROD
          or mpi_op_ == MPI_LAND or mpi_op_ == MPI_BAND
          or mpi_op_ == MPI_LOR or mpi_op_ == MPI_BOR
          or mpi_op_ == MPI_LXOR or mpi_op_ == MPI_BXOR
          or mpi_op_ == MPI_MAXLOC or mpi_op_ == MPI_MINLOC
          or mpi_op_ == MPI_REPLACE or mpi_op_ == MPI_NO_OP)
        return;

      int const error_code = MPI_Op_free(YAMPI_addressof(mpi_op_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::binary_operation::free", environment);
    }


    bool is_null() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_op_ == MPI_OP_NULL))
    { return mpi_op_ == MPI_OP_NULL; }

    bool operator==(binary_operation const& other) const
      BOOST_NOEXCEPT_OR_NOTHROW/*BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_op_ == other.mpi_op_))*/
    { return mpi_op_ == other.mpi_op_; }

    MPI_Op const& mpi_op() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_op_; }

    void swap(binary_operation& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Op>::value)
    {
      using std::swap;
      swap(mpi_op_, other.mpi_op_);
    }
  };

  inline bool operator!=(
    ::yampi::binary_operation const& lhs, ::yampi::binary_operation const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(::yampi::binary_operation& lhs, ::yampi::binary_operation& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_addressof
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_move_assignable
# undef YAMPI_is_nothrow_move_constructible
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible

#endif
