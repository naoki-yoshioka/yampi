#ifndef YAMPI_BINARY_OPERATION_HPP
# define YAMPI_BINARY_OPERATION_HPP

# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
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

  namespace tags
  {
# if __cplusplus >= 201703L
#   define YAMPI_DEFINE_TAG(op) \
    inline constexpr ::yampi:: op ## _t op {};
# else
#   define YAMPI_DEFINE_TAG(op) \
    constexpr ::yampi:: op ## _t op {};
# endif

    YAMPI_DEFINE_TAG(maximum)
    YAMPI_DEFINE_TAG(minimum)
    YAMPI_DEFINE_TAG(plus)
    YAMPI_DEFINE_TAG(multiplies)
    YAMPI_DEFINE_TAG(logical_and)
    YAMPI_DEFINE_TAG(bit_and)
    YAMPI_DEFINE_TAG(logical_or)
    YAMPI_DEFINE_TAG(bit_or)
    YAMPI_DEFINE_TAG(logical_xor)
    YAMPI_DEFINE_TAG(bit_xor)
    YAMPI_DEFINE_TAG(maximum_location)
    YAMPI_DEFINE_TAG(minimum_location)
    YAMPI_DEFINE_TAG(replace)
    YAMPI_DEFINE_TAG(no_op)

# undef YAMPI_DEFINE_TAG
  }

  namespace binary_operation_detail
  {
    inline bool is_predefined_binary_operation(MPI_Op const& mpi_op)
    {
      return mpi_op == MPI_MAX or mpi_op == MPI_MIN
          or mpi_op == MPI_SUM or mpi_op == MPI_PROD
          or mpi_op == MPI_LAND or mpi_op == MPI_BAND
          or mpi_op == MPI_LOR or mpi_op == MPI_BOR
          or mpi_op == MPI_LXOR or mpi_op == MPI_BXOR
          or mpi_op == MPI_MAXLOC or mpi_op == MPI_MINLOC
          or mpi_op == MPI_REPLACE or mpi_op == MPI_NO_OP;
    }
  }

  class binary_operation
  {
    MPI_Op mpi_op_;

   public:
    binary_operation() noexcept(std::is_nothrow_copy_constructible<MPI_Op>::value)
      : mpi_op_{MPI_OP_NULL}
    { }

    binary_operation(binary_operation const&) = delete;
    binary_operation& operator=(binary_operation const&) = delete;
    binary_operation(binary_operation&& other)
      noexcept(
        std::is_nothrow_move_constructible<MPI_Op>::value
        and std::is_nothrow_copy_assignable<MPI_Op>::value)
      : mpi_op_{std::move(other.mpi_op_)}
    { other.mpi_op_ = MPI_OP_NULL; }

    binary_operation& operator=(binary_operation&& other)
      noexcept(
        std::is_nothrow_move_assignable<MPI_Op>::value
        and std::is_nothrow_copy_assignable<MPI_Op>::value)
    {
      if (this != std::addressof(other))
      {
        if (mpi_op_ != MPI_OP_NULL and (not ::yampi::binary_operation_detail::is_predefined_binary_operation(mpi_op_)))
          MPI_Op_free(std::addressof(mpi_op_));
        mpi_op_ = std::move(other.mpi_op_);
        other.mpi_op_ = MPI_OP_NULL;
      }
      return *this;
    }

    ~binary_operation() noexcept
    {
      if (mpi_op_ == MPI_OP_NULL or ::yampi::binary_operation_detail::is_predefined_binary_operation(mpi_op_))
        return;

      MPI_Op_free(std::addressof(mpi_op_));
    }

    explicit binary_operation(MPI_Op const& mpi_op)
      noexcept(std::is_nothrow_copy_constructible<MPI_Op>::value)
      : mpi_op_{mpi_op}
    { }

# define YAMPI_DEFINE_OPERATION_CONSTRUCTOR(op, mpiop) \
    explicit binary_operation(::yampi:: op ## _t const)\
      noexcept(std::is_nothrow_copy_constructible<MPI_Op>::value)\
      : mpi_op_{MPI_ ## mpiop }\
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
# if MPI_VERSION >= 4
    binary_operation(
      MPI_User_function_c* mpi_user_function, bool const is_commutative,
      ::yampi::environment const& environment)
      : mpi_op_{create(mpi_user_function, is_commutative, environment)}
    { }
# else // MPI_VERSION >= 4
    binary_operation(
      MPI_User_function* mpi_user_function, bool const is_commutative,
      ::yampi::environment const& environment)
      : mpi_op_{create(mpi_user_function, is_commutative, environment)}
    { }
# endif // MPI_VERSION >= 4

   private:
# if MPI_VERSION >= 4
    MPI_Op create(
      MPI_User_function_c* mpi_user_function, bool const is_commutative,
      ::yampi::environment const& environment) const
    {
      MPI_Op result;
      int const error_code
        = MPI_Op_create_c(
            mpi_user_function, static_cast<int>(is_commutative),
            std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::binary_operation::create", environment);
    }
# else // MPI_VERSION >= 4
    MPI_Op create(
      MPI_User_function* mpi_user_function, bool const is_commutative,
      ::yampi::environment const& environment) const
    {
      MPI_Op result;
      int const error_code
        = MPI_Op_create(
            mpi_user_function, static_cast<int>(is_commutative),
            std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::binary_operation::create", environment);
    }
# endif // MPI_VERSION >= 4

   public:
    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(MPI_Op const& mpi_op, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_op_ = mpi_op;
    }

    void reset(binary_operation&& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_op_ = std::move(other.mpi_op_);
      other.mpi_op_ = MPI_OP_NULL;
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
# if MPI_VERSION >= 4
    void reset(
      MPI_User_function_c* mpi_user_function, bool const is_commutative,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_op_ = create(mpi_user_function, is_commutative, environment);
    }
# else // MPI_VERSION >= 4
    void reset(
      MPI_User_function* mpi_user_function, bool const is_commutative,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_op_ = create(mpi_user_function, is_commutative, environment);
    }
# endif // MPI_VERSION >= 4

    void free(::yampi::environment const& environment)
    {
      if (mpi_op_ == MPI_OP_NULL or ::yampi::binary_operation_detail::is_predefined_binary_operation(mpi_op_))
        return;

      int const error_code = MPI_Op_free(std::addressof(mpi_op_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::binary_operation::free", environment);
    }

    bool is_null() const
      noexcept(noexcept(mpi_op_ == MPI_OP_NULL))
    { return mpi_op_ == MPI_OP_NULL; }

    bool operator==(binary_operation const& other) const noexcept(noexcept(mpi_op_ == other.mpi_op_))
    { return mpi_op_ == other.mpi_op_; }

    MPI_Op const& mpi_op() const noexcept { return mpi_op_; }

    void swap(binary_operation& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Op>::value)
    {
      using std::swap;
      swap(mpi_op_, other.mpi_op_);
    }
  };

  inline bool operator!=(::yampi::binary_operation const& lhs, ::yampi::binary_operation const& rhs) noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(::yampi::binary_operation& lhs, ::yampi::binary_operation& rhs)
    noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }

# if __cplusplus >= 201703L
#   define YAMPI_DEFINE_OPERATION(op) \
  inline ::yampi::binary_operation op { ::yampi::tags:: op };

    YAMPI_DEFINE_OPERATION(maximum)
    YAMPI_DEFINE_OPERATION(minimum)
    YAMPI_DEFINE_OPERATION(plus)
    YAMPI_DEFINE_OPERATION(multiplies)
    YAMPI_DEFINE_OPERATION(logical_and)
    YAMPI_DEFINE_OPERATION(bit_and)
    YAMPI_DEFINE_OPERATION(logical_or)
    YAMPI_DEFINE_OPERATION(bit_or)
    YAMPI_DEFINE_OPERATION(logical_xor)
    YAMPI_DEFINE_OPERATION(bit_xor)
    YAMPI_DEFINE_OPERATION(maximum_location)
    YAMPI_DEFINE_OPERATION(minimum_location)
    YAMPI_DEFINE_OPERATION(replace)
    YAMPI_DEFINE_OPERATION(no_op)

#   undef YAMPI_DEFINE_OPERATION
# endif
}


# undef YAMPI_is_nothrow_swappable

#endif
