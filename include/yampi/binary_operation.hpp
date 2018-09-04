#ifndef YAMPI_BINARY_OPERATION_HPP
# define YAMPI_BINARY_OPERATION_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/utility/is_nothrow_swappable.hpp>

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

  class binary_operation
  {
    MPI_Op mpi_op_;

   public:
    binary_operation() : mpi_op_(MPI_OP_NULL) { }
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    binary_operation(binary_operation const&) = delete;
    binary_operation& operator=(binary_operation const&) = delete;
# else
   private:
    binary_operation(binary_operation const&);
    binary_operation& operator=(binary_operation const&);

   public:
# endif
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    binary_operation(binary_operation&&) = default;
    binary_operation& operator=(binary_operation&&) = default;
#   else
    binary_operation(binary_operation&& other) : mpi_op_(std::move(other.mpi_op_)) { }
    binary_operation& operator=(binary_operation&& other)
    {
      if (this != YAMPI_addressof(other))
        mpi_op_ = std::move(other.mpi_op_);
      return *this;
    }
#   endif
# endif

    ~binary_operation() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_op_ == MPI_OP_NULL
          or mpi_op_ == MPI_MAX or mpi_op_ == MPI_MIN
          or mpi_op_ == MPI_SUM or mpi_op_ == MPI_PROD
          or mpi_op_ == MPI_LAND or mpi_op_ == MPI_BAND
          or mpi_op_ == MPI_LOR or mpi_op_ == MPI_BOR
          or mpi_op_ == MPI_LXOR or mpi_op_ == MPI_BXOR
          or mpi_op_ == MPI_MAXLOC or mpi_op_ == MPI_MINLOC)
        return;

      MPI_Op_free(YAMPI_addressof(mpi_op_));
    }

    explicit binary_operation(MPI_Op const mpi_op) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_op_(mpi_op)
    { }

# define YAMPI_DEFINE_OPERATION_CONSTRUCTOR(op, mpiop) \
    explicit binary_operation(::yampi:: op ## _t const) BOOST_NOEXCEPT_OR_NOTHROW\
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
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::binary_operation::create", environment);

      return result;
    }

   public:
    void release(::yampi::environment const& environment)
    {
      if (mpi_op_ == MPI_OP_NULL
          or mpi_op_ == MPI_MAX or mpi_op_ == MPI_MIN
          or mpi_op_ == MPI_SUM or mpi_op_ == MPI_PROD
          or mpi_op_ == MPI_LAND or mpi_op_ == MPI_BAND
          or mpi_op_ == MPI_LOR or mpi_op_ == MPI_BOR
          or mpi_op_ == MPI_LXOR or mpi_op_ == MPI_BXOR
          or mpi_op_ == MPI_MAXLOC or mpi_op_ == MPI_MINLOC)
        return;

      int const error_code = MPI_Op_free(YAMPI_addressof(mpi_op_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::binary_operation::release", environment);
    }


    bool is_null() const { return mpi_op_ == MPI_OP_NULL; }

    bool operator==(binary_operation const& other) const
    { return mpi_op_ == other.mpi_op_; }

    MPI_Op const& mpi_op() const { return mpi_op_; }

    void swap(binary_operation& other)
      BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable<MPI_Op>::value ))
    {
      using std::swap;
      swap(mpi_op_, other.mpi_op_);
    }
  };

  inline bool operator!=(
    ::yampi::binary_operation const& lhs, ::yampi::binary_operation const& rhs)
  { return not (lhs == rhs); }

  inline void swap(::yampi::binary_operation& lhs, ::yampi::binary_operation& rhs)
    BOOST_NOEXCEPT_IF((
      ::yampi::utility::is_nothrow_swappable< ::yampi::binary_operation >::value ))
  { lhs.swap(rhs); }
}


# undef YAMPI_addressof

#endif
