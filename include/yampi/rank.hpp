#ifndef YAMPI_RANK_HPP
# define YAMPI_RANK_HPP

# include <boost/config.hpp>

# include <cassert>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/utility/enable_if.hpp>
#   include <boost/type_traits/is_integral.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
#   define YAMPI_is_integral std::is_integral
# else
#   define YAMPI_enable_if boost::enable_if_c
#   define YAMPI_is_integral boost::is_integral
# endif

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  struct host_process_t { };
  struct io_process_t { };
  struct any_source_t { };
  struct null_process_t { };

  class rank
  {
    int mpi_rank_;

   public:
    BOOST_CONSTEXPR rank() BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_rank_(0)
    { }

    explicit BOOST_CONSTEXPR rank(int const mpi_rank) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_rank_(mpi_rank)
    { }

    explicit BOOST_CONSTEXPR rank(::yampi::any_source_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_rank_(MPI_ANY_SOURCE)
    { }

    explicit BOOST_CONSTEXPR rank(::yampi::null_process_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_rank_(MPI_PROC_NULL)
    { }

    explicit rank(::yampi::host_process_t const, ::yampi::environment const& environment)
      : mpi_rank_(inquire(MPI_HOST, environment))
    { }

    explicit rank(::yampi::io_process_t const, ::yampi::environment const& environment)
      : mpi_rank_(inquire(MPI_IO, environment))
    { }

   private:
    int inquire(int const key_value, ::yampi::environment const& environment) const
    {
      // don't check flag because users cannnot delete the attribute MPI_HOST
      int* result;
      int flag;
      int const error_code = MPI_Comm_get_attr(MPI_COMM_WORLD, key_value, &result, &flag);

      return error_code == MPI_SUCCESS
        ? *result
        : throw ::yampi::error(error_code, "yampi::rank::inquire_environment", environment);
    }

   public:
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    rank(rank const&) = default;
    rank& operator=(rank const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    rank(rank&&) = default;
    rank& operator=(rank&&) = default;
#   endif
    ~rank() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    bool is_null() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_rank_ == MPI_PROC_NULL; }

    BOOST_CONSTEXPR bool operator==(rank const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_rank_ == other.mpi_rank_; }

    bool operator<(rank const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      assert(
        other.mpi_rank_ != MPI_ANY_SOURCE and other.mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return mpi_rank_ < other.mpi_rank_;
    }

    rank& operator++() BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      ++mpi_rank_;
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL);
      return *this;
    }

    rank& operator--() BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      --mpi_rank_;
      assert(mpi_rank_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      rank&>::type
    operator+=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      mpi_rank_ += n;
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      rank&>::type
    operator-=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      mpi_rank_ -= n;
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      rank&>::type
    operator*=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      assert(n >= static_cast<Integer>(0));
      mpi_rank_ *= n;
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      rank&>::type
    operator/=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      assert(n > static_cast<Integer>(0));
      mpi_rank_ /= n;
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      rank&>::type
    operator%=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      assert(n > static_cast<Integer>(0));
      mpi_rank_ %= n;
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return *this;
    }

    int operator-(rank const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      assert(
        other.mpi_rank_ != MPI_ANY_SOURCE and other.mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return mpi_rank_-other.mpi_rank_;
    }

    BOOST_CONSTEXPR int const& mpi_rank() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_rank_; }
# ifndef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
    explicit BOOST_CONSTEXPR operator int() const { return mpi_rank_; }
# else
    BOOST_CONSTEXPR operator int() const { return mpi_rank_; }
# endif

    void swap(rank& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<int>::value)
    {
      using std::swap;
      swap(mpi_rank_, other.mpi_rank_);
    }
  };

  inline BOOST_CONSTEXPR bool operator!=(::yampi::rank const& lhs, ::yampi::rank const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs == rhs); }

  inline bool operator>=(::yampi::rank const& lhs, ::yampi::rank const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs < rhs); }

  inline bool operator>(::yampi::rank const& lhs, ::yampi::rank const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return rhs < lhs; }

  inline bool operator<=(::yampi::rank const& lhs, ::yampi::rank const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (rhs < lhs); }

  inline ::yampi::rank operator++(::yampi::rank& lhs, int)
    BOOST_NOEXCEPT_OR_NOTHROW
  { ::yampi::rank result = lhs; ++lhs; return result; }

  inline ::yampi::rank operator--(::yampi::rank& lhs, int)
    BOOST_NOEXCEPT_OR_NOTHROW
  { ::yampi::rank result = lhs; --lhs; return result; }

  template <typename Integer>
  inline ::yampi::rank operator+(::yampi::rank lhs, Integer const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs += rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::rank operator-(::yampi::rank lhs, Integer const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs -= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::rank operator*(::yampi::rank lhs, Integer const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs *= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::rank operator/(::yampi::rank lhs, Integer const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs /= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::rank operator%(::yampi::rank lhs, Integer const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs %= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::rank operator+(Integer const lhs, ::yampi::rank const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return rhs+lhs; }

  template <typename Integer>
  inline ::yampi::rank operator*(Integer const lhs, ::yampi::rank const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return rhs*lhs; }

  inline void swap(::yampi::rank& lhs, ::yampi::rank& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }


  inline BOOST_CONSTEXPR ::yampi::rank any_source()
    BOOST_NOEXCEPT_OR_NOTHROW
  { return ::yampi::rank(::yampi::any_source_t()); }

  inline ::yampi::rank host_process(::yampi::environment const& environment)
  { return ::yampi::rank(::yampi::host_process_t(), environment); }

  inline ::yampi::rank io_process(::yampi::environment const& environment)
  { return ::yampi::rank(::yampi::io_process_t(), environment); }


  inline bool exists_host_process(::yampi::environment const& environment)
  { return not ::yampi::host_process(environment).is_null(); }

  inline bool is_host_process(::yampi::rank const& self, ::yampi::environment const& environment)
  { return self == ::yampi::host_process(environment); }

  inline bool exists_io_process(::yampi::environment const& environment)
  { return not ::yampi::io_process(environment).is_null(); }

  inline bool is_io_process(::yampi::rank const& self, ::yampi::environment const& environment)
  {
    ::yampi::rank const io = ::yampi::io_process(environment);
    return io == ::yampi::any_source() or self == io;
  }
}


# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_integral
# undef YAMPI_enable_if

#endif

