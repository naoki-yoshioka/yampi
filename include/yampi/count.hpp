#ifndef YAMPI_COUNT_HPP
# define YAMPI_COUNT_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/has_nothrow_copy.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# include <mpi.h>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
# else
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
# endif

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  class count
  {
# if MPI_VERSION >= 3
    MPI_Count mpi_count_;
# else
    int mpi_count_;
# endif

   public:
    BOOST_CONSTEXPR count() : mpi_count_() { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    count(count const&) = default;
    count& operator=(count const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    count(count&&) = default;
    count& operator=(count&&) = default;
#   endif
    ~count() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

# if MPI_VERSION >= 3
    BOOST_CONSTEXPR count(int const mpi_count) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_count_(static_cast<MPI_Count>(mpi_count))
    { }

    explicit BOOST_CONSTEXPR count(MPI_Count const& mpi_count)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Count>::value)
      : mpi_count_(mpi_count)
    { }

    BOOST_CONSTEXPR MPI_Count const& mpi_count() const { return mpi_count_; }
    BOOST_CONSTEXPR int int_mpi_count() const { return static_cast<int>(mpi_count_); }

    void swap(count& other) BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Count>::value)
    {
      using std::swap;
      swap(mpi_count_, other.mpi_count_);
    }
# else
    BOOST_CONSTEXPR count(int const mpi_count) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_count_(mpi_count)
    { }

    BOOST_CONSTEXPR int const& mpi_count() const { return mpi_count_; }
    BOOST_CONSTEXPR int int_mpi_count() const { return mpi_count_; }

    void swap(count& other) BOOST_NOEXCEPT_OR_NOTHROW
    {
      using std::swap;
      swap(mpi_count_, other.mpi_count_);
    }
# endif

    BOOST_CONSTEXPR bool operator==(count const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_count_ == other.mpi_count_; }

    BOOST_CONSTEXPR bool operator<(count const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_count_ < other.mpi_count_; }

    count& operator++() BOOST_NOEXCEPT_OR_NOTHROW
    { ++mpi_count_; return *this; }

    count& operator--() BOOST_NOEXCEPT_OR_NOTHROW
    { --mpi_count_; return *this; }

    count& operator+=(count const& other) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_count_ += other.mpi_count_; return *this; }

    count& operator-=(count const& other) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_count_ -= other.mpi_count_; return *this; }

    template <typename Integer>
    count& operator*=(Integer const scalar) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_count_ *= scalar; return *this; }

    template <typename Integer>
    count& operator/=(Integer const scalar) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_count_ /= scalar; return *this; }

    template <typename Integer>
    count& operator%=(Integer const scalar) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_count_ %= scalar; return *this; }
  };

  inline BOOST_CONSTEXPR bool operator!=(::yampi::count const& lhs, ::yampi::count const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs == rhs); }

  inline BOOST_CONSTEXPR bool operator>(::yampi::count const& lhs, ::yampi::count const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return rhs < lhs; }

  inline BOOST_CONSTEXPR bool operator<=(::yampi::count const& lhs, ::yampi::count const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs > rhs); }

  inline BOOST_CONSTEXPR bool operator>=(::yampi::count const& lhs, ::yampi::count const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs < rhs); }

  inline ::yampi::count operator++(::yampi::count& self, int)
  { ::yampi::count result = self; ++self; return result; }

  inline ::yampi::count operator--(::yampi::count& self, int)
  { ::yampi::count result = self; --self; return result; }

  inline ::yampi::count operator+(::yampi::count lhs, ::yampi::count const& rhs)
  { lhs += rhs; return lhs; }

  inline ::yampi::count operator-(::yampi::count lhs, ::yampi::count const& rhs)
  { lhs -= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::count operator*(::yampi::count lhs, Integer const scalar)
  { lhs *= scalar; return lhs; }

  template <typename Integer>
  inline ::yampi::count operator*(Integer const scalar, ::yampi::count rhs)
  { rhs *= scalar; return rhs; }

  template <typename Integer>
  inline ::yampi::count operator/(::yampi::count lhs, Integer const scalar)
  { lhs /= scalar; return lhs; }

  template <typename Integer>
  inline ::yampi::count operator%(::yampi::count lhs, Integer const scalar)
  { lhs %= scalar; return lhs; }

  inline void swap(::yampi::count& lhs, ::yampi::count& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_copy_constructible

#endif

