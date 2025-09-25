#ifndef YAMPI_COUNT_HPP
# define YAMPI_COUNT_HPP

# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# include <mpi.h>

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
    constexpr count() : mpi_count_{} { }

    count(count const&) = default;
    count& operator=(count const&) = default;
    count(count&&) = default;
    count& operator=(count&&) = default;
    ~count() noexcept = default;

# if MPI_VERSION >= 3
    explicit constexpr count(MPI_Count const& mpi_count)
      noexcept(std::is_nothrow_copy_constructible<MPI_Count>::value)
      : mpi_count_{mpi_count}
    { }

    template <typename Integer>
    explicit constexpr count(Integer const mpi_count)
      : mpi_count_{static_cast<MPI_Count>(mpi_count)}
    { static_assert(std::is_integral<Integer>::value, "Integer must be an integral type"); }

    constexpr MPI_Count const& mpi_count() const noexcept { return mpi_count_; }
    constexpr int int_mpi_count() const noexcept { return static_cast<int>(mpi_count_); }

    void swap(count& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Count>::value)
    {
      using std::swap;
      swap(mpi_count_, other.mpi_count_);
    }
# else
    constexpr count(int const mpi_count) noexcept
      : mpi_count_(mpi_count)
    { }

    template <typename Integer>
    explicit constexpr count(Integer const mpi_count)
      : mpi_count_{static_cast<int>(mpi_count)}
    { static_assert(std::is_integral<Integer>::value, "Integer must be an integral type"); }

    constexpr int const& mpi_count() const noexcept { return mpi_count_; }
    constexpr int int_mpi_count() const noexcept { return mpi_count_; }

    void swap(count& other) noexcept
    {
      using std::swap;
      swap(mpi_count_, other.mpi_count_);
    }
# endif

    constexpr bool operator==(count const& other) const noexcept
    { return mpi_count_ == other.mpi_count_; }

    constexpr bool operator<(count const& other) const noexcept
    { return mpi_count_ < other.mpi_count_; }

    count& operator++() noexcept { ++mpi_count_; return *this; }
    count& operator--() noexcept { --mpi_count_; return *this; }
    count& operator+=(count const& other) noexcept { mpi_count_ += other.mpi_count_; return *this; }
    count& operator-=(count const& other) noexcept { mpi_count_ -= other.mpi_count_; return *this; }

    template <typename Integer>
    count& operator*=(Integer const scalar) noexcept { mpi_count_ *= scalar; return *this; }

    template <typename Integer>
    count& operator/=(Integer const scalar) noexcept { mpi_count_ /= scalar; return *this; }

    template <typename Integer>
    count& operator%=(Integer const scalar) noexcept { mpi_count_ %= scalar; return *this; }
  };

  namespace literals
  {
    inline namespace count_literals
    {
# if MPI_VERSION >= 3
      inline constexpr ::yampi::count operator"" _n(unsigned long long int const value) noexcept
      { return ::yampi::count{static_cast<MPI_Count>(value)}; }
# else // MPI_VERSION >= 3
      inline constexpr ::yampi::count operator"" _n(unsigned long long int const value) noexcept
      { return ::yampi::count{static_cast<int>(value)}; }
# endif // MPI_VERSION >= 3
    }
  }

  inline constexpr bool operator!=(::yampi::count const& lhs, ::yampi::count const& rhs) noexcept
  { return not (lhs == rhs); }

  inline constexpr bool operator>(::yampi::count const& lhs, ::yampi::count const& rhs) noexcept
  { return rhs < lhs; }

  inline constexpr bool operator<=(::yampi::count const& lhs, ::yampi::count const& rhs) noexcept
  { return not (lhs > rhs); }

  inline constexpr bool operator>=(::yampi::count const& lhs, ::yampi::count const& rhs) noexcept
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

  inline void swap(::yampi::count& lhs, ::yampi::count& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif

