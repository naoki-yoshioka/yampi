#ifndef YAMPI_PARTITION_HPP
# define YAMPI_PARTITION_HPP

# include <cassert>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif

# if MPI_VERSION >= 4
namespace yampi
{
  class partition
  {
    int mpi_partition_;

   public:
    constexpr partition() noexcept : mpi_partition_{0} { }

    explicit constexpr partition(int const mpi_partition) noexcept : mpi_partition_{mpi_partition} { }

    partition(partition const&) = default;
    partition& operator=(partition const&) = default;
    partition(partition&&) = default;
    partition& operator=(partition&&) = default;
    ~partition() noexcept = default;

    constexpr bool operator==(partition const& other) const noexcept
    { return mpi_partition_ == other.mpi_partition_; }

    bool operator<(partition const& other) const noexcept
    {
      assert(mpi_partition_ >= 0);
      assert(other.mpi_partition_ >= 0);
      return mpi_partition_ < other.mpi_partition_;
    }

    partition& operator++() noexcept
    {
      assert(mpi_partition_ >= 0);
      ++mpi_partition_;
      return *this;
    }

    partition& operator--() noexcept
    {
      assert(mpi_partition_ >= 0);
      --mpi_partition_;
      assert(mpi_partition_ >= 0);
      return *this;
    }

    template <typename Integer>
    std::enable_if_t<std::is_integral<Integer>::value, partition&>
    operator+=(Integer const n) noexcept
    {
      assert(mpi_partition_ >= 0);
      mpi_partition_ += n;
      assert(mpi_partition_ >= 0);
      return *this;
    }

    template <typename Integer>
    std::enable_if_t<std::is_integral<Integer>::value, partition&>
    operator-=(Integer const n) noexcept
    {
      assert(mpi_partition_ >= 0);
      mpi_partition_ -= n;
      assert(mpi_partition_ >= 0);
      return *this;
    }

    template <typename Integer>
    std::enable_if_t<std::is_integral<Integer>::value, partition&>
    operator*=(Integer const n) noexcept
    {
      assert(mpi_partition_ >= 0);
      assert(n >= static_cast<Integer>(0));
      mpi_partition_ *= n;
      assert(mpi_partition_ >= 0);
      return *this;
    }

    template <typename Integer>
    std::enable_if_t<std::is_integral<Integer>::value, partition&>
    operator/=(Integer const n) noexcept
    {
      assert(mpi_partition_ >= 0);
      assert(n > static_cast<Integer>(0));
      mpi_partition_ /= n;
      assert(mpi_partition_ >= 0);
      return *this;
    }

    template <typename Integer>
    std::enable_if_t<std::is_integral<Integer>::value, partition&>
    operator%=(Integer const n) noexcept
    {
      assert(mpi_partition_ >= 0);
      assert(n > static_cast<Integer>(0));
      mpi_partition_ %= n;
      assert(mpi_partition_ >= 0);
      return *this;
    }

    int operator-(partition const& other) const noexcept
    {
      assert(mpi_partition_ >= 0);
      assert(other.mpi_partition_ >= 0);
      return mpi_partition_ - other.mpi_partition_;
    }

    constexpr int const& mpi_partition() const noexcept { return mpi_partition_; }
    explicit constexpr operator int() const { return mpi_partition_; }

    void swap(partition& other) noexcept(YAMPI_is_nothrow_swappable<int>::value)
    {
      using std::swap;
      swap(mpi_partition_, other.mpi_partition_);
    }
  };

  namespace literals
  {
    inline namespace partition_literals
    {
      inline constexpr ::yampi::partition operator"" _p(unsigned long long int const value) noexcept
      { return ::yampi::partition{static_cast<int>(value)}; }
    }
  }

  inline constexpr bool operator!=(::yampi::partition const& lhs, ::yampi::partition const& rhs) noexcept
  { return not (lhs == rhs); }

  inline bool operator>=(::yampi::partition const& lhs, ::yampi::partition const& rhs) noexcept
  { return not (lhs < rhs); }

  inline bool operator>(::yampi::partition const& lhs, ::yampi::partition const& rhs) noexcept
  { return rhs < lhs; }

  inline bool operator<=(::yampi::partition const& lhs, ::yampi::partition const& rhs) noexcept
  { return not (rhs < lhs); }

  inline ::yampi::partition operator++(::yampi::partition& lhs, int) noexcept
  { auto result = lhs; ++lhs; return result; }

  inline ::yampi::partition operator--(::yampi::partition& lhs, int) noexcept
  { auto result = lhs; --lhs; return result; }

  template <typename Integer>
  inline ::yampi::partition operator+(::yampi::partition lhs, Integer const rhs) noexcept
  { lhs += rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::partition operator-(::yampi::partition lhs, Integer const rhs) noexcept
  { lhs -= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::partition operator*(::yampi::partition lhs, Integer const rhs) noexcept
  { lhs *= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::partition operator/(::yampi::partition lhs, Integer const rhs) noexcept
  { lhs /= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::partition operator%(::yampi::partition lhs, Integer const rhs) noexcept
  { lhs %= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::partition operator+(Integer const lhs, ::yampi::partition const& rhs) noexcept
  { return rhs + lhs; }

  template <typename Integer>
  inline ::yampi::partition operator*(Integer const lhs, ::yampi::partition const& rhs) noexcept
  { return rhs * lhs; }

  inline void swap(::yampi::partition& lhs, ::yampi::partition& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}
# endif // MPI_VERSION >= 4

# undef YAMPI_is_nothrow_swappable

#endif

