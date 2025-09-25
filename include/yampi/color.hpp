#ifndef YAMPI_COLOR_HPP
# define YAMPI_COLOR_HPP

# include <cassert>
# include <string>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <stdexcept>

# include <mpi.h>

# include <yampi/environment.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  struct undefined_color_t { };

  namespace tags
  {
# if __cplusplus >= 201703L
    inline constexpr ::yampi::undefined_color_t undefined_color{};
# else
    constexpr ::yampi::undefined_color_t undefined_color{};
# endif
  }

  class color
  {
    int mpi_color_;

   public:
    constexpr color() noexcept : mpi_color_{0} { }

    explicit constexpr color(int const mpi_color) noexcept : mpi_color_{mpi_color} { }

    explicit constexpr color(::yampi::undefined_color_t const) noexcept
      : mpi_color_{MPI_UNDEFINED}
    { }

    color(color const&) = default;
    color& operator=(color const&) = default;
    color(color&&) = default;
    color& operator=(color&&) = default;
    ~color() noexcept = default;

    constexpr bool operator==(color const& other) const noexcept
    { return mpi_color_ == other.mpi_color_; }

    bool operator<(color const& other) const noexcept
    {
      assert(mpi_color_ >= 0);
      assert(other.mpi_color_ >= 0);
      return mpi_color_ < other.mpi_color_;
    }

    color& operator++() noexcept
    {
      assert(mpi_color_ >= 0);
      ++mpi_color_;
      return *this;
    }

    color& operator--() noexcept
    {
      assert(mpi_color_ >= 0);
      --mpi_color_;
      assert(mpi_color_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      color&>::type
    operator+=(Integer const n) noexcept
    {
      assert(mpi_color_ >= 0);
      mpi_color_ += n;
      assert(mpi_color_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      color&>::type
    operator-=(Integer const n) noexcept
    {
      assert(mpi_color_ >= 0);
      mpi_color_ -= n;
      assert(mpi_color_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      color&>::type
    operator*=(Integer const n) noexcept
    {
      assert(mpi_color_ >= 0);
      assert(n >= static_cast<Integer>(0));
      mpi_color_ *= n;
      assert(mpi_color_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      color&>::type
    operator/=(Integer const n) noexcept
    {
      assert(mpi_color_ >= 0);
      assert(n > static_cast<Integer>(0));
      mpi_color_ /= n;
      assert(mpi_color_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      color&>::type
    operator%=(Integer const n) noexcept
    {
      assert(mpi_color_ >= 0);
      assert(n > static_cast<Integer>(0));
      mpi_color_ %= n;
      assert(mpi_color_ >= 0);
      return *this;
    }

    int operator-(color const& other) const noexcept
    {
      assert(mpi_color_ >= 0);
      assert(other.mpi_color_ >= 0);
      return mpi_color_-other.mpi_color_;
    }

    constexpr int const& mpi_color() const noexcept { return mpi_color_; }
    explicit constexpr operator int() const { return mpi_color_; }

    void swap(color& other) noexcept(YAMPI_is_nothrow_swappable<int>::value)
    {
      using std::swap;
      swap(mpi_color_, other.mpi_color_);
    }
  };

  namespace literals
  {
    inline namespace color_literals
    {
      inline constexpr ::yampi::color operator"" _c(unsigned long long int const value) noexcept
      { return ::yampi::color{static_cast<int>(value)}; }
    }
  }

  inline constexpr bool operator!=(::yampi::color const& lhs, ::yampi::color const& rhs) noexcept
  { return not (lhs == rhs); }

  inline bool operator>=(::yampi::color const& lhs, ::yampi::color const& rhs) noexcept
  { return not (lhs < rhs); }

  inline bool operator>(::yampi::color const& lhs, ::yampi::color const& rhs) noexcept
  { return rhs < lhs; }

  inline bool operator<=(::yampi::color const& lhs, ::yampi::color const& rhs) noexcept
  { return not (rhs < lhs); }

  inline ::yampi::color operator++(::yampi::color& lhs, int) noexcept
  { ::yampi::color result = lhs; ++lhs; return result; }

  inline ::yampi::color operator--(::yampi::color& lhs, int) noexcept
  { ::yampi::color result = lhs; --lhs; return result; }

  template <typename Integer>
  inline ::yampi::color operator+(::yampi::color lhs, Integer const rhs) noexcept
  { lhs += rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::color operator-(::yampi::color lhs, Integer const rhs) noexcept
  { lhs -= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::color operator*(::yampi::color lhs, Integer const rhs) noexcept
  { lhs *= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::color operator/(::yampi::color lhs, Integer const rhs) noexcept
  { lhs /= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::color operator%(::yampi::color lhs, Integer const rhs) noexcept
  { lhs %= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::color operator+(Integer const lhs, ::yampi::color const rhs) noexcept
  { return rhs+lhs; }

  template <typename Integer>
  inline ::yampi::color operator*(Integer const lhs, ::yampi::color const rhs) noexcept
  { return rhs*lhs; }

  inline void swap(::yampi::color& lhs, ::yampi::color& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }

# if __cplusplus >= 201703L
  inline constexpr ::yampi::color undefined_color{::yampi::tags::undefined_color};
# else
  constexpr ::yampi::color undefined_color{::yampi::tags::undefined_color};
# endif
}


# undef YAMPI_is_nothrow_swappable

#endif

