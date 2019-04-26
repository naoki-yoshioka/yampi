#ifndef YAMPI_COLOR_HPP
# define YAMPI_COLOR_HPP

# include <boost/config.hpp>

# include <cassert>
# include <string>
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
# include <stdexcept>

# include <mpi.h>

# include <yampi/environment.hpp>

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
  struct undefined_color_t { };

  class color
  {
    int mpi_color_;

   public:
    BOOST_CONSTEXPR color() BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_color_(0)
    { }

    explicit BOOST_CONSTEXPR color(int const mpi_color) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_color_(mpi_color)
    { }

    explicit BOOST_CONSTEXPR color(::yampi::undefined_color_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_color_(MPI_UNDEFINED)
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    color(color const&) = default;
    color& operator=(color const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    color(color&&) = default;
    color& operator=(color&&) = default;
#   endif
    ~color() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    BOOST_CONSTEXPR bool operator==(color const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_color_ == other.mpi_color_; }

    bool operator<(color const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_color_ >= 0);
      assert(other.mpi_color_ >= 0);
      return mpi_color_ < other.mpi_color_;
    }

    color& operator++() BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_color_ >= 0);
      ++mpi_color_;
      return *this;
    }

    color& operator--() BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_color_ >= 0);
      --mpi_color_;
      assert(mpi_color_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      color&>::type
    operator+=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_color_ >= 0);
      mpi_color_ += n;
      assert(mpi_color_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      color&>::type
    operator-=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_color_ >= 0);
      mpi_color_ -= n;
      assert(mpi_color_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      color&>::type
    operator*=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_color_ >= 0);
      assert(n >= static_cast<Integer>(0));
      mpi_color_ *= n;
      assert(mpi_color_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      color&>::type
    operator/=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_color_ >= 0);
      assert(n > static_cast<Integer>(0));
      mpi_color_ /= n;
      assert(mpi_color_ >= 0);
      return *this;
    }

    int operator-(color const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_color_ >= 0);
      assert(other.mpi_color_ >= 0);
      return mpi_color_-other.mpi_color_;
    }

    BOOST_CONSTEXPR int const& mpi_color() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_color_; }
# ifndef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
    explicit BOOST_CONSTEXPR operator int() const { return mpi_color_; }
# else
    BOOST_CONSTEXPR operator int() const { return mpi_color_; }
# endif

    void swap(color& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<int>::value)
    {
      using std::swap;
      swap(mpi_color_, other.mpi_color_);
    }
  };

  inline BOOST_CONSTEXPR bool operator!=(::yampi::color const& lhs, ::yampi::color const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs == rhs); }

  inline bool operator>=(::yampi::color const& lhs, ::yampi::color const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs < rhs); }

  inline bool operator>(::yampi::color const& lhs, ::yampi::color const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return rhs < lhs; }

  inline bool operator<=(::yampi::color const& lhs, ::yampi::color const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (rhs < lhs); }

  inline ::yampi::color operator++(::yampi::color& lhs, int)
    BOOST_NOEXCEPT_OR_NOTHROW
  { ::yampi::color result = lhs; ++lhs; return result; }

  inline ::yampi::color operator--(::yampi::color& lhs, int)
    BOOST_NOEXCEPT_OR_NOTHROW
  { ::yampi::color result = lhs; --lhs; return result; }

  template <typename Integral>
  inline ::yampi::color operator+(::yampi::color lhs, Integral const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs += rhs; return lhs; }

  template <typename Integral>
  inline ::yampi::color operator-(::yampi::color lhs, Integral const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs -= rhs; return lhs; }

  template <typename Integral>
  inline ::yampi::color operator*(::yampi::color lhs, Integral const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs *= rhs; return lhs; }

  template <typename Integral>
  inline ::yampi::color operator/(::yampi::color lhs, Integral const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs /= rhs; return lhs; }

  template <typename Integral>
  inline ::yampi::color operator+(Integral const lhs, ::yampi::color const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return rhs+lhs; }

  template <typename Integral>
  inline ::yampi::color operator*(Integral const lhs, ::yampi::color const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return rhs*lhs; }

  inline void swap(::yampi::color& lhs, ::yampi::color& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_integral
# undef YAMPI_enable_if

#endif

