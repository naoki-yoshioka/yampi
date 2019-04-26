#ifndef YAMPI_BOUNDS_HPP
# define YAMPI_BOUNDS_HPP

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
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   define YAMPI_is_nothrow_copy_assignable std::is_nothrow_copy_assignable
# else
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   define YAMPI_is_nothrow_copy_assignable boost::has_nothrow_assign
# endif

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  template <typename Count>
  class bounds
  {
    Count lower_bound_;
    Count extent_;

   public:
    typedef Count count_type;

    bounds(Count const& lower_bound, Count const& extent)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<Count>::value)
      : lower_bound_(lower_bound), extent_(extent)
    { }

    bool operator==(bounds const& other) const
      BOOST_NOEXCEPT_OR_NOTHROW/*BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lower_bound_ == other.lower_bound_))*/
    { return lower_bound_ == other.lower_bound_ and extent_ == other.extent_; }

    Count const& lower_bound() const BOOST_NOEXCEPT_OR_NOTHROW { return lower_bound_; }
    void lower_bound(Count const& lb)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_assignable<Count>::value)
    { lower_bound_ = lb; }

    Count const& upper_bound() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lower_bound_ + extent_))
    { return lower_bound_ + extent_; }
    void upper_bound(Count const& ub)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_copy_assignable<Count>::value
        and BOOST_NOEXCEPT_EXPR(ub - lower_bound_))
    { extent_ = ub - lower_bound_; }

    Count const& extent() const BOOST_NOEXCEPT_OR_NOTHROW { return extent_; }
    void extent(Count const& ex)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_assignable<Count>::value)
    { extent_ = ex; }

    void swap(bounds& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<Count>::value)
    {
      using std::swap;
      swap(lower_bound_, other.lower_bound_);
      swap(extent_, other.extent_);
    }
  };

  template <typename Count>
  inline bool operator!=(
    ::yampi::bounds<Count> const& lhs, ::yampi::bounds<Count> const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs == rhs))
  { return not (lhs == rhs); }

  template <typename Count>
  inline void swap(::yampi::bounds<Count>& lhs, ::yampi::bounds<Count>& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  template <typename Count>
  inline ::yampi::bounds<Count> make_bounds(
    Count const& lower_bound, Count const& extent)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(::yampi::bounds<Count>(lower_bound, extent)))
  { return ::yampi::bounds<Count>(lower_bound, extent); }
}


# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible

#endif

