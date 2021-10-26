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

# include <yampi/extent.hpp>

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
  class bounds
  {
    ::yampi::extent lower_bound_;
    ::yampi::extent extent_;

   public:
    bounds(::yampi::extent const& lower_bound, ::yampi::extent const& extent)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible< ::yampi::extent >::value)
      : lower_bound_(lower_bound), extent_(extent)
    { }

    bool operator==(bounds const& other) const
      BOOST_NOEXCEPT_OR_NOTHROW/*BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lower_bound_ == other.lower_bound_))*/
    { return lower_bound_ == other.lower_bound_ and extent_ == other.extent_; }

    ::yampi::extent const& lower_bound() const BOOST_NOEXCEPT_OR_NOTHROW { return lower_bound_; }
    void lower_bound(::yampi::extent const& lb)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_assignable< ::yampi::extent >::value)
    { lower_bound_ = lb; }

    ::yampi::extent upper_bound() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lower_bound_ + extent_))
    { return lower_bound_ + extent_; }
    void upper_bound(::yampi::extent const& ub)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_copy_assignable< ::yampi::extent >::value
        and BOOST_NOEXCEPT_EXPR(ub - lower_bound_))
    { extent_ = ub - lower_bound_; }

    ::yampi::extent const& extent() const BOOST_NOEXCEPT_OR_NOTHROW { return extent_; }
    void extent(::yampi::extent const& ex)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_assignable< ::yampi::extent >::value)
    { extent_ = ex; }

    void swap(bounds& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable< ::yampi::extent >::value)
    {
      using std::swap;
      swap(lower_bound_, other.lower_bound_);
      swap(extent_, other.extent_);
    }
  };

  inline bool operator!=(
    ::yampi::bounds const& lhs, ::yampi::bounds const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(::yampi::bounds& lhs, ::yampi::bounds& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible

#endif

