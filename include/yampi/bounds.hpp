#ifndef YAMPI_BOUNDS_HPP
# define YAMPI_BOUNDS_HPP

# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# include <yampi/extent.hpp>

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
      noexcept(std::is_nothrow_copy_constructible< ::yampi::extent >::value)
      : lower_bound_{lower_bound}, extent_{extent}
    { }

    bool operator==(bounds const& other) const
      noexcept(noexcept(lower_bound_ == other.lower_bound_))
    { return lower_bound_ == other.lower_bound_ and extent_ == other.extent_; }

    ::yampi::extent const& lower_bound() const noexcept { return lower_bound_; }
    void lower_bound(::yampi::extent const& lb)
      noexcept(std::is_nothrow_copy_assignable< ::yampi::extent >::value)
    { lower_bound_ = lb; }

    ::yampi::extent upper_bound() const
      noexcept(noexcept(lower_bound_ + extent_))
    { return lower_bound_ + extent_; }
    void upper_bound(::yampi::extent const& ub)
      noexcept(std::is_nothrow_copy_assignable< ::yampi::extent >::value and noexcept(ub - lower_bound_))
    { extent_ = ub - lower_bound_; }

    ::yampi::extent const& extent() const noexcept { return extent_; }
    void extent(::yampi::extent const& ex)
      noexcept(std::is_nothrow_copy_assignable< ::yampi::extent >::value)
    { extent_ = ex; }

    void swap(bounds& other)
      noexcept(YAMPI_is_nothrow_swappable< ::yampi::extent >::value)
    {
      using std::swap;
      swap(lower_bound_, other.lower_bound_);
      swap(extent_, other.extent_);
    }
  };

  inline bool operator!=(::yampi::bounds const& lhs, ::yampi::bounds const& rhs) noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(::yampi::bounds& lhs, ::yampi::bounds& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif

