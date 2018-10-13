#ifndef YAMPI_BOUNDS_HPP
# define YAMPI_BOUNDS_HPP

# include <boost/config.hpp>

# include <utility>

# include <yampi/address.hpp>
# include <yampi/utility/is_nothrow_swappable.hpp>


namespace yampi
{
  class bounds
  {
    ::yampi::address lower_bound_;
    ::yampi::address extent_;

   public:
    bounds(::yampi::address const lower_bound, ::yampi::address const extent)
      : lower_bound_(lower_bound), extent_(extent)
    { }

    bool operator==(::yampi::bounds const& other) const
    { return lower_bound_ == other.lower_bound_ and extent_ == other.extent_; }

    ::yampi::address const& lower_bound() const { return lower_bound_; }
    ::yampi::address const& upper_bound() const { return lower_bound_ + extent_; }
    ::yampi::address const& extent() const { return extent_; }

    void swap(::yampi::bounds& other)
      BOOST_NOEXCEPT_IF(
        ::yampi::utility::is_nothrow_swappable< ::yampi::address >::value)
    {
      using std::swap;
      swap(lower_bound_, other.lower_bound_);
      swap(extent_, other.extent_);
    }
  };

  inline bool operator!=(::yampi::bounds const& lhs, ::yampi::bounds const& rhs)
  { return not (lhs == rhs); }

  inline void swap(::yampi::bounds& lhs, ::yampi::bounds& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


#endif

