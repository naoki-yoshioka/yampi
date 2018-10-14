#ifndef YAMPI_BOUNDS_HPP
# define YAMPI_BOUNDS_HPP

# include <boost/config.hpp>

# include <utility>

# include <yampi/utility/is_nothrow_swappable.hpp>


namespace yampi
{
  template <typename Count>
  class bounds
  {
    Count lower_bound_;
    Count extent_;

   public:
    typedef Count count_type;

    bounds(Count const lower_bound, Count const extent)
      : lower_bound_(lower_bound), extent_(extent)
    { }

    bool operator==(bounds const& other) const
    { return lower_bound_ == other.lower_bound_ and extent_ == other.extent_; }

    Count const& lower_bound() const { return lower_bound_; }
    void lower_bound(Count const lb) { lower_bound_ = lb; }

    Count const& upper_bound() const { return lower_bound_ + extent_; }
    void upper_bound(Count const ub) { extent_ = ub - lower_bound_; }

    Count const& extent() const { return extent_; }
    void extent(Count const ex) { extent_ = ex; }

    void swap(bounds& other)
      BOOST_NOEXCEPT_IF(::yampi::utility::is_nothrow_swappable<Count>::value)
    {
      using std::swap;
      swap(lower_bound_, other.lower_bound_);
      swap(extent_, other.extent_);
    }
  };

  template <typename Count>
  inline bool operator!=(
    ::yampi::bounds<Count> const& lhs, ::yampi::bounds<Count> const& rhs)
  { return not (lhs == rhs); }

  template <typename Count>
  inline void swap(::yampi::bounds<Count>& lhs, ::yampi::bounds<Count>& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  template <typename Count>
  inline ::yampi::bounds<Count> make_bounds(
    Count const lower_bound, Count const extent)
  { return ::yampi::bounds<Count>(lower_bound, extent); }
}


#endif

