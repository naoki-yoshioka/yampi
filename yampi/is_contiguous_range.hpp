#ifndef YAMPI_IS_CONTIGUOUS_RANGE_HPP
# define YAMPI_IS_CONTIGUOUS_RANGE_HPP

# include <boost/range/iterator.hpp>

# include <yampi/is_contiguous_iterator.hpp>


namespace yampi
{
  template <typename Range>
  struct is_contiguous_range
    : ::yampi::is_contiguous_iterator<typename boost::range_iterator<Range>::type>
  { };
}


#endif

