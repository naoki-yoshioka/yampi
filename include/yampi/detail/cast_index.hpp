#ifndef YAMPI_DETAIL_CAST_INDEX_HPP
# define YAMPI_DETAIL_CAST_INDEX_HPP

# include <boost/config.hpp>


namespace yampi
{
  namespace detail
  {
    struct cast_index
    {
      std::size_t operator()(int const index) const
      { return static_cast<std::size_t>(index); }
    };
  }
}


#endif

