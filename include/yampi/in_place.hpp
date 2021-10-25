#ifndef YAMPI_IN_PLACE_HPP
# define YAMPI_IN_PLACE_HPP

# include <boost/config.hpp>

# include <mpi.h>


namespace yampi
{
  struct in_place_t { };

  namespace tags
  {
    inline BOOST_CONSTEXPR ::yampi::in_place_t in_place() { return ::yampi::in_place_t(); }
  }
}


#endif

