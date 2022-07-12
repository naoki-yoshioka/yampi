#ifndef YAMPI_IN_PLACE_HPP
# define YAMPI_IN_PLACE_HPP

# include <mpi.h>


namespace yampi
{
  struct in_place_t { };

# if __cplusplus >= 201703L
  inline constexpr ::yampi::in_place_t in_place{};
# else
  constexpr ::yampi::in_place_t in_place{};
# endif
}


#endif

