#ifndef YAMPI_START_ALL_HPP
# define YAMPI_START_ALL_HPP

# include <type_traits>
# include <iterator>
# include <memory>

# include <mpi.h>

# include <yampi/startable_request.hpp>
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  template <typename ContiguousIterator>
  inline void start_all(
    ContiguousIterator const first, ContiguousIterator const last, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_base_of< ::yampi::startable_request, typename std::iterator_traits<ContiguousIterator>::value_type >::value,
      "value_type of ContiguousIterator must be derived from yampi::startable_request");

    auto const error_code
      = MPI_Startall(static_cast<int>(last - first), reinterpret_cast<MPI_Request*>(std::addressof(*first)));

    if (error_code != MPI_SUCCESS
      throw ::yampi::error{error_code, "yampi::start_all", environment};
  }
}


#endif

