#ifndef YAMPI_WAIT_ANY_HPP
# define YAMPI_WAIT_ANY_HPP

# include <type_traits>
# include <iterator>
# include <memory>

# include <mpi.h>

# include <yampi/request_base.hpp>
# include <yampi/status.hpp>
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  template <typename ContiguousIterator>
  inline std::pair< ::yampi::status, ContiguousIterator > wait_any(
    ContiguousIterator const first, ContiguousIterator const last, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_base_of< ::yampi::request_base, typename std::iterator_traits<ContiguousIterator>::value_type >::value,
      "value_type of ContiguousIterator must be derived from yampi::request_base");

    MPI_Status mpi_status;
    int index;
    auto const error_code
      = MPI_Waitany(
          static_cast<int>(last - first), reinterpret_cast<MPI_Request*>(std::addressof(*first)),
          std::addressof(index), std::addressof(mpi_status));

    return error_code == MPI_SUCCESS
      ? index != MPI_UNDEFINED
        ? std::make_pair(::yampi::status{mpi_status}, first + index)
        : std::make_pair(::yampi::status{mpi_status}, last)
      : throw ::yampi::error{error_code, "yampi::wait_any", environment};
  }

  template <typename ContiguousIterator>
  inline ContiguousIterator wait_any(
    ::yampi::ignore_status_t const,
    ContiguousIterator const first, ContiguousIterator const last, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_base_of< ::yampi::request_base, typename std::iterator_traits<ContiguousIterator>::value_type >::value,
      "value_type of ContiguousIterator must be derived from yampi::request_base");

    int index;
    auto const error_code
      = MPI_Waitany(
          static_cast<int>(last - first), reinterpret_cast<MPI_Request*>(std::addressof(*first)),
          std::addressof(index), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? index != MPI_UNDEFINED
        ? first + index
        : last
      : throw ::yampi::error{error_code, "yampi::wait_any", environment};
  }
}


#endif

