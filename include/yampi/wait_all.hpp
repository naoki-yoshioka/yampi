#ifndef YAMPI_WAIT_ALL_HPP
# define YAMPI_WAIT_ALL_HPP

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
  template <typename ContiguousIterator1, typename ContiguousIterator2>
  inline ContiguousIterator2 wait_all(
    ContiguousIterator1 const first, ContiguousIterator1 const last,
    ContiguousIterator2 const out, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_base_of< ::yampi::request_base, typename std::iterator_traits<ContiguousIterator1>::value_type >::value,
      "value_type of ContiguousIterator1 must be derived from yampi::request_base");
    static_assert(
      std::is_same< ::yampi::status, typename std::iterator_traits<ContiguousIterator2>::value_type >::value,
      "value_type of ContiguousIterator2 must be the same to yampi::status");

    auto const error_code
      = MPI_Waitall(
          static_cast<int>(last - first), reinterpret_cast<MPI_Request*>(std::addressof(*first)),
          reinterpret_cast<MPI_Status*>(std::addressof(*out)));

    return error_code == MPI_SUCCESS
      ? out + static_cast<int>(last - first)
      : throw ::yampi::error{error_code, "yampi::wait_all", environment};
  }

  template <typename ContiguousIterator>
  inline void wait_all(
    ::yampi::ignore_status_t const,
    ContiguousIterator const first, ContiguousIterator const last, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_base_of< ::yampi::request_base, typename std::iterator_traits<ContiguousIterator>::value_type >::value,
      "value_type of ContiguousIterator must be derived from yampi::request_base");

    auto const error_code
      = MPI_Waitall(static_cast<int>(last - first), reinterpret_cast<MPI_Request*>(std::addressof(*first)), MPI_STATUSES_IGNORE);

    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::wait_all", environment};
  }
}


#endif

