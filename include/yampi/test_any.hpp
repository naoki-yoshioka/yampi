#ifndef YAMPI_TEST_ANY_HPP
# define YAMPI_TEST_ANY_HPP

# include <type_traits>
# include <iterator>
# include <memory>

# include <mpi.h>

# include <boost/optional.hpp>

# include <yampi/request_base.hpp>
# include <yampi/status.hpp>
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  template <typename ContiguousIterator>
  inline boost::optional<std::pair< ::yampi::status, ContiguousIterator >> test_any(
    ContiguousIterator const first, ContiguousIterator const last, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_base_of< ::yampi::request_base, typename std::iterator_traits<ContiguousIterator>::value_type >::value,
      "value_type of ContiguousIterator must be derived from yampi::request_base");

    MPI_Status mpi_status;
    int index;
    int flag;
    auto const error_code
      = MPI_Testany(
          static_cast<int>(last - first), reinterpret_cast<MPI_Request*>(std::addressof(*first)),
          std::addressof(index), std::addressof(flag), std::addressof(mpi_status));

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? index != MPI_UNDEFINED
          ? boost::make_optional(std::make_pair(::yampi::status{mpi_status}, first + index))
          : boost::make_optional(std::make_pair(::yampi::status{mpi_status}, last))
        : boost::none
      : throw ::yampi::error{error_code, "yampi::test_any", environment};
  }

  template <typename ContiguousIterator>
  inline boost::optional<ContiguousIterator> test_any(
    ::yampi::ignore_status_t const,
    ContiguousIterator const first, ContiguousIterator const last, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_base_of< ::yampi::request_base, typename std::iterator_traits<ContiguousIterator>::value_type >::value,
      "value_type of ContiguousIterator must be derived from yampi::request_base");

    int index;
    int flag;
    auto const error_code
      = MPI_Testany(
          static_cast<int>(last - first), reinterpret_cast<MPI_Request*>(std::addressof(*first)),
          std::addressof(index), std::addressof(flag), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? index != MPI_UNDEFINED
          ? boost::make_optional(first + index)
          : boost::make_optional(last)
        : boost::none
      : throw ::yampi::error{error_code, "yampi::test_any", environment};
  }
}


#endif

