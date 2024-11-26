#ifndef YAMPI_STATUS_ALL_HPP
# define YAMPI_STATUS_ALL_HPP

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


# if MPI_VERSION >= 4 and MPI_SUBVERSION >= 1
namespace yampi
{
  template <typename ContiguousIterator1, typename ContiguousIterator2>
  inline boost::optional<ContiguousIterator2> status_all(
    ContiguousIterator1 const first, ContiguousIterator1 const last,
    ContiguousIterator2 const out, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_base_of< ::yampi::request_base, typename std::iterator_traits<ContiguousIterator1>::value_type >::value,
      "value_type of ContiguousIterator1 must be derived from yampi::request_base");
    static_assert(
      std::is_same< ::yampi::status, typename std::iterator_traits<ContiguousIterator2>::value_type >::value,
      "value_type of ContiguousIterator2 must be the same to yampi::status");

    int flag;
    auto const error_code
      = MPI_Testall(
          static_cast<int>(last - first), reinterpret_cast<MPI_Request const*>(std::addressof(*first)),
          std::addressof(flag), reinterpret_cast<MPI_Status*>(std::addressof(*out)));

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(out + static_cast<int>(last - first))
        : boost::none
      : throw ::yampi::error{error_code, "yampi::status_all", environment};
  }

  template <typename ContiguousIterator>
  inline bool status_all(
    ::yampi::ignore_status_t const,
    ContiguousIterator const first, ContiguousIterator const last, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_base_of< ::yampi::request_base, typename std::iterator_traits<ContiguousIterator>::value_type >::value,
      "value_type of ContiguousIterator must be derived from yampi::request_base");

    int flag;
    auto const error_code
      = MPI_Testall(
          static_cast<int>(last - first), reinterpret_cast<MPI_Request const*>(std::addressof(*first)),
          std::addressof(flag), MPI_STATUSES_IGNORE);

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
      : throw ::yampi::error{error_code, "yampi::status_all", environment};
  }
}
# endif // MPI_VERSION >= 4 and MPI_SUBVERSION >= 1


#endif

