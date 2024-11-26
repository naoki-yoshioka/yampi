#ifndef YAMPI_STATUS_SOME_HPP
# define YAMPI_STATUS_SOME_HPP

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
  template <typename ContiguousIterator1, typename ContiguousIterator2, typename ContiguousIterator3>
  inline boost::optional<std::pair<ContiguousIterator2, ContiguousIterator3>> status_some(
    ContiguousIterator1 const first, ContiguousIterator1 const last,
    ContiguousIterator2 const index_out, ContiguousIterator3 const status_out, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_base_of< ::yampi::request_base, typename std::iterator_traits<ContiguousIterator1>::value_type >::value,
      "value_type of ContiguousIterator1 must be derived from yampi::request_base");
    static_assert(
      std::is_same<int, typename std::iterator_traits<ContiguousIterator2>::value_type>::value,
      "value_type of ContiguousIterator2 must be the same to int");
    static_assert(
      std::is_same< ::yampi::status, typename std::iterator_traits<ContiguousIterator3>::value_type >::value,
      "value_type of ContiguousIterator3 must be the same to yampi::status");

    int count;
    auto const error_code
      = MPI_Testsome(
          static_cast<int>(last - first), reinterpret_cast<MPI_Request const*>(std::addressof(*first)),
          std::addressof(count), std::addressof(*index_out), reinterpret_cast<MPI_Status*>(std::addressof(*status_out)));

    return error_code == MPI_SUCCESS
      ? count != MPI_UNDEFINED
        ? boost::make_optional(std::make_pair(index_out + count, status_out + count))
        : boost::none
      : throw ::yampi::error{error_code, "yampi::status_some", environment};
  }

  template <typename ContiguousIterator1, typename ContiguousIterator2>
  inline boost::optional<ContiguousIterator2> status_some(
    ::yampi::ignore_status_t const,
    ContiguousIterator1 const first, ContiguousIterator1 const last,
    ContiguousIterator2 const out, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_base_of< ::yampi::request_base, typename std::iterator_traits<ContiguousIterator1>::value_type >::value,
      "value_type of ContiguousIterator1 must be derived from yampi::request_base");
    static_assert(
      std::is_same<int, typename std::iterator_traits<ContiguousIterator2>::value_type>::value,
      "value_type of ContiguousIterator2 must be the same to int");

    int count;
    auto const error_code
      = MPI_Testsome(
          static_cast<int>(last - first), reinterpret_cast<MPI_Request const*>(std::addressof(*first)),
          std::addressof(count), std::addressof(*out), MPI_STATUSES_IGNORE);

    return error_code == MPI_SUCCESS
      ? count != MPI_UNDEFINED
        ? boost::make_optional(out + count)
        : boost::none
      : throw ::yampi::error{error_code, "yampi::status_some", environment};
  }
}
# endif // MPI_VERSION >= 4 and MPI_SUBVERSION >= 1


#endif

