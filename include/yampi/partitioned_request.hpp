#ifndef YAMPI_PARTITIONED_REQUEST_HPP
# define YAMPI_PARTITIONED_REQUEST_HPP

# include <iterator>
# include <type_traits>
# include <memory>

# include <mpi.h>

# include <yampi/partition.hpp>
# include <yampi/startable_request.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# if MPI_VERSION >= 4
namespace yampi
{
  class partitioned_request
    : public ::yampi::startable_request
  {
    using base_type = ::yampi::startable_request;

   public:
    using base_type::base_type;

    partitioned_request() noexcept(std::is_nothrow_default_constructible<base_type>::value)
      : base_type{}
    { }

    partitioned_request(partitioned_request const&) = delete;
    partitioned_request& operator=(partitioned_request const&) = delete;
    partitioned_request(partitioned_request&&) = default;
    partitioned_request& operator=(partitioned_request&&) = default;
    ~partitioned_request() noexcept = default;

    void ready(::yampi::partition const partition, ::yampi::environment const& environment)
    {
      auto const error_code = MPI_Pready(partition.mpi_partition(), mpi_request_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::partitioned_request::ready", environment};
    }

    void ready(::yampi::partition const first_partition, int const num_ready_partitions, ::yampi::environment const& environment)
    {
      auto const error_code
        = MPI_Pready_range(first_partition.mpi_partition(), (first_partition + (num_ready_partitions - 1)).mpi_partition(), mpi_request_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::partitioned_request::ready", environment};
    }

    template <typename ContiguousIterator>
    void ready(ContiguousIterator const first, ContiguousIterator const last, ::yampi::environment const& environment)
    {
      static_assert(
        std::is_same< ::yampi::partition, typename std::iterator_traits<ContiguousIterator>::value_type >::value,
        "value_type of ContiguousIterator must be the same to yampi::partition");
      auto const error_code
        = MPI_Pready_list(static_cast<int>(last - first), reinterpret_cast<int*>(std::addressof(*first)), mpi_request_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::partitioned_request::ready", environment};
    }

    bool is_arrived(::yampi::partition const partition, ::yampi::environment const& environment)
    {
      int flag;
      auto const error_code = MPI_Parrived(mpi_request_, partition.mpi_partition(), std::addressof(flag));
      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
        : throw ::yampi::error{error_code, "yampi::partitioned_request::is_arrived", environment};
    }
  };
}
# endif // MPI_VERSION >= 4

#endif

