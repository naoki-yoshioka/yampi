#ifndef YAMPI_INTERCOMMUNICATOR_HPP
# define YAMPI_INTERCOMMUNICATOR_HPP

# include <utility>
# include <type_traits>
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>
# include <yampi/group.hpp>
# include <yampi/information.hpp>
# include <yampi/color.hpp>
# include <yampi/split_type.hpp>


namespace yampi
{
  class intercommunicator
    : public ::yampi::communicator_base
  {
    typedef ::yampi::communicator_base base_type;

   public:
    intercommunicator() noexcept(std::is_nothrow_default_constructible<base_type>::value)
      : base_type()
    { }

    intercommunicator(intercommunicator const&) = delete;
    intercommunicator& operator=(intercommunicator const&) = delete;
    intercommunicator(intercommunicator&&) = default;
    intercommunicator& operator=(intercommunicator&&) = default;
    ~intercommunicator() noexcept = default;

    using base_type::base_type;

    intercommunicator(
      ::yampi::communicator const& local_communicator, ::yampi::rank const local_leader,
      ::yampi::communicator const& peer_communicator, ::yampi::rank const remote_leader,
      ::yampi::tag const tag, ::yampi::environment const& environment)
      : base_type(create(local_communicator, local_leader, peer_communicator, remote_leader, tag, environment))
    { }

   private:
    MPI_Comm create(
      ::yampi::communicator const& local_communicator, ::yampi::rank const local_leader,
      ::yampi::communicator const& peer_communicator, ::yampi::rank const remote_leader,
      ::yampi::tag const tag, ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Intercomm_create(
            local_communicator.mpi_comm(), local_leader.mpi_rank(),
            peer_communicator.mpi_comm(), remote_leader.mpi_rank(),
            tag.mpi_tag(), std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::intercommunicator::create", environment);
    }

   public:
    using base_type::reset;

    void reset(
      ::yampi::communicator const& local_communicator, ::yampi::rank const local_leader,
      ::yampi::communicator const& peer_communicator, ::yampi::rank const remote_leader,
      ::yampi::tag const tag, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_comm_ = create(local_communicator, local_leader, peer_communicator, remote_leader, tag, environment);
    }

    void merge(::yampi::communicator& communicator, ::yampi::environment const& environment) const
    { merge(communicator, true, environment); }

    void merge(
      ::yampi::communicator& communicator, bool const is_higher_rank_preferred,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Intercomm_merge(mpi_comm_, static_cast<int>(is_higher_rank_preferred), std::addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::intercommunicator::merge", environment);

      communicator.reset(result, environment);
    }

    int remote_size(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Comm_remote_size(mpi_comm_, std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::intercommunicator::remote_size", environment);
    }

    void remote_group(::yampi::group& group, ::yampi::environment const& environment) const
    {
      MPI_Group mpi_group;
      int const error_code = MPI_Comm_remote_group(mpi_comm_, std::addressof(mpi_group));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::intercommunicator::remote_group", environment);

      group.reset(mpi_group, environment);
    }
  };

  inline void swap(::yampi::intercommunicator& lhs, ::yampi::intercommunicator& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


#endif

