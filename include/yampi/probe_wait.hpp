#ifndef YAMPI_PROBE_WAIT_HPP
# define YAMPI_PROBE_WAIT_HPP

# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/status.hpp>
# include <yampi/error.hpp>
# include <yampi/message.hpp>


namespace yampi
{
  inline ::yampi::status probe_wait(
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status mpi_status;
    auto const error_code
      = MPI_Probe(source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), std::addressof(mpi_status));

    return error_code == MPI_SUCCESS
      ? ::yampi::status{mpi_status}
      : throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline ::yampi::status probe_wait(
    ::yampi::rank const source,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status mpi_status;
    auto const error_code
      = MPI_Probe(source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(), std::addressof(mpi_status));

    return error_code == MPI_SUCCESS
      ? ::yampi::status{mpi_status}
      : throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline ::yampi::status probe_wait(
    ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status mpi_status;
    auto const error_code
      = MPI_Probe(MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(), std::addressof(mpi_status));

    return error_code == MPI_SUCCESS
      ? ::yampi::status{mpi_status}
      : throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline ::yampi::status probe_wait(::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status mpi_status;
    auto const error_code
      = MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(), std::addressof(mpi_status));

    return error_code == MPI_SUCCESS
      ? ::yampi::status{mpi_status}
      : throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }
# if MPI_VERSION >= 3

  inline void probe_wait(
    ::yampi::ignore_status_t const,
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    auto const error_code
      = MPI_Probe(source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);

    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline void probe_wait(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::rank const source, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    auto const error_code
      = MPI_Probe(source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(), MPI_STATUS_IGNORE);

    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline void probe_wait(
    ::yampi::ignore_status_t const,
    ::yampi::tag const tag, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    auto const error_code
      = MPI_Probe(MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);

    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline void probe_wait(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    auto const error_code
      = MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(), MPI_STATUS_IGNORE);

    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline std::pair< ::yampi::message, ::yampi::status > probe_wait(
    ::yampi::return_message_t const,
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    MPI_Status mpi_status;
    auto const error_code
      = MPI_Mprobe(
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_message), std::addressof(mpi_status));

    using result_type = std::pair< ::yampi::message, ::yampi::status >;
    return error_code == MPI_SUCCESS
      ? result_type{::yampi::message{mpi_message}, ::yampi::status{mpi_status}}
      : throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline std::pair< ::yampi::message, ::yampi::status > probe_wait(
    ::yampi::return_message_t const return_message,
    ::yampi::rank const source, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    MPI_Status mpi_status;
    auto const error_code
      = MPI_Mprobe(
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(mpi_message), std::addressof(mpi_status));

    using result_type = std::pair< ::yampi::message, ::yampi::status >;
    return error_code == MPI_SUCCESS
      ? result_type{::yampi::message{mpi_message}, ::yampi::status{mpi_status}}
      : throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline std::pair< ::yampi::message, ::yampi::status > probe_wait(
    ::yampi::return_message_t const,
    ::yampi::tag const tag, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    MPI_Status mpi_status;
    auto const error_code
      = MPI_Mprobe(
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_message), std::addressof(mpi_status));

    using result_type = std::pair< ::yampi::message, ::yampi::status >;
    return error_code == MPI_SUCCESS
      ? result_type{::yampi::message{mpi_message}, ::yampi::status{mpi_status}}
      : throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline std::pair< ::yampi::message, ::yampi::status > probe_wait(
    ::yampi::return_message_t const return_message,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    MPI_Status mpi_status;
    auto const error_code
      = MPI_Mprobe(
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(mpi_message), std::addressof(mpi_status));

    using result_type = std::pair< ::yampi::message, ::yampi::status >;
    return error_code == MPI_SUCCESS
      ? result_type{::yampi::message{mpi_message}, ::yampi::status{mpi_status}}
      : throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline ::yampi::message probe_wait(
    ::yampi::return_message_t const, ::yampi::ignore_status_t const,
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    auto const error_code
      = MPI_Mprobe(
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_message), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? ::yampi::message{mpi_message}
      : throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline ::yampi::message probe_wait(
    ::yampi::return_message_t const return_message, ::yampi::ignore_status_t const ignore_status,
    ::yampi::rank const source, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    auto const error_code
      = MPI_Mprobe(
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(mpi_message), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? ::yampi::message{mpi_message}
      : throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline ::yampi::message probe_wait(
    ::yampi::return_message_t const, ::yampi::ignore_status_t const,
    ::yampi::tag const tag, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    auto const error_code
      = MPI_Mprobe(
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_message), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? ::yampi::message{mpi_message}
      : throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }

  inline ::yampi::message probe_wait(
    ::yampi::return_message_t const return_message, ::yampi::ignore_status_t const ignore_status,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    auto const error_code
      = MPI_Mprobe(
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(mpi_message), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? ::yampi::message{mpi_message}
      : throw ::yampi::error{error_code, "yampi::probe_wait", environment};
  }
# endif
}


#endif

