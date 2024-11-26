#ifndef YAMPI_PROBE_TEST_HPP
# define YAMPI_PROBE_TEST_HPP

# include <memory>

# include <mpi.h>

# include <boost/optional.hpp>

# include <yampi/environment.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/status.hpp>
# include <yampi/error.hpp>
# include <yampi/message.hpp>


namespace yampi
{
  inline boost::optional< ::yampi::status > probe_test(
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status mpi_status;
    int flag;
    auto const error_code
      = MPI_Iprobe(
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(flag), std::addressof(mpi_status));

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(::yampi::status{mpi_status})
        : boost::none
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline boost::optional< ::yampi::status > probe_test(
    ::yampi::rank const source, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status mpi_status;
    int flag;
    auto const error_code
      = MPI_Iprobe(
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(flag), std::addressof(mpi_status));

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(::yampi::status{mpi_status})
        : boost::none
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline boost::optional< ::yampi::status > probe_test(
    ::yampi::tag const tag, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status mpi_status;
    int flag;
    auto const error_code
      = MPI_Iprobe(
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(flag), std::addressof(mpi_status));

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(::yampi::status{mpi_status})
        : boost::none
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline boost::optional< ::yampi::status > probe_test(
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status mpi_status;
    int flag;
    auto const error_code
      = MPI_Iprobe(
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(flag), std::addressof(mpi_status));

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(::yampi::status{mpi_status})
        : boost::none
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }
# if MPI_VERSION >= 3

  inline bool probe_test(
    ::yampi::ignore_status_t const,
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    int flag;
    auto const error_code
      = MPI_Iprobe(
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(flag), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline bool probe_test(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::rank const source, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    int flag;
    auto const error_code
      = MPI_Iprobe(
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(flag), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline bool probe_test(
    ::yampi::ignore_status_t const,
    ::yampi::tag const tag, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    int flag;
    auto const error_code
      = MPI_Iprobe(
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(flag), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline bool probe_test(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    int flag;
    auto const error_code
      = MPI_Iprobe(
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(flag), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline boost::optional< std::pair< ::yampi::message, ::yampi::status > > probe_test(
    ::yampi::return_message_t const,
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    MPI_Status mpi_status;
    int flag;
    auto const error_code
      = MPI_Improbe(
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(flag), std::addressof(mpi_message), std::addressof(mpi_status));

    using result_type = typedef std::pair< ::yampi::message, ::yampi::status >;
    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(result_type{::yampi::message{mpi_message}, ::yampi::status{mpi_status}})
        : boost::none
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline boost::optional< std::pair< ::yampi::message, ::yampi::status > > probe_test(
    ::yampi::return_message_t const return_message,
    ::yampi::rank const source, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    MPI_Status mpi_status;
    int flag;
    auto const error_code
      = MPI_Improbe(
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(flag), std::addressof(mpi_message), std::addressof(mpi_status));

    using result_type = typedef std::pair< ::yampi::message, ::yampi::status >;
    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(result_type{::yampi::message{mpi_message}, ::yampi::status{mpi_status}})
        : boost::none
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline boost::optional< std::pair< ::yampi::message, ::yampi::status > > probe_test(
    ::yampi::return_message_t const,
    ::yampi::tag const tag, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    MPI_Status mpi_status;
    int flag;
    auto const error_code
      = MPI_Improbe(
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(flag), std::addressof(mpi_message), std::addressof(mpi_status));

    using result_type = typedef std::pair< ::yampi::message, ::yampi::status >;
    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(result_type{::yampi::message{mpi_message}, ::yampi::status{mpi_status}})
        : boost::none
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline boost::optional< std::pair< ::yampi::message, ::yampi::status > > probe_test(
    ::yampi::return_message_t const return_message,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    MPI_Status mpi_status;
    int flag;
    auto const error_code
      = MPI_Improbe(
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(flag), std::addressof(mpi_message), std::addressof(mpi_status));

    using result_type = typedef std::pair< ::yampi::message, ::yampi::status >;
    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(result_type{::yampi::message{mpi_message}, ::yampi::status{mpi_status}})
        : boost::none
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline boost::optional< ::yampi::message > probe_test(
    ::yampi::return_message_t const, ::yampi::ignore_status_t const,
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    int flag;
    auto const error_code
      = MPI_Improbe(
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(flag), std::addressof(mpi_message), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(::yampi::message{mpi_message})
        : boost::none
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline boost::optional< ::yampi::message > probe_test(
    ::yampi::return_message_t const return_message, ::yampi::ignore_status_t const ignore_status,
    ::yampi::rank const source, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    int flag;
    auto const error_code
      = MPI_Improbe(
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(flag), std::addressof(mpi_message), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(::yampi::message{mpi_message})
        : boost::none
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline boost::optional< ::yampi::message > probe_test(
    ::yampi::return_message_t const, ::yampi::ignore_status_t const,
    ::yampi::tag const tag, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    int flag;
    auto const error_code
      = MPI_Improbe(
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(flag), std::addressof(mpi_message), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(::yampi::message{mpi_message})
        : boost::none
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }

  inline boost::optional< ::yampi::message > probe_test(
    ::yampi::return_message_t const return_message, ::yampi::ignore_status_t const ignore_status,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    int flag;
    auto const error_code
      = MPI_Improbe(
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(flag), std::addressof(mpi_message), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(::yampi::message{mpi_message})
        : boost::none
      : throw ::yampi::error{error_code, "yampi::probe_test", environment};
  }
# endif
}


#endif

