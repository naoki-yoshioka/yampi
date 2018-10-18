#ifndef YAMPI_TEST_FOR_PROBE_HPP
# define YAMPI_TEST_FOR_PROBE_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <boost/optional.hpp>

# include <yampi/environment.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/status.hpp>
# include <yampi/error.hpp>
# include <yampi/message.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  inline boost::optional< ::yampi::status > test_for_probe(
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status mpi_status;
    int flag;
    int const error_code
      = MPI_Iprobe(
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          YAMPI_addressof(flag), YAMPI_addressof(mpi_status));

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(::yampi::status(mpi_status))
        : boost::none
      : throw ::yampi::error(error_code, "yampi::test_for_probe", environment);
  }

  inline boost::optional< ::yampi::status > test_for_probe(
    ::yampi::rank const source,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  { return ::yampi::test_for_probe(source, ::yampi::any_tag(), communicator, environment); }

  inline boost::optional< ::yampi::status > test_for_probe(
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  { return ::yampi::test_for_probe(::yampi::any_source(), ::yampi::any_tag(), communicator, environment); }
# if MPI_VERSION >= 3

  inline bool test_for_probe(
    ::yampi::ignore_status_t const,
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    int flag;
    int const error_code
      = MPI_Iprobe(
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          YAMPI_addressof(flag), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
      : throw ::yampi::error(error_code, "yampi::test_for_probe", environment);
  }

  inline bool test_for_probe(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::rank const source,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  { return ::yampi::test_for_probe(ignore_status, source, ::yampi::any_tag(), communicator, environment); }

  inline bool test_for_probe(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  { return ::yampi::test_for_probe(ignore_status, ::yampi::any_source(), ::yampi::any_tag(), communicator, environment); }

  inline boost::optional< std::pair< ::yampi::message, ::yampi::status > > test_for_probe(
    ::yampi::return_message_t const,
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    MPI_Status mpi_status;
    int flag;
    int const error_code
      = MPI_Improbe(
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          YAMPI_addressof(flag), YAMPI_addressof(mpi_message), YAMPI_addressof(mpi_status));

    typedef std::pair< ::yampi::message, ::yampi::status > result_type;
    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(result_type(::yampi::message(mpi_message), ::yampi::status(mpi_status)))
        : boost::none
      : throw ::yampi::error(error_code, "yampi::test_for_probe", environment);
  }

  inline boost::optional< std::pair< ::yampi::message, ::yampi::status > > test_for_probe(
    ::yampi::return_message_t const return_message,
    ::yampi::rank const source,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  { return ::yampi::test_for_probe(return_message, source, ::yampi::any_tag(), communicator, environment); }

  inline boost::optional< std::pair< ::yampi::message, ::yampi::status > > test_for_probe(
    ::yampi::return_message_t const return_message,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  { return ::yampi::test_for_probe(return_message, ::yampi::any_source(), ::yampi::any_tag(), communicator, environment); }

  inline boost::optional< ::yampi::message > test_for_probe(
    ::yampi::return_message_t const, ::yampi::ignore_status_t const,
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Message mpi_message;
    int flag;
    int const error_code
      = MPI_Improbe(
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          YAMPI_addressof(flag), YAMPI_addressof(mpi_message), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? boost::make_optional(::yampi::message(mpi_message))
        : boost::none
      : throw ::yampi::error(error_code, "yampi::test_for_probe", environment);
  }

  inline boost::optional< ::yampi::message > test_for_probe(
    ::yampi::return_message_t const return_message, ::yampi::ignore_status_t const ignore_status,
    ::yampi::rank const source,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  { return ::yampi::test_for_probe(return_message, ignore_status, source, ::yampi::any_tag(), communicator, environment); }

  inline boost::optional< ::yampi::message > test_for_probe(
    ::yampi::return_message_t const return_message, ::yampi::ignore_status_t const ignore_status,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  { return ::yampi::test_for_probe(return_message, ignore_status, ::yampi::any_source(), ::yampi::any_tag(), communicator, environment); }
# endif
}


# undef YAMPI_addressof

#endif

