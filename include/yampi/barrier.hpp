#ifndef YAMPI_BARRIER_HPP
# define YAMPI_BARRIER_HPP

# include <mpi.h>

# include <yampi/communicator_base.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/immediate_request.hpp>
# if MPI_VERSION >= 4
#   include <yampi/persistent_request.hpp>
#   include <yampi/information.hpp>
# endif // MPI_VERSION >= 4


namespace yampi
{
  // Blocking barrier
  inline void barrier(::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    auto const error_code = MPI_Barrier(communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::barrier", environment};
  }

  // Nonblocking barrier
  inline void barrier(
    ::yampi::immediate_request& request, ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code = MPI_Ibarrier(communicator.mpi_comm(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::barrier", environment};
    request.reset(mpi_request, environment);
  }
# if MPI_VERSION >= 4

  // Persistent barrier
  inline void barrier(
    ::yampi::persistent_request& request, ::yampi::information const& information,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code = MPI_Barrier_init(communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::barrier", environment};
    request.reset(mpi_request, environment);
  }

  // information omitted
  inline void barrier(
    ::yampi::persistent_request& request,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code = MPI_Barrier_init(communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::barrier", environment};
    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 4
}


#endif

