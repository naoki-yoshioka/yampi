#ifndef YAMPI_DYNAMIC_WINDOW_HPP
# define YAMPI_DYNAMIC_WINDOW_HPP

# include <utility>
# include <type_traits>
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/window_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/information.hpp>


# if MPI_VERSION >= 3
namespace yampi
{
  class dynamic_window
    : public ::yampi::window_base< ::yampi::dynamic_window >
  {
    typedef ::yampi::window_base< ::yampi::dynamic_window > base_type;

    MPI_Win mpi_win_;

   public:
    dynamic_window() = default;
    dynamic_window(dynamic_window const&) = delete;
    dynamic_window& operator=(dynamic_window const&) = delete;
    dynamic_window(dynamic_window&&) = default;
    dynamic_window& operator=(dynamic_window&&) = default;
    ~dynamic_window() noexcept = default;

    dynamic_window(::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{create(MPI_INFO_NULL, communicator, environment)}
    { }

    dynamic_window(
      ::yampi::information const& information,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{create(information.mpi_info(), communicator, environment)}
    { }

   private:
    MPI_Win create(
      MPI_Info const& mpi_info,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Win result;
      int const error_code
        = MPI_Win_create_dynamic(mpi_info, communicator.mpi_comm(), std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::dynamic_window::create", environment);
    }

   private:
    friend base_type;

    void do_reset(dynamic_window&& other, ::yampi::environment const& environment) { }
    void do_swap(window& other) noexcept { }
  };
}
# endif // MPI_VERSION >= 3


#endif

