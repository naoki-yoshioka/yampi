#ifndef YAMPI_WINDOW_HPP
# define YAMPI_WINDOW_HPP

# include <cassert>
# include <cstddef>
# include <iterator>
# include <utility>
# include <type_traits>
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/window_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/addressof.hpp>
# include <yampi/information.hpp>


namespace yampi
{
  class window
    : public ::yampi::window_base< ::yampi::window >
  {
    typedef ::yampi::window_base< ::yampi::window > base_type;

    MPI_Win mpi_win_;

   public:
    window() = default;
    window(window const&) = delete;
    window& operator=(window const&) = delete;
    window(window&&) = default;
    window& operator=(window&&) = default;
    ~window() noexcept = default;

    template <typename ContiguousIterator>
    window(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
      : base_type{create(first, last, MPI_INFO_NULL, communicator, environment)}
    { assert(last >= first); }

    template <typename ContiguousIterator>
    window(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::information const& information,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
      : base_type{create(first, last, information.mpi_info(), communicator, environment)}
    { assert(last >= first); }

   private:
    template <typename ContiguousIterator>
    MPI_Win create(
      ContiguousIterator const first, ContiguousIterator const last,
      MPI_Info const& mpi_info, ::yampi::communicator const& communicator,
      ::yampi::environment const& environment) const
    {
      assert(last >= first);

      using value_type = typename std::remove_cv<typename std::iterator_traits<ContiguousIterator>::value_type>::type;
      MPI_Win result;
      int const error_code
        = MPI_Win_create(
            std::addressof(*first),
            (::yampi::addressof(*last, environment) - ::yampi::addressof(*first, environment)).mpi_byte_displacement(),
            sizeof(value_type), mpi_info, communicator.mpi_comm(),
            std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::window::create", environment);
    }

   private:
    friend base_type;

    void do_reset(window&& other, ::yampi::environment const& environment) noexcept { }
    void do_swap(window& other) noexcept { }

   public:
    template <typename ContiguousIterator>
    void reset(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = create(first, last, MPI_INFO_NULL, communicator, environment);
    }

    template <typename ContiguousIterator>
    void reset(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::information const& information,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = create(first, last, information.mpi_info(), communicator, environment);
    }
  };
}


#endif

