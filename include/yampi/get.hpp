#ifndef YAMPI_GET_HPP
# define YAMPI_GET_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/window.hpp>
# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>
# include <yampi/datatype.hpp>


namespace yampi
{
  template <typename Value, typename ContiguousIterator>
  inline void get(
    ::yampi::buffer<Value> const& origin_buffer,
    ::yampi::rank const target, int const target_index,
    ::yampi::datatype const& target_datatype, int const target_count,
    ::yampi::window<Value> const& window,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Put(
          origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), static_cast<MPI_Aint>(target_index),
          target_count, target_datatype.mpi_datatype(),
          window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::get", environment);
  }

  template <typename Value, typename ContiguousIterator>
  inline void get(
    ::yampi::buffer<Value> const& origin_buffer,
    ::yampi::rank const target, int const target_index,
    ::yampi::datatype const& target_datatype,
    ::yampi::window<Value> const& window,
    ::yampi::environment const& environment)
  {
    ::yampi::get(
      origin_buffer, target, target_index, target_datatype, origin_buffer.count(),
      window, environment);
  }

  template <typename Value, typename ContiguousIterator>
  inline void get(
    ::yampi::buffer<Value> const& origin_buffer,
    ::yampi::rank const target, int const target_index,
    ::yampi::window<Value> const& window,
    ::yampi::environment const& environment)
  {
    ::yampi::get(
      origin_buffer, target, target_index, origin_buffer.datatype(), origin_buffer.count(),
      window, environment);
  }
}


#endif

