#ifndef YAMPI_ALGORITHM_SWAP_HPP
# define YAMPI_ALGORITHM_SWAP_HPP

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/send_receive.hpp>
# include <yampi/status.hpp>
# include <yampi/tag.hpp>
# include <yampi/rank.hpp>
# include <yampi/communicator.hpp>
# include <yampi/cartesian.hpp>


namespace yampi
{
  namespace algorithm
  {
    template <typename SendValue, typename ReceiveValue>
    inline ::yampi::status swap(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::rank const swap_rank, ::yampi::tag const tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { return ::yampi::send_receive(send_buffer, swap_rank, tag, receive_buffer, swap_rank, tag, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    inline ::yampi::status swap(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::rank const swap_rank,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { return ::yampi::algorithm::swap(send_buffer, receive_buffer, swap_rank, ::yampi::tag(0), communicator, environment); }

    // with replacement
    template <typename Value>
    inline ::yampi::status swap(
      ::yampi::buffer<Value> buffer, ::yampi::rank const swap_rank, ::yampi::tag const tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { return ::yampi::send_receive(buffer, swap_rank, tag, swap_rank, tag, communicator, environment); }

    template <typename Value>
    inline ::yampi::status swap(
      ::yampi::buffer<Value> buffer, ::yampi::rank const swap_rank,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { return ::yampi::algorithm::swap(buffer, swap_rank, ::yampi::tag(0), communicator, environment); }

    // ignoring status
    template <typename SendValue, typename ReceiveValue>
    inline void swap(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::rank const swap_rank, ::yampi::tag const tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::send_receive(ignore_status, send_buffer, swap_rank, tag, receive_buffer, swap_rank, tag, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    inline void swap(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::rank const swap_rank,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::algorithm::swap(ignore_status, send_buffer, receive_buffer, swap_rank, ::yampi::tag(0), communicator, environment); }

    // with replacement, ignoring status
    template <typename Value>
    inline void swap(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::buffer<Value> buffer, ::yampi::rank const swap_rank, ::yampi::tag const tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::send_receive(ignore_status, buffer, swap_rank, tag, swap_rank, tag, communicator, environment); }

    template <typename Value>
    inline void swap(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::buffer<Value> buffer, ::yampi::rank const swap_rank,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::algorithm::swap(ignore_status, buffer, swap_rank, ::yampi::tag(0), communicator, environment); }

    /* Cartesian versions */
    template <typename SendValue, typename ReceiveValue>
    inline ::yampi::status swap(
      ::yampi::cartesian_shift const& shift,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::tag const tag,
      ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
    { return ::yampi::send_receive(shift, send_buffer, tag, receive_buffer, tag, cartesian, environment); }

    template <typename SendValue, typename ReceiveValue>
    inline ::yampi::status swap(
      ::yampi::cartesian_shift const& shift,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
    { return ::yampi::algorithm::swap(shift, send_buffer, receive_buffer, ::yampi::tag(0), cartesian, environment); }

    // with replacement
    template <typename Value>
    inline ::yampi::status swap(
      ::yampi::cartesian_shift const& shift,
      ::yampi::buffer<Value> buffer, ::yampi::tag const tag,
      ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
    { return ::yampi::send_receive(shift, buffer, tag, tag, cartesian, environment); }

    template <typename Value>
    inline ::yampi::status swap(
      ::yampi::cartesian_shift const& shift,
      ::yampi::buffer<Value> buffer,
      ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
    { return ::yampi::algorithm::swap(shift, buffer, ::yampi::tag(0), cartesian, environment); }

    // ignoring status
    template <typename SendValue, typename ReceiveValue>
    inline void swap(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::cartesian_shift const& shift,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::tag const tag,
      ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
    { ::yampi::send_receive(ignore_status, shift, send_buffer, tag, receive_buffer, tag, cartesian, environment); }

    template <typename SendValue, typename ReceiveValue>
    inline void swap(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::cartesian_shift const& shift,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
    { ::yampi::algorithm::swap(ignore_status, shift, send_buffer, receive_buffer, ::yampi::tag(0), cartesian, environment); }

    // with replacement, ignoring status
    template <typename Value>
    inline void swap(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::cartesian_shift const& shift,
      ::yampi::buffer<Value> buffer, ::yampi::tag const tag,
      ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
    { ::yampi::send_receive(ignore_status, shift, buffer, tag, tag, cartesian, environment); }

    template <typename Value>
    inline void swap(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::cartesian_shift const& shift,
      ::yampi::buffer<Value> buffer,
      ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
    { ::yampi::algorithm::swap(ignore_status, shift, buffer, ::yampi::tag(0), cartesian, environment); }
  }
}


#endif

