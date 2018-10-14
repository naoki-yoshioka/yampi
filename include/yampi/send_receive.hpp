#ifndef YAMPI_SEND_RECEIVE_HPP
# define YAMPI_SEND_RECEIVE_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/status.hpp>
# include <yampi/error.hpp>
# include <yampi/cartesian.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    MPI_Status stat;
    int const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::status(stat);
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::rank const source,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    return ::yampi::send_receive(
      send_buffer, destination, send_tag,
      receive_buffer, source, ::yampi::any_tag(),
      communicator, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    return ::yampi::send_receive(
      send_buffer, destination, send_tag,
      receive_buffer, ::yampi::any_source(), ::yampi::any_tag(),
      communicator, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    MPI_Status stat;
    int const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          const_cast<ReceiveValue*>(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::status(stat);
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::rank const source,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    return ::yampi::send_receive(
      send_buffer, destination, send_tag,
      receive_buffer, source, ::yampi::any_tag(),
      communicator, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    return ::yampi::send_receive(
      send_buffer, destination, send_tag,
      receive_buffer, ::yampi::any_source(), ::yampi::any_tag(),
      communicator, environment);
  }


  // with replacement
  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::buffer<Value>& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    MPI_Status stat;
    int const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::status(stat);
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::buffer<Value>& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    return ::yampi::send_receive(
      buffer, destination, send_tag, source, ::yampi::any_tag(),
      communicator, environment);
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::buffer<Value>& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    return ::yampi::send_receive(
      buffer, destination, send_tag, ::yampi::any_source(), ::yampi::any_tag(),
      communicator, environment);
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::buffer<Value> const& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    MPI_Status stat;
    int const error_code
      = MPI_Sendrecv_replace(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::status(stat);
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::buffer<Value> const& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    return ::yampi::send_receive(
      buffer, destination, send_tag, source, ::yampi::any_tag(),
      communicator, environment);
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::buffer<Value> const& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    return ::yampi::send_receive(
      buffer, destination, send_tag, ::yampi::any_source(), ::yampi::any_tag(),
      communicator, environment);
  }


  // ignoring status
  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::rank const source,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    ::yampi::send_receive(
      ignore_status,
      send_buffer, destination, send_tag,
      receive_buffer, source, ::yampi::any_tag(),
      communicator, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    ::yampi::send_receive(
      ignore_status,
      send_buffer, destination, send_tag,
      receive_buffer, ::yampi::any_source(), ::yampi::any_tag(),
      communicator, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          const_cast<ReceiveValue*>(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::rank const source,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    ::yampi::send_receive(
      ignore_status,
      send_buffer, destination, send_tag,
      receive_buffer, source, ::yampi::any_tag(),
      communicator, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    ::yampi::send_receive(
      ignore_status,
      send_buffer, destination, send_tag,
      receive_buffer, ::yampi::any_source(), ::yampi::any_tag(),
      communicator, environment);
  }


  // with replacement, ignoring status
  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<Value>& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::buffer<Value>& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    ::yampi::send_receive(
      ignore_status, buffer, destination, send_tag, source, ::yampi::any_tag(),
      communicator, environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::buffer<Value>& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    ::yampi::send_receive(
      ignore_status, buffer, destination, send_tag, ::yampi::any_source(), ::yampi::any_tag(),
      communicator, environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Sendrecv_replace(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    ::yampi::send_receive(
      ignore_status, buffer, destination, send_tag, source, ::yampi::any_tag(),
      communicator, environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    ::yampi::send_receive(
      ignore_status, buffer, destination, send_tag, ::yampi::any_source(), ::yampi::any_tag(),
      communicator, environment);
  }



  /* Cartesian versions */
  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    int const direction, int const displacement,
    ::yampi::buffer<SendValue> const& send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    int const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), direction, displacement,
          YAMPI_addressof(mpi_source), YAMPI_addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::send_receive(
      send_buffer, ::yampi::rank(mpi_destination), send_tag,
      receive_buffer, ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    int const direction, int const displacement,
    ::yampi::buffer<SendValue> const& send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    return ::yampi::send_receive(
      direction, displacement,
      send_buffer, send_tag, receive_buffer, ::yampi::any_tag(),
      cartesian, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    int const direction, int const displacement,
    ::yampi::buffer<SendValue> const& send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    int const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), direction, displacement,
          YAMPI_addressof(mpi_source), YAMPI_addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::send_receive(
      send_buffer, ::yampi::rank(mpi_destination), send_tag,
      receive_buffer, ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    int const direction, int const displacement,
    ::yampi::buffer<SendValue> const& send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    return ::yampi::send_receive(
      direction, displacement,
      send_buffer, send_tag, receive_buffer, ::yampi::any_tag(),
      cartesian, environment);
  }


  // with replacement
  template <typename Value>
  inline ::yampi::status send_receive(
    int const direction, int const displacement,
    ::yampi::buffer<Value>& buffer,
    ::yampi::tag const send_tag, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    int const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), direction, displacement,
          YAMPI_addressof(mpi_source), YAMPI_addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::send_receive(
      buffer, ::yampi::rank(mpi_destination), send_tag,
      ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    int const direction, int const displacement,
    ::yampi::buffer<Value>& buffer,
    ::yampi::tag const send_tag,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    return ::yampi::send_receive(
      direction, displacement, buffer, send_tag, ::yampi::any_tag(),
      cartesian, environment);
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    int const direction, int const displacement,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::tag const send_tag, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    int const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), direction, displacement,
          YAMPI_addressof(mpi_source), YAMPI_addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::send_receive(
      buffer, ::yampi::rank(mpi_destination), send_tag,
      ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    int const direction, int const displacement,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::tag const send_tag,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    return ::yampi::send_receive(
      direction, displacement, buffer, send_tag, ::yampi::any_tag(),
      cartesian, environment);
  }


  // ignoring status
  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    int const direction, int const displacement,
    ::yampi::buffer<SendValue> const& send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    int const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), direction, displacement,
          YAMPI_addressof(mpi_source), YAMPI_addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    ::yampi::send_receive(
      ignore_status,
      send_buffer, ::yampi::rank(mpi_destination), send_tag,
      receive_buffer, ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    int const direction, int const displacement,
    ::yampi::buffer<SendValue> const& send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    ::yampi::send_receive(
      ignore_status, direction, displacement,
      send_buffer, send_tag, receive_buffer, ::yampi::any_tag(),
      cartesian, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    int const direction, int const displacement,
    ::yampi::buffer<SendValue> const& send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    int const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), direction, displacement,
          YAMPI_addressof(mpi_source), YAMPI_addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    ::yampi::send_receive(
      ignore_status,
      send_buffer, ::yampi::rank(mpi_destination), send_tag,
      receive_buffer, ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    int const direction, int const displacement,
    ::yampi::buffer<SendValue> const& send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    ::yampi::send_receive(
      ignore_status, direction, displacement,
      send_buffer, send_tag, receive_buffer, ::yampi::any_tag(),
      cartesian, communicator);
  }


  // with replacement, ignoring status
  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    int const direction, int const displacement,
    ::yampi::buffer<Value>& buffer,
    ::yampi::tag const send_tag, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    int const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), direction, displacement,
          YAMPI_addressof(mpi_source), YAMPI_addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    ::yampi::send_receive(
      ignore_status,
      buffer, ::yampi::rank(mpi_destination), send_tag,
      ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    int const direction, int const displacement,
    ::yampi::buffer<Value>& buffer,
    ::yampi::tag const send_tag,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    ::yampi::send_receive(
      ignore_status, direction, displacement,
      buffer, send_tag, ::yampi::any_tag(),
      cartesian, environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    int const direction, int const displacement,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::tag const send_tag, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    int const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), direction, displacement,
          YAMPI_addressof(mpi_source), YAMPI_addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    ::yampi::send_receive(
      ignore_status,
      buffer, ::yampi::rank(mpi_destination), send_tag,
      ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    int const direction, int const displacement,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::tag const send_tag,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    ::yampi::send_receive(
      ignore_status, direction, displacement,
      buffer, send_tag, ::yampi::any_tag(),
      cartesian, environment);
  }
}


# undef YAMPI_addressof

#endif

