#ifndef YAMPI_ALGORITHM_MIN_ELEMENT_HPP
# define YAMPI_ALGORITHM_MIN_ELEMENT_HPP

# include <array>
# include <iterator>
# include <algorithm>
# include <utility>
# include <type_traits>
# include <memory>

# include <boost/optional.hpp>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/reduce.hpp>
# include <yampi/scatter.hpp>
# include <yampi/all_reduce.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/status.hpp>
# include <yampi/message_envelope.hpp>
# include <yampi/datatype.hpp>
# include <yampi/predefined_datatype.hpp>
# include <yampi/has_predefined_datatype.hpp>
# include <yampi/addressof.hpp>
# include <yampi/byte_displacement.hpp>
# include <yampi/algorithm/copy.hpp>

# include <mpi.h>


namespace yampi
{
  namespace algorithm
  {
    template <typename Value>
    inline boost::optional< std::pair< ::yampi::rank, int > > min_element(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const root,
      ::yampi::datatype const& value_int_datatype,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      Value const* min_value_ptr = std::min_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      std::pair<Value, int> value_rank = std::make_pair(*min_value_ptr, present_rank.mpi_rank());
      ::yampi::reduce(
        ::yampi::make_buffer(value_rank, value_int_datatype), std::addressof(value_rank),
        ::yampi::binary_operation(::yampi::minimum_location_t()), root, communicator, environment);

      ::yampi::rank result_rank;
      ::yampi::scatter(
        std::addressof(value_rank.second), ::yampi::make_buffer(result_rank.mpi_rank()),
        root, communicator, environment);

      int index = static_cast<int>(min_value_ptr - buffer.data());
      if (result_rank != root)
        ::yampi::copy(
          ::yampi::ignore_status,
          ::yampi::make_buffer(index), ::yampi::make_buffer(index),
          ::yampi::message_envelope(result_rank, root, communicator),
          environment);

      if (present_rank == root)
        return boost::make_optional(std::make_pair(result_rank, index));

      return boost::none;
    }

    template <typename Value>
    inline
    typename std::enable_if<
      ::yampi::has_predefined_datatype< std::pair<Value, int> >::value,
      boost::optional< std::pair< ::yampi::rank, int > > >::type
    min_element(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      Value const* min_value_ptr = std::min_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      std::pair<Value, int> value_rank = std::make_pair(*min_value_ptr, present_rank.mpi_rank());
      ::yampi::reduce(
        ::yampi::make_buffer(value_rank), std::addressof(value_rank),
        ::yampi::binary_operation(::yampi::minimum_location_t()), root, communicator, environment);

      ::yampi::rank result_rank;
      ::yampi::scatter(
        std::addressof(value_rank.second), ::yampi::make_buffer(result_rank.mpi_rank()),
        root, communicator, environment);

      int index = static_cast<int>(min_value_ptr - buffer.data());
      if (result_rank != root)
        ::yampi::copy(
          ::yampi::ignore_status,
          ::yampi::make_buffer(index), ::yampi::make_buffer(index),
          ::yampi::message_envelope(result_rank, root, communicator),
          environment);

      if (present_rank == root)
        return boost::make_optional(std::make_pair(result_rank, index));

      return boost::none;
    }

    template <typename Value>
    inline
    typename std::enable_if<
      not ::yampi::exists_basic_datatype_tag< std::pair<Value, int> >::value,
      boost::optional< std::pair< ::yampi::rank, int > > >::type
    min_element(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      std::array<MPI_Datatype, 2u> mpi_datatypes
        = { buffer.datatype().mpi_datatype(), ::yampi::predefined_datatype<int>().mpi_datatype() };
      std::pair<Value, int> value_int;
      std::array< ::yampi::byte_displacement, 2u > byte_displacements
        = { ::yampi::addressof(value_int.first, environment) - ::yampi::addressof(value_int, environment),
            ::yampi::addressof(value_int.second, environment) - ::yampi::addressof(value_int, environment) };
      std::array<int, 2u> block_lengths = { 1, 1 };
      ::yampi::datatype value_int_datatype(
        mpi_datatypes.begin(), mpi_datatypes.end(),
        byte_displacements.begin(), block_lengths.begin(), environment);

      return ::yampi::algorithm::min_element(buffer, root, value_int_datatype, communicator, environment);
    }

    template <typename Value>
    inline std::pair< ::yampi::rank, int > min_element(
      ::yampi::buffer<Value> const buffer, ::yampi::datatype const& value_int_datatype,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      Value const* min_value_ptr
        = std::min_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      std::pair<Value, int> value_rank
        = std::make_pair(*min_value_ptr, present_rank.mpi_rank());
      ::yampi::rank result_rank(
        ::yampi::all_reduce(
          ::yampi::make_buffer(value_rank, value_int_datatype),
          ::yampi::binary_operation(::yampi::minimum_location_t()),
          communicator, environment).second);

      int index = static_cast<int>(min_value_ptr - buffer.data());
      ::yampi::scatter(std::addressof(index), ::yampi::make_buffer(index), result_rank, communicator, environment);

      return std::make_pair(result_rank, index);
    }

    template <typename Value>
    inline
    typename std::enable_if<
      ::yampi::has_predefined_datatype< std::pair<Value, int> >::value,
      std::pair< ::yampi::rank, int > >::type
    min_element(
      ::yampi::buffer<Value> const buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      Value const* min_value_ptr
        = std::min_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      std::pair<Value, int> value_rank
        = std::make_pair(*min_value_ptr, present_rank.mpi_rank());
      ::yampi::rank result_rank(
        ::yampi::all_reduce(
          ::yampi::make_buffer(value_rank),
          ::yampi::binary_operation(::yampi::minimum_location_t()),
          communicator, environment).second);

      int index = static_cast<int>(min_value_ptr - buffer.data());
      ::yampi::scatter(std::addressof(index), ::yampi::make_buffer(index), result_rank, communicator, environment);

      return std::make_pair(result_rank, index);
    }

    template <typename Value>
    inline
    typename std::enable_if<
      not ::yampi::has_predefined_datatype< std::pair<Value, int> >::value,
      std::pair< ::yampi::rank, int > >::type
    min_element(
      ::yampi::buffer<Value> const buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      std::array<MPI_Datatype, 2u> mpi_datatypes
        = { buffer.datatype().mpi_datatype(), ::yampi::predefined_datatype<int>().mpi_datatype() };
      std::pair<Value, int> value_int;
      std::array< ::yampi::byte_displacement, 2u > byte_displacements
        = { ::yampi::addressof(value_int.first, environment) - ::yampi::addressof(value_int, environment),
            ::yampi::addressof(value_int.second, environment) - ::yampi::addressof(value_int, environment) };
      std::array<int, 2u> block_lengths = { 1, 1 };
      ::yampi::datatype value_int_datatype(
        mpi_datatypes.begin(), mpi_datatypes.end(),
        byte_displacements.begin(), block_lengths.begin(), environment);

      return ::yampi::algorithm::min_element(buffer, value_int_datatype, communicator, environment);
    }
  }
}


#endif

