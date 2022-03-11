#ifndef YAMPI_ALGORITHM_MINMAX_ELEMENT_HPP
# define YAMPI_ALGORITHM_MINMAX_ELEMENT_HPP

# include <boost/config.hpp>

# include <array>
# include <iterator>
# include <algorithm>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

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

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_enable_if boost::enable_if_c
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  namespace algorithm
  {
    template <typename Value>
    inline
    boost::optional<
      std::pair<
        std::pair< ::yampi::rank, int >,
        std::pair< ::yampi::rank, int > > >
    minmax_element(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const root, ::yampi::datatype const& value_int_datatype,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      std::pair<Value const*, Value const*> minmax_value_ptrs
        = std::minmax_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      // min_element
      std::pair<Value, int> min_value_rank = std::make_pair(*(minmax_value_ptrs.first), present_rank.mpi_rank());
      ::yampi::reduce(
        ::yampi::make_buffer(min_value_rank, value_int_datatype), YAMPI_addressof(min_value_rank),
        ::yampi::binary_operation(::yampi::minimum_location_t()), root, communicator, environment);

      ::yampi::rank min_result_rank;
      ::yampi::scatter(
        YAMPI_addressof(min_malue_rank.second), ::yampi::make_buffer(min_result_rank.mpi_rank()),
        root, communicator, environment);

      int min_index = static_cast<int>(minmax_value_ptrs.first - buffer.data());
      if (min_result_rank != root)
        ::yampi::copy(
          ::yampi::ignore_status(),
          ::yampi::make_buffer(min_index), ::yampi::make_buffer(min_index),
          ::yampi::message_envelope(min_result_rank, root, communicator),
          environment);

      // max_element
      std::pair<Value, int> max_value_rank = std::make_pair(*(minmax_value_ptrs.second), present_rank.mpi_rank());
      ::yampi::reduce(
        ::yampi::make_buffer(max_value_rank, value_int_datatype), YAMPI_addressof(max_value_rank),
        ::yampi::binary_operation(::yampi::maximum_location_t()), root, communicator, environment);

      ::yampi::rank max_result_rank;
      ::yampi::scatter(
        YAMPI_addressof(max_value_rank.second), ::yampi::make_buffer(max_result_rank.mpi_rank()),
        root, communicator, environment);

      int max_index = static_cast<int>(minmax_value_ptrs.second - buffer.data());
      if (max_result_rank != root)
        ::yampi::copy(
          ::yampi::ignore_status(),
          ::yampi::make_buffer(max_index), ::yampi::make_buffer(max_index),
          ::yampi::message_envelope(max_result_rank, root, communicator),
          environment);

      // return
      if (present_rank == root)
        return boost::make_optional(
          std::make_pair(
            std::make_pair(min_result_rank, min_index),
            std::make_pair(max_result_rank, max_index)));

      return boost::none;
    }

    template <typename Value>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_predefined_datatype< std::pair<Value, int> >::value,
      boost::optional<
        std::pair<
          std::pair< ::yampi::rank, int >,
          std::pair< ::yampi::rank, int > > > >::type
    minmax_element(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      std::pair<Value const*, Value const*> minmax_value_ptrs
        = std::minmax_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      // min_element
      std::pair<Value, int> min_value_rank = std::make_pair(*(minmax_value_ptrs.first), present_rank.mpi_rank());
      ::yampi::reduce(
        ::yampi::make_buffer(min_value_rank), YAMPI_addressof(min_value_rank),
        ::yampi::binary_operation(::yampi::minimum_location_t()), root, communicator, environment);

      ::yampi::rank min_result_rank;
      ::yampi::scatter(
        YAMPI_addressof(min_value_rank.second), ::yampi::make_buffer(min_result_rank.mpi_rank()),
        root, communicator, environment);

      int min_index = static_cast<int>(minmax_value_ptrs.first - buffer.data());
      if (min_result_rank != root)
        ::yampi::copy(
          ::yampi::ignore_status(),
          ::yampi::make_buffer(min_index), ::yampi::make_buffer(min_index),
          ::yampi::message_envelope(min_result_rank, root, communicator),
          environment);

      // max_element
      std::pair<Value, int> max_value_rank = std::make_pair(*(minmax_value_ptrs.second), present_rank.mpi_rank());
      ::yampi::reduce(
        ::yampi::make_buffer(max_value_rank), YAMPI_addressof(max_value_rank),
        ::yampi::binary_operation(::yampi::maximum_location_t()), root, communicator, environment);

      ::yampi::rank max_result_rank;
      ::yampi::scatter(
        YAMPI_addressof(max_value_rank.second), ::yampi::make_buffer(max_result_rank.mpi_rank()),
        root, communicator, environment);

      int max_index = static_cast<int>(minmax_value_ptrs.second - buffer.data());
      if (max_result_rank != root)
        ::yampi::copy(
          ::yampi::ignore_status(),
          ::yampi::make_buffer(max_index), ::yampi::make_buffer(max_index),
          ::yampi::message_envelope(max_result_rank, root, communicator),
          environment);

      // return
      if (present_rank == root)
        return boost::make_optional(
          std::make_pair(
            std::make_pair(min_result_rank, min_index),
            std::make_pair(max_result_rank, max_index)));

      return boost::none;
    }

    template <typename Value>
    inline
    typename YAMPI_enable_if<
      not ::yampi::has_predefined_datatype< std::pair<Value, int> >::value,
      boost::optional<
        std::pair<
          std::pair< ::yampi::rank, int >,
          std::pair< ::yampi::rank, int > > > >::type
    minmax_element(
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

      return ::yampi::algorithm::minmax_element(buffer, root, value_int_datatype, communicator, environment);
    }

    template <typename Value>
    inline
    std::pair<
      std::pair< ::yampi::rank, int >,
      std::pair< ::yampi::rank, int > >
    minmax_element(
      ::yampi::buffer<Value> const buffer, ::yampi::datatype const& value_int_datatype,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      std::pair<Value const*, Value const*> minmax_value_ptrs
        = std::minmax_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      // min_element
      std::pair<Value, int> min_value_rank = std::make_pair(*(minmax_value_ptrs.first), present_rank.mpi_rank());
      ::yampi::rank min_result_rank(
        ::yampi::all_reduce(
          ::yampi::make_buffer(min_value_rank, value_int_datatype),
          ::yampi::binary_operation(::yampi::minimum_location_t()),
          communicator, environment).second);

      int min_index = static_cast<int>(minmax_value_ptrs.first - buffer.data());
      ::yampi::scatter(YAMPI_addressof(min_index), ::yampi::make_buffer(min_index), min_result_rank, communicator, environment);

      // max_element
      std::pair<Value, int> max_value_rank
        = std::make_pair(*(minmax_value_ptrs.second), present_rank.mpi_rank());
      ::yampi::rank max_result_rank(
        ::yampi::all_reduce(
          ::yampi::make_buffer(max_value_rank, value_int_datatype),
          ::yampi::binary_operation(::yampi::maximum_location_t()),
          communicator, environment).second);

      int max_index = static_cast<int>(minmax_value_ptrs.second - buffer.data());
      ::yampi::scatter(YAMPI_addressof(max_index), ::yampi::make_buffer(max_index), max_result_rank, communicator, environment);

      // return
      return std::make_pair(
        std::make_pair(min_result_rank, min_index),
        std::make_pair(max_result_rank, max_index));
    }

    template <typename Value>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_predefined_datatype< std::pair<Value, int> >::value,
      std::pair<
        std::pair< ::yampi::rank, int >,
        std::pair< ::yampi::rank, int > > >::type
    minmax_element(
      ::yampi::buffer<Value> const buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      std::pair<Value const*, Value const*> minmax_value_ptrs
        = std::minmax_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      // min_element
      std::pair<Value, int> min_value_rank
        = std::make_pair(*(minmax_value_ptrs.first), present_rank.mpi_rank());
      ::yampi::rank min_result_rank(
        ::yampi::all_reduce(
          ::yampi::make_buffer(min_value_rank),
          ::yampi::binary_operation(::yampi::minimum_location_t()),
          communicator, environment).second);

      int min_index = static_cast<int>(minmax_value_ptrs.first - buffer.data());
      ::yampi::scatter(YAMPI_addressof(min_index), ::yampi::make_buffer(min_index), min_result_rank, communicator, environment);

      // max_element
      std::pair<Value, int> max_value_rank
        = std::make_pair(*(minmax_value_ptrs.second), present_rank.mpi_rank());
      ::yampi::rank max_result_rank(
        ::yampi::all_reduce(
          ::yampi::make_buffer(max_value_rank),
          ::yampi::binary_operation(::yampi::maximum_location_t()),
          communicator, environment).second);

      int max_index = static_cast<int>(minmax_value_ptrs.second - buffer.data());
      ::yampi::scatter(YAMPI_addressof(max_index), ::yampi::make_buffer(max_index), max_result_rank, communicator, environment);

      // return
      return std::make_pair(
        std::make_pair(min_result_rank, min_index),
        std::make_pair(max_result_rank, max_index));
    }

    template <typename Value>
    inline
    typename YAMPI_enable_if<
      not ::yampi::has_predefined_datatype< std::pair<Value, int> >::value,
      std::pair<
        std::pair< ::yampi::rank, int >,
        std::pair< ::yampi::rank, int > > >::type
    minmax_element(
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

      return ::yampi::algorithm::minmax_element(buffer, value_int_datatype, communicator, environment);
    }
  }
}


# undef YAMPI_addressof
# undef YAMPI_enable_if

#endif

