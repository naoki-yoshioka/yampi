#ifndef YAMPI_ALGORITHM_MIN_ELEMENT_HPP
# define YAMPI_ALGORITHM_MIN_ELEMENT_HPP

# include <boost/config.hpp>

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
# include <yampi/datatype.hpp>
# include <yampi/exits_basic_datatype_tag.hpp>
# include <yampi/basic_datatype_tag_of.hpp>
# include <yampi/message_envelope.hpp>
# include <yampi/algorithm/copy.hpp>

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
    typename YAMPI_enable_if<
      ::yampi::exists_basic_datatype_tag< std::pair<Value, int> >::value,
      boost::optional< std::pair< ::yampi::rank, int > > >::type
    min_element(
      ::yampi::buffer<Value> const& buffer,
      ::yampi::rank const& root,
      ::yampi:communicator const& communicator,
      ::yampi::environment const& environment)
    {
      Value const* min_value_ptr
        = std::min_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      std::pair<Value, int> value_rank
        = std::make_pair(*min_value_ptr, present_rank.mpi_rank());
      ::yampi::reduce(root, communicator).call(
        ::yampi::make_buffer(
          value_rank,
          ::yampi::datatype(::yampi::basic_datatype_tag_of< std::pair<Value, int> >::call())),
        YAMPI_addressof(value_rank),
        ::yampi::binary_operation(::yampi::minimum_location_t()),
        environment);

      ::yampi::rank result_rank;
      ::yampi::datatype int_datatype(::yampi::int_datatype_t());
      ::yampi::scatter(root, communicator).call(
        YAMPI_addressof(value_rank.second),
        ::yampi::make_buffer(result_rank.mpi_rank(), int_datatype),
        environment);

      int index = static_cast<int>(min_ptr - buffer.data());
      if (result_rank != root)
        ::yampi::copy(
          ::yampi::ignore_status(),
          ::yampi::make_buffer(index, int_datatype),
          ::yampi::make_buffer(index, int_datatype),
          ::yampi::message_envelope(result_rank, root, communicator),
          environment);

      if (present_rank == root)
        return boost::make_optional(std::make_pair(root, index));

      return boost::none;
    }

    template <typename Value>
    inline
    boost::optional< std::pair< ::yampi::rank, int > >
    min_element(
      ::yampi::buffer<Value> const& buffer,
      ::yampi::rank const& root,
      ::yampi::datatype const& value_int_datatype,
      ::yampi:communicator const& communicator,
      ::yampi::environment const& environment)
    {
      Value const* min_value_ptr
        = std::min_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      std::pair<Value, int> value_rank
        = std::make_pair(*min_value_ptr, present_rank.mpi_rank());
      ::yampi::reduce(root, communicator).call(
        ::yampi::make_buffer(value_rank, value_int_datatype),
        YAMPI_addressof(value_rank),
        ::yampi::binary_operation(::yampi::minimum_location_t()),
        environment);

      ::yampi::rank result_rank;
      ::yampi::datatype int_datatype(::yampi::int_datatype_t());
      ::yampi::scatter(root, communicator).call(
        YAMPI_addressof(value_rank.second),
        ::yampi::make_buffer(result_rank.mpi_rank(), int_datatype),
        environment);

      int index = static_cast<int>(min_ptr - buffer.data());
      if (result_rank != root)
        ::yampi::copy(
          ::yampi::ignore_status(),
          ::yampi::make_buffer(index, int_datatype),
          ::yampi::make_buffer(index, int_datatype),
          ::yampi::message_envelope(result_rank, root, communicator),
          environment);

      if (present_rank == root)
        return boost::make_optional(std::make_pair(root, index));

      return boost::none;
    }


    template <typename Value>
    inline
    typename YAMPI_enable_if<
      ::yampi::exists_basic_datatype_tag< std::pair<Value, int> >::value,
      std::pair< ::yampi::rank, int > >::type
    min_element(
      ::yampi::buffer<Value> const& buffer,
      ::yampi:communicator const& communicator, ::yampi::environment const& environment)
    {
      Value const* min_value_ptr
        = std::min_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      std::pair<Value, int> value_rank
        = std::make_pair(*min_value_ptr, present_rank.mpi_rank());
      ::yampi::rank result_rank(
        ::yampi::all_reduce(
          ::yampi::make_buffer(
            value_rank,
            ::yampi::datatype(::yampi::basic_datatype_tag_of< std::pair<Value, int> >::call())),
          ::yampi::binary_operation(::yampi::minimum_location_t())
          communicator, environment).second);

      int index = static_cast<int>(min_ptr - buffer.data());
      ::yampi::scatter(result_rank, communicator).call(
        YAMPI_addressof(index),
        ::yampi::make_buffer(index, ::yampi::datatype(::yampi::int_datatype_t())),
        environment);

      return std::make_pair(result_rank, index);
    }

    template <typename Value>
    inline std::pair< ::yampi::rank, int > min_element(
      ::yampi::buffer<Value> const& buffer,
      ::yampi::datatype const& value_int_datatype,
      ::yampi:communicator const& communicator, ::yampi::environment const& environment)
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

      int index = static_cast<int>(min_ptr - buffer.data());
      ::yampi::scatter(result_rank, communicator).call(
        YAMPI_addressof(index),
        ::yampi::make_buffer(index, ::yampi::datatype(::yampi::int_datatype_t())),
        environment);

      return std::make_pair(result_rank, index);
    }
  }
}


# undef YAMPI_addressof
# undef YAMPI_enable_if

#endif

