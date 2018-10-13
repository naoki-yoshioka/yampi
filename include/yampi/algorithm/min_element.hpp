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
# include <yampi/algorithm/copy.hpp>
# include <yampi/algorithm/ranked_buffer.hpp>

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
      ::yampi:communicator const& communicator, ::yampi::rank const root,
      ::yampi::environment const& environment,
      ::yampi::buffer<Value> const& buffer)
    {
      Value const* min_value_ptr
        = std::min_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      std::pair<Value, int> value_rank
        = std::make_pair(*min_value_ptr, present_rank.mpi_rank());
      ::yampi::reduce(communicator, root).call(
        environment,
        ::yampi::make_buffer(
          value_rank,
          ::yampi::datatype(::yampi::basic_datatype_tag_of< std::pair<Value, int> >::call())),
        YAMPI_addressof(value_rank),
        ::yampi::binary_operation(::yampi::minimum_location_t()));

      ::yampi::rank result_rank;
      ::yampi::datatype int_datatype(::yampi::int_datatype_t());
      ::yampi::scatter(communicator, root).call(
        environment,
        YAMPI_addressof(value_rank.second),
        ::yampi::make_buffer(result_rank.mpi_rank(), int_datatype));

      int index = static_cast<int>(min_ptr - buffer.data());
      if (result_rank != root)
        ::yampi::copy(
          ::yampi::ignore_status(), communicator, environment,
          ::yampi::make_ranked_buffer(index, int_datatype, result_rank),
          ::yampi::make_ranked_buffer(index, int_datatype, root_rank));

      if (present_rank == root)
        return boost::make_optional(std::make_pair(root, index));

      return boost::none;
    }

    template <typename Value>
    inline
    boost::optional< std::pair< ::yampi::rank, int > >
    min_element(
      ::yampi:communicator const& communicator, ::yampi::rank const root,
      ::yampi::environment const& environment,
      ::yampi::buffer<Value> const& buffer,
      ::yampi::datatype const& value_int_datatype)
    {
      Value const* min_value_ptr
        = std::min_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      std::pair<Value, int> value_rank
        = std::make_pair(*min_value_ptr, present_rank.mpi_rank());
      ::yampi::reduce(communicator, root).call(
        environment,
        ::yampi::make_buffer(value_rank, value_int_datatype),
        YAMPI_addressof(value_rank),
        ::yampi::binary_operation(::yampi::minimum_location_t()));

      ::yampi::rank result_rank;
      ::yampi::datatype int_datatype(::yampi::int_datatype_t());
      ::yampi::scatter(communicator, root).call(
        environment,
        YAMPI_addressof(value_rank.second),
        ::yampi::make_buffer(result_rank.mpi_rank(), int_datatype));

      int index = static_cast<int>(min_ptr - buffer.data());
      if (result_rank != root)
        ::yampi::copy(
          ::yampi::ignore_status(), communicator, environment,
          ::yampi::make_ranked_buffer(index, int_datatype, result_rank),
          ::yampi::make_ranked_buffer(index, int_datatype, root_rank));

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
      ::yampi:communicator const& communicator, ::yampi::environment const& environment,
      ::yampi::buffer<Value> const& buffer)
    {
      Value const* min_value_ptr
        = std::min_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      std::pair<Value, int> value_rank
        = std::make_pair(*min_value_ptr, present_rank.mpi_rank());
      ::yampi::rank result_rank(
        ::yampi::all_reduce(
          communicator, environment,
          ::yampi::make_buffer(
            value_rank,
            ::yampi::datatype(::yampi::basic_datatype_tag_of< std::pair<Value, int> >::call())),
          ::yampi::binary_operation(::yampi::minimum_location_t())).second);

      int index = static_cast<int>(min_ptr - buffer.data());
      ::yampi::scatter(communicator, result_rank).call(
        environment, YAMPI_addressof(index),
        ::yampi::make_buffer(index, ::yampi::datatype(::yampi::int_datatype_t())));

      return std::make_pair(result_rank, index);
    }

    template <typename Value>
    inline std::pair< ::yampi::rank, int > min_element(
      ::yampi:communicator const& communicator, ::yampi::environment const& environment,
      ::yampi::buffer<Value> const& buffer,
      ::yampi::datatype const& value_int_datatype)
    {
      Value const* min_value_ptr
        = std::min_element(buffer.data(), buffer.data() + buffer.count());
      ::yampi::rank const present_rank = communicator.rank(environment);

      std::pair<Value, int> value_rank
        = std::make_pair(*min_value_ptr, present_rank.mpi_rank());
      ::yampi::rank result_rank(
        ::yampi::all_reduce(
          communicator, environment,
          ::yampi::make_buffer(value_rank, value_int_datatype),
          ::yampi::binary_operation(::yampi::minimum_location_t())).second);

      int index = static_cast<int>(min_ptr - buffer.data());
      ::yampi::scatter(communicator, result_rank).call(
        environment, YAMPI_addressof(index),
        ::yampi::make_buffer(index, ::yampi::datatype(::yampi::int_datatype_t())));

      return std::make_pair(result_rank, index);
    }
  }
}


# undef YAMPI_addressof
# undef YAMPI_enable_if

#endif

