#ifndef YAMPI_ALGORITHM_COUNT_HPP
# define YAMPI_ALGORITHM_COUNT_HPP

# include <boost/config.hpp>

# include <iterator>
# include <algorithm>
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
# include <yampi/all_reduce.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/datatype.hpp>
# include <yampi/basic_datatype_tag_of.hpp>
# if MPI_VERSION >= 3
#   include <yampi/request.hpp>
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
    boost::optional<typename std::iterator_traits<Value const*>::difference_type>
    count(
      ::yampi::buffer<Value> const& buffer,
      ::yampi::rank const& root,
      Value const& value,
      ::yampi:communicator const& communicator,
      ::yampi::environment const& environment)
    {
      typedef typename std::iterator_traits<Value const*>::difference_type count_type;
      count_type result
        = std::count(buffer.data(), buffer.data() + buffer.count(), value);

      ::yampi::reduce const reducer(root, communicator);

      ::yampi::datatype count_datatype(::yampi::basic_datatype_tag_of<count_type>::call());
      if (communicator.rank(environment) == root)
      {
        reducer.call(
          ::yampi::make_buffer(result, count_datatype), YAMPI_addressof(result),
          ::yampi::binary_operation(::yampi::plus_t()), environment);
        return boost::make_optional(result);
      }

      reducer.call(
        ::yampi::make_buffer(result, count_datatype),
        ::yampi::binary_operation(::yampi::plus_t()), environment);
      return boost::none;
    }

    template <typename Value>
    inline typename std::iterator_traits<Value const*>::difference_type
    count(
      ::yampi::buffer<Value> const& buffer, Value const& value,
      ::yampi:communicator const& communicator, ::yampi::environment const& environment)
    {
      return ::yampi::all_reduce(
        ::yampi::make_buffer(
          std::count(buffer.data(), buffer.data() + buffer.count(), value),
          ::yampi::datatype(::yampi::basic_datatype_tag_of<count_type>::call())),
        ::yampi::binary_operation(::yampi::plus_t()),
        communicator, environment);
    }
# if MPI_VERSION >= 3


    template <typename Value>
    inline
    boost::optional<typename std::iterator_traits<Value const*>::difference_type>
    count(
      ::yampi::request& request,
      ::yampi::buffer<Value> const& buffer,
      ::yampi::rank const& root,
      Value const& value,
      ::yampi:communicator const& communicator,
      ::yampi::environment const& environment)
    {
      typedef typename std::iterator_traits<Value const*>::difference_type count_type;
      count_type result
        = std::count(buffer.data(), buffer.data() + buffer.count(), value);

      ::yampi::reduce const reducer(root, communicator);

      ::yampi::datatype count_datatype(::yampi::basic_datatype_tag_of<count_type>::call());
      if (communicator.rank(environment) == root)
      {
        reducer.call(
          request,
          ::yampi::make_buffer(result, count_datatype), YAMPI_addressof(result),
          ::yampi::binary_operation(::yampi::plus_t()), environment);
        return boost::make_optional(result);
      }

      reducer.call(
        request,
        ::yampi::make_buffer(result, count_datatype),
        ::yampi::binary_operation(::yampi::plus_t()), environment);
      return boost::none;
    }

    template <typename Value>
    inline typename std::iterator_traits<Value const*>::difference_type
    count(
      ::yampi::request& request,
      ::yampi::buffer<Value> const& buffer, Value const& value,
      ::yampi:communicator const& communicator, ::yampi::environment const& environment)
    {
      return ::yampi::all_reduce(
        request,
        ::yampi::make_buffer(
          std::count(buffer.data(), buffer.data() + buffer.count(), value),
          ::yampi::datatype(::yampi::basic_datatype_tag_of<count_type>::call())),
        ::yampi::binary_operation(::yampi::plus_t()),
        communicator, environment);
    }
# endif
  }
}


# undef YAMPI_addressof

#endif

